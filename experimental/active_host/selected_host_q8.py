"""Explicit, disabled-by-default selected-latent CUDA prototype.

Importing this file does not import Torch or initialize CUDA. This is a kernel
primitive, not an installed engine patch. The caller owns host-page identity,
copy completion, graph lifetime, cancellation and the unchanged GPU indexer.
"""
from __future__ import annotations


class UnpublishedStepError(RuntimeError):
    """Completion was observed, but guarded failure/cancellation forbids output."""


class SelectedHostQ8:
    def __init__(self, extension, host_q, host_s, *, device_index,
                 max_host_bytes, max_staging_bytes, allow_experimental=False):
        if not allow_experimental:
            raise RuntimeError("active host KV prototype is disabled by default")
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("allocate source aliases and fixed buffers before capture")
        if (host_q.device.type != "cpu" or host_s.device.type != "cpu" or
                host_q.dtype != torch.int32 or host_s.dtype != torch.float16 or
                host_q.ndim != 3 or tuple(host_q.shape[1:]) != (256, 128) or
                tuple(host_s.shape) != (host_q.shape[0], 256, 16) or
                not host_q.is_contiguous() or not host_s.is_contiguous() or
                not host_q.is_pinned() or not host_s.is_pinned()):
            raise ValueError("pinned packed Q8 [pages,256,128]/scales[pages,256,16] required")
        host_bytes = host_q.numel() * 4 + host_s.numel() * 2
        # This per-layer primitive cannot authorize an eleven-layer aggregate
        # budget; HostPageOwner must reserve that total before these allocations.
        if host_bytes > max_host_bytes:
            raise ValueError("source storage exceeds declared per-layer host budget")
        self.torch, self.extension = torch, extension
        self.device = torch.device("cuda", device_index)
        self.host_q, self.host_s = host_q, host_s
        self.q_alias = extension.pinned_alias(host_q, device_index)
        self.s_alias = extension.pinned_alias(host_s, device_index)
        max_indices = 8 * 2080
        self.pool_pages = (max_indices + 255) // 256
        self.hash_size = 1 << (2 * max_indices - 1).bit_length()
        required = (self.pool_pages * 256 * 544 + max_indices * 4 +
                    self.hash_size * 4 * 2 + self.pool_pages * 4 + 16)
        if required > max_staging_bytes:
            raise ValueError("fixed staging buffers exceed declared device budget")
        self.staging_bytes = required
        self.hot_q = torch.empty((self.pool_pages, 256, 128), dtype=torch.int32, device=self.device)
        self.hot_s = torch.empty((self.pool_pages, 256, 16), dtype=torch.float16, device=self.device)
        self.output_indices = torch.empty(max_indices, dtype=torch.int32, device=self.device)
        self.hash_keys = torch.empty(self.hash_size, dtype=torch.int32, device=self.device)
        self.hash_values = torch.empty_like(self.hash_keys)
        self.error = torch.empty(1, dtype=torch.int32, device=self.device)
        self.metrics = torch.empty(3, dtype=torch.int32, device=self.device)
        self.block_table = torch.arange(self.pool_pages, dtype=torch.int32,
                                       device=self.device).unsqueeze(0)
        self.rope_dummy = torch.empty((self.pool_pages, 256, 1, 0),
                                      dtype=torch.float16, device=self.device)
        self._event = None
        self._cancelled = False
        self._closed = False

    def launch_into(self, indices, *, source_table, slot_generations,
                    expected_generations, row_visible_limits):
        """Fixed-shape launch primitive; warm before capture, then may capture.

        There is no .item(), CPU copy, synchronization, host allocation or
        data-dependent output shape. Capture/replay must hold an external source
        and hot-buffer lease. No concurrent launches on this object's buffers.
        """
        if self._closed:
            raise RuntimeError("stager closed")
        if indices.ndim != 2 or not 1 <= indices.shape[0] <= 8:
            raise ValueError("only Q1..Q8 verification is supported")
        remapped = self.output_indices[:indices.numel()].view(indices.shape)
        self.hash_keys.fill_(-1)
        self.hash_values.fill_(-1)
        self.error.zero_()
        self.metrics.zero_()
        self.extension.selected_host_q8(
            self.q_alias, self.s_alias, source_table, slot_generations,
            expected_generations, row_visible_limits, indices,
            self.hot_q, self.hot_s, remapped, self.hash_keys, self.hash_values,
            self.error, self.metrics)
        return remapped

    def begin_eager(self, indices, **metadata):
        """Convenience eager lease; call record_completion AFTER downstream DSA."""
        if self._event is not None:
            raise RuntimeError("previous eager lease has not been drained")
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("use launch_into with an external lease inside capture")
        self._event = self.torch.cuda.Event()
        self._cancelled = False
        try:
            return self.launch_into(indices, **metadata)
        except BaseException:
            # A launch may have partially enqueued work. Preserve source/hot
            # ownership and fence it; the caller must drain before retry/close.
            self.record_completion()
            self._cancelled = True
            raise

    def record_completion(self):
        if self._event is None:
            raise RuntimeError("no eager read lease")
        self._event.record(self.torch.cuda.current_stream(self.device))

    def cancel(self):
        if self._event is None:
            raise RuntimeError("no eager read lease")
        self._cancelled = True

    def finish_eager(self):
        """Check an already completed event, then permit output publication/reuse."""
        if self._event is None or not self._event.query():
            raise RuntimeError("completion has not been observed; no reuse permitted")
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("completion/error publication is outside capture")
        flags = int(self.error.item())
        counts = tuple(int(v) for v in self.metrics.cpu().tolist())
        cancelled = self._cancelled
        self._event = None
        self._cancelled = False
        if flags or cancelled:
            raise UnpublishedStepError(f"unpublished host step: error_mask={flags}, cancelled={cancelled}")
        return {"unique_rows": counts[0], "valid_selections": counts[1],
                "hash_probes": counts[2], "copied_packed_bytes": counts[0] * 544}

    def close(self):
        if self._event is not None:
            raise RuntimeError("cannot release aliases/buffers with an undrained eager lease")
        self._closed = True
