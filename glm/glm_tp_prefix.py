"""Opt-in, previous-request prefix reuse for the single-sequence DFlash ring.

Target KV pages and recurrent checkpoints remain owned by EXL3. Only the draft's
2048-token SWA history is copied to host RAM at matching native checkpoints.
No engine files or running processes are modified by importing this module.

Experimental TP extension: target states remain in EXL3's per-rank checkpoint
registry. Only the single-device draft ring is copied here. CPU contract tests
do not qualify this extension for GPU deployment.
"""
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path


ENGINE_SOURCE_PINS = {'exllamav3.cache.recurrent': '43c7480cf82d5432697ae1a78f1f462960564e0b17e40d598d65c03443bdcecd', 'exllamav3.generator.pagetable': '0cde876b9969df7120ed7137b35bff3040dc3a7613c9f5a9547e58da4a9849ea', 'exllamav3.generator.job': '22217d6405b1ae0507c7c2a607e17fc64ab56c6ce542ae74476b6f7205ecb200', 'exllamav3.generator.generator': '0e6fa14e210a92b093b08fe107f2c78a87eae230b05f569fdbce36bf2f90f178', 'exllamav3.modules.gated_delta_net': '779a665fbf35dea7e7a098b3e4304a9008fc8270a8f0be6e709f68ff93afe59b', 'exllamav3.model.model_tp_fn': '4984e3a18e65eae039633ee5c150773a66f7b08e8ac8515e3fb16324ed2dc974', 'exllamav3.generator.cpu_cache': 'd3697bcc9f7eb022db1f5a66399c03f9377138c8f7c44a8a29c416ed206ab608', 'exllamav3.util.pinned_arena': '6d8b9aabea21302eadfac75d477833a1c7d6c97a4b6c10df8b3eddf0ee42ac87', 'exllamav3.util.tp_lazy_cache': '2851996beae98ef391601e9cf04f0042b198448d94675ca661962e7bf6ded863'}


def validate_engine_sources():
    for name, expected in ENGINE_SOURCE_PINS.items():
        path = getattr(importlib.import_module(name), "__file__", None)
        if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"TP paired retention requires pinned engine source: {name}")


PAGE_SIZE = 256
WINDOW_TOKENS = 2048


@dataclass(frozen=True)
class DraftCheckpoint:
    position: int
    key: bytes
    prefix: object
    page_indices: tuple
    tensors: tuple

    @property
    def host_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.prefix, *self.tensors))


class PreviousRequestCache:
    """Keep the prompt-end checkpoint plus the latest decode checkpoint.

    Requests run serially. A miss invalidates the previous request entirely;
    this deliberately does not implement a multi-session cache. Missing target
    pages/checkpoints, changed tokens and cancellations all fall back to cold PP.
    """
    def __init__(self, generator):
        self.generator = generator
        cache = generator.draft_cache
        ring = getattr(cache, "dflash_ring_tokens", 0)
        if (not generator.dflash_draft or generator.max_batch_size != 1
                or generator.recurrent_cache is None
                or ring < WINDOW_TOKENS + generator.max_chunk_size + 8
                or ring % PAGE_SIZE):
            raise ValueError("Prefix reuse requires a single-sequence recurrent target and DFlash GPU ring")
        if not generator.model.loaded_tp:
            raise ValueError("TP paired retention requires an actual TP target")
        self._validate_tp_contract()
        self._owners = (generator.model, generator.cache, generator.draft_cache, generator.recurrent_cache)
        self._cache_owner_serial = getattr(generator.cache, "owner_serial", None)
        self.ring_pages = ring // PAGE_SIZE
        self.storage = tuple(cache.get_all_tensors())
        if not self.storage or any(t is None or t.shape[0] != self.ring_pages
                                   or t.shape[1] != PAGE_SIZE for t in self.storage):
            raise ValueError("Unsupported draft ring tensor layout")
        self.published = ()
        self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
        self.job = None
        self.hits = self.misses = self.restored_tokens = 0
        self.last_reason = "empty"

    def _validate_tp_contract(self):
        g = self.generator
        model, target_cache = g.model, g.cache
        state_cls = getattr(target_cache, "recurrent_state_cls", None)
        if (getattr(target_cache, "model", None) is not model
                or getattr(g.recurrent_cache, "model", None) is not model
                or getattr(g.draft_cache.model, "loaded_tp", False)
                or not callable(getattr(model, "tp_drain_acks", None))
                or not callable(getattr(model, "tp_dispatch_all", None))
                or getattr(g.draft_model.config, "experimental_tensor_parallel", False) is not True
                or not callable(getattr(target_cache, "new_from_stashed", None))
                or not all(callable(getattr(state_cls, name, None))
                           for name in ("stash", "unstash", "tp_export"))):
            raise ValueError("TP prefix reuse requires native per-rank recurrent checkpoint custody")

    def _tp_fence(self):
        g = self.generator
        if (any(current is not owned for current, owned in zip(
                    (g.model, g.cache, g.draft_cache, g.recurrent_cache), self._owners))
                or getattr(g.cache, "owner_serial", None) != self._cache_owner_serial):
            raise ValueError("TP paired cache lost its original generator/model/cache ownership")
        if self.generator.model.loaded_tp:
            self.generator.model.tp_drain_acks()

    def _paired_stash(self, key, position):
        stash = self.generator.recurrent_cache.get(key)
        if (stash is None or stash.get("position") != position
                or type(stash.get("checkpoint_size")) is not int
                or stash["checkpoint_size"] < 0):
            return None
        if self.generator.model.loaded_tp:
            handle = stash.get("tp_handle")
            if not isinstance(handle, int) or isinstance(handle, bool) or handle < 0:
                return None
        return stash

    def invalidate(self, reason):
        self.published = ()
        self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
        self.job = None
        self.last_reason = reason

    def _cold(self):
        self._tp_fence()
        g = self.generator
        g.pagetable.reset_page_table()
        g.recurrent_cache.prune_stranded()

    def begin(self, job):
        import torch
        self._tp_fence()
        # The ring patch already rejects parallel jobs. Refuse to reuse states
        # for features whose rewind/requeue semantics this integration doesn't cover.
        supported = (len(job.sequences) == 1 and not job.is_requeued
                     and not job.embeddings and not job.filters and not job.banned_strings
                     and job.prefix_token is None and job.orig_max_rq_tokens is None)
        seq = job.sequences[0]
        ids = seq.sequence_ids.torch()
        candidates = self.published
        self.published = ()
        self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
        self.job = job if supported else None
        if supported:
            for cp in sorted(candidates, key=lambda cp: cp.position, reverse=True):
                stash = self._paired_stash(cp.key, cp.position)
                if (cp.position < ids.shape[-1] and stash is not None
                        and stash["position"] == cp.position
                        and self.generator.pagetable.is_resumable(cp.key)
                        and torch.equal(ids[:, :cp.position], cp.prefix)):
                    self.plan = cp
                    # Keep the restored boundary available even if no new native
                    # checkpoint is reached in this very short follow-up.
                    self.prompt_checkpoint = cp
                    self.last_reason = "matched"
                    break
        if self.plan is None:
            self.misses += 1
            self.last_reason = "prefix_or_state_miss" if supported else "unsupported_request"
            self._cold()

    def restrict(self, job):
        if job is not self.job:
            return
        # Cap native KV/recurrent reuse to the boundary for which draft state is
        # also available. Never let EXL3 restore a newer, unpaired checkpoint.
        pages = self.plan.position // PAGE_SIZE if self.plan else 0
        seq = job.sequences[0]
        seq.max_cached_pages = pages
        job.all_unique_hashes = list(set(seq.page_hashes[:pages]))

    def restore(self, job, allocate):
        if job is not self.job or self.plan is None:
            return
        cp = self.plan
        self._tp_fence()
        seq = job.sequences[0]
        if (seq.kv_position != cp.position or job.recurrent_state is None
                or job.recurrent_state.position != cp.position
                or job.cached_pages * PAGE_SIZE != cp.position):
            # KV eviction/allocation must not silently resume from an older
            # target checkpoint with a newer draft ring. Reallocate cold instead.
            job.deallocate_pages()
            self._cold()
            self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
            seq.kv_position = 0
            seq.prefill_complete = False
            seq.max_cached_pages = 0
            job.all_unique_hashes = []
            job.cached_pages = job.cached_tokens = job.total_pages = job.non_sequential_pages = 0
            job.last_recurrent_checkpoint_pos = None
            self.misses += 1
            self.last_reason = "allocation_miss"
            allocate(job)
            return
        for dst, saved in zip(self.storage, cp.tensors):
            for row, index in enumerate(cp.page_indices):
                dst[index].copy_(saved[row])
        self.hits += 1
        self.restored_tokens += cp.position
        self.last_reason = "restored"
        self.plan = None  # The prompt slot retains it; don't keep a third host snapshot alive.

    def capture(self, job):
        import torch
        if job is not self.job:
            return
        self._tp_fence()
        seq = job.sequences[0]
        pos = seq.kv_position
        if not pos or pos % PAGE_SIZE or job.recurrent_state.position != pos:
            return
        key = seq.allocated_pages[pos // PAGE_SIZE - 1].phash
        stash = self._paired_stash(key, pos)
        if stash is None or stash["position"] != pos:
            return
        end_page = pos // PAGE_SIZE
        indices = tuple(p % self.ring_pages
                        for p in range(max(0, end_page - WINDOW_TOKENS // PAGE_SIZE), end_page))
        # Copy raw packed Q8 values AND scales (or raw FP16 K/V), without
        # dequantization or a transient GPU clone. Blocking host copies ensure
        # the snapshot is complete before the ring can wrap/overwrite it.
        saved = []
        for src in self.storage:
            dst = torch.empty((len(indices), *src.shape[1:]), dtype=src.dtype, device="cpu")
            for row, index in enumerate(indices):
                dst[row].copy_(src[index])
            saved.append(dst)
        cp = DraftCheckpoint(pos, key, seq.sequence_ids.torch_slice(0, pos).clone(),
                             indices, tuple(saved))
        prompt_boundary = (len(seq.input_ids) - 1) // PAGE_SIZE * PAGE_SIZE
        if pos <= prompt_boundary:
            self.prompt_checkpoint = cp
        else:
            self.decode_checkpoint = cp

    def finish(self, results):
        for result in results:
            if result.get("job") is not self.job:
                continue
            if result.get("stage") == "error":
                self.invalidate("generation_error")
                return
            if result.get("eos"):
                self.published = tuple(cp for cp in (self.prompt_checkpoint, self.decode_checkpoint)
                                       if cp is not None)
                self.prompt_checkpoint = self.decode_checkpoint = self.plan = None
                self.job = None
                return

    def stats(self):
        snapshots = self.published + tuple(cp for cp in
                                          (self.prompt_checkpoint, self.decode_checkpoint, self.plan)
                                          if cp is not None)
        unique = {id(cp): cp for cp in snapshots}
        return {"mode": "previous_request", "hits": self.hits, "misses": self.misses,
                "restored_tokens": self.restored_tokens, "host_bytes": sum(cp.host_bytes for cp in unique.values()),
                "snapshots": len(unique), "last_reason": self.last_reason}


def enable_prefix_cache(generator, manager=None):
    """Install instance-gated hooks; GPU verification is required before adoption."""
    from exllamav3.generator.generator import Generator
    from exllamav3.generator.job import Job

    if manager is None:
        validate_engine_sources()
        if (generator.num_remaining_jobs() or getattr(generator, "job_serial", 0)
                or getattr(generator, "cpu_page_cache", None) is not None):
            raise ValueError("TP prefix helper requires a fresh idle generator without a generic CPU tier")
        manager = PreviousRequestCache(generator)

    if not getattr(Job.prepare_for_queue, "_glm_dflash_prefix", False):
        prepare, allocate, stash = Job.prepare_for_queue, Job.allocate_pages, Job.maybe_stash_recurrent
        iterate, cancel = Generator.iterate, Generator.cancel

        def prepare_hook(self, g, *args, **kwargs):
            result = prepare(self, g, *args, **kwargs)
            manager = getattr(g, "_glm_dflash_prefix_cache", None)
            if manager:
                manager.restrict(self)
            return result

        def allocate_hook(self):
            result = allocate(self)
            manager = getattr(self.generator, "_glm_dflash_prefix_cache", None)
            if manager:
                manager.restore(self, allocate)
            return result

        def stash_hook(self, cache, *args, **kwargs):
            before = self.last_recurrent_checkpoint_pos
            result = stash(self, cache, *args, **kwargs)
            manager = getattr(self.generator, "_glm_dflash_prefix_cache", None)
            if manager and self.last_recurrent_checkpoint_pos != before:
                manager.capture(self)
            return result

        def iterate_hook(self):
            manager = getattr(self, "_glm_dflash_prefix_cache", None)
            try:
                result = iterate(self)
            except BaseException:
                if manager:
                    manager.invalidate("engine_error")
                raise
            if manager:
                manager.finish(result)
            return result

        def cancel_hook(self, job):
            manager = getattr(self, "_glm_dflash_prefix_cache", None)
            if manager and (job in self.active_jobs or job in self.pending_jobs):
                manager.invalidate("cancelled")
            return cancel(self, job)

        prepare_hook._glm_dflash_prefix = True
        Job.prepare_for_queue, Job.allocate_pages, Job.maybe_stash_recurrent = prepare_hook, allocate_hook, stash_hook
        Generator.iterate, Generator.cancel = iterate_hook, cancel_hook

    generator._glm_dflash_prefix_cache = manager
    return manager
