"""Private LS-only, request-scoped Q1/Q8 layer probe; inert during loading."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from functools import wraps

SOURCE_PINS = {
    "exllamav3/modules/gated_delta_net.py": "8017a2e75eed98fb46945543ea2f42637d3a390bf42c12a724f5e8ad04375374",
    "exllamav3/util/tensor.py": "d49d599e6a64a2dbc235c6ecf153c78dc635e9111ab7302f2317d94b9365c931",
    "exllamav3/modules/attention_fn/bc_attn.py": "561ed1de5d76cff176c84afa7a26fa35697a174ae0d0d1b30ae2c6863cc7ee13",
    "exllamav3/generator/generator.py": "e49340011fbc19447bfdc592b6c739638cfd59220074fa4f97863c7c07f0cb86",
    "exllamav3/modules/linear.py": "249616b34108d1ad40bc995a64f6999bc65a3c8842e38c52a95e024188fa3a94",
    "exllamav3/modules/quant/exl3.py": "3f285c0a34d4bee9bd1ccbbc476036254f8427f1c7279583efad9b1ca4b1eddf",
    "exllamav3/exllamav3_ext/quant/exl3_gemm.cu": "1754f22a4dcbd9a732121e5bb6e079bff4d7ec54005a8d8e19b610fd5ff30598",
    "exllamav3/exllamav3_ext/quant/exl3_gemm.cuh": "606ec462784d2550707df83e052447d111ad9d59ae64806dd272ef2f898a0b9f",
    "exllamav3/exllamav3_ext/quant/exl3_kernel_map.cu": "8a026efbede0d439019c5db95a744fdb0e49ef1cfea847aa30e3595f6e87f6ea",
    "exllamav3/exllamav3_ext/quant/exl3_kernel_map.cuh": "68afe01ded6198adf74dd124f0cd151fe536f918c0067ed77844e34e7688b20e",
    "exllamav3/exllamav3_ext/bindings.cpp": "7b339aafb233ad39dd86d0ef139c7d0eaeed67ffb078761a959b64fa4fb230f1",
}
BUFFER_ORDER = ("s_qkv", "s_kb", "s_kfa", "s_kfb", "s_kga", "s_z",
                "s_beta", "s_kg4", "s_mqkv", "s_conv", "s_cao", "s_caof",
                "s_qkv_xh", "s_o_xh")
_installed = None


class Refusal(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_config(config):
    """All pin checks precede imports of Torch, the engine and the extension."""
    root = Path(config["engine_root"]).resolve()
    for relative, expected in SOURCE_PINS.items():
        if sha(root / relative) != expected:
            raise Refusal(f"source pin differs: {relative}")
    for label in ("api", "native"):
        if sha(config[label + "_path"]) != config[label + "_sha256"]:
            raise Refusal(f"{label} pin differs")
    if not 1 <= config["max_layers"] <= 34:
        raise Refusal("max_layers must be in 1..34")
    if not 1 <= config["max_clone_bytes"] <= 96 * 1024**2:
        raise Refusal("clone budget must be at most 96 MiB")
    if type(config.get("probe_output_projection", False)) is not bool:
        raise Refusal("output projection opt-in must be a boolean")
    if config.get("probe_output_projection", False):
        if type(config.get("max_projection_layers")) is not int or not 1 <= config["max_projection_layers"] <= 3:
            raise Refusal("projection layer limit must be in 1..3")
        if type(config.get("max_projection_bytes")) is not int or not 1 <= config["max_projection_bytes"] <= 4 * 1024**2:
            raise Refusal("owned projection budget must be at most 4 MiB")
        if any(os.environ.get(name) != "0" for name in ("EXL3_GEMV", "EXL3_INT8_GEMV")):
            raise Refusal("projection opt-in requires both GEMV environment flags 0 before native import")
    trace = Path(config["trace_directory"])
    if not trace.is_absolute() or not trace.is_dir():
        raise Refusal("trace directory must be an existing absolute directory")


def buffer_specs(layer, torch, seqlen):
    b, f = 1, layer.fdim_qkv
    nv, hk, hv = layer.num_v_heads, layer.k_head_dim, layer.v_head_dim
    return {
        "s_qkv": ((b, seqlen, f), torch.float, 1),
        "s_z": ((b, seqlen, nv, hv), torch.float, 1),
        "s_kb": ((b, seqlen, nv), torch.float, 1),
        "s_kfa": ((b, seqlen, hk), torch.float, 1),
        "s_kfb": ((b, seqlen, nv * hk), torch.float, 1),
        "s_kga": ((b, seqlen, hv), torch.float, 1),
        "s_beta": ((b, seqlen, nv), torch.bfloat16, 1),
        "s_kg4": ((b, seqlen, nv, hk), torch.float, 1),
        "s_mqkv": ((b, f, seqlen), torch.bfloat16, 2),
        "s_conv": ((b, seqlen, f), torch.bfloat16, 1),
        "s_cao": ((b, seqlen, nv, hv), torch.bfloat16, 1),
        "s_caof": ((b, seqlen, nv * hv), torch.half, 1),
        "s_qkv_xh": ((b, seqlen, layer.hidden_size), torch.half, 1),
        "s_o_xh": ((b, seqlen, nv * hv), torch.half, 1),
    }


def cpu_copy(tensor):
    # clone ensures the CPU test double cannot alias the original; on CUDA the
    # transfer also orders all prior work on the producing stream.
    return tensor.detach().to("cpu").contiguous().clone()


def byte_equal(torch, left, right):
    return left.shape == right.shape and left.dtype == right.dtype and bool(
        torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))


def metadata(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device)}


def metrics(torch, left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise Refusal("comparison geometry or dtype differs")
    lf, rf = left.float(), right.float()
    finite = torch.isfinite(lf) & torch.isfinite(rf)
    delta = (lf[finite] - rf[finite]).abs()
    return {"exact": bool(torch.equal(left, right)), "bitwise_exact": byte_equal(torch, left, right),
            "max_abs_finite": float(delta.max().item()) if delta.numel() else None,
            "different_elements": int((left != right).sum().item()),
            "nonfinite_pairs": int((~finite).sum().item()), "numel": left.numel()}


class ActualKDAProbe:
    def __init__(self, config, torch, tensor_cache, get_for_device, writer, native=None):
        self.config, self.torch = config, torch
        self.cache, self.get_for_device, self.writer = tensor_cache, get_for_device, writer
        self.lock = threading.RLock()
        self.seen = set()
        self.request_phase = threading.local()
        self.iteration_number = 0
        self.native = native
        self.projection_seen = set()

    def request_context(self):
        return getattr(self.request_phase, "context", None)

    def wrap_iterate(self, original):
        """Only enqueued LS work may enter the probe's thread-local phase.

        The pinned Generator constructor and model loader do not call iterate.
        Holding the same reentrant lock used by forward serializes owned steps;
        unrelated threads cannot inherit this request scope. TP worker IPC is
        deliberately unsupported rather than falsely labelled request-only.
        """
        probe = self
        @wraps(original)
        def iterate(generator, *args, **kwargs):
            with probe.lock:
                if (getattr(generator.model, "loaded_tp", False) or
                        getattr(getattr(generator, "draft_model", None), "loaded_tp", False)):
                    raise Refusal("actual KDA v2 diagnostic is layer-split-only; TP request phase is unsupported")
                jobs = generator.num_remaining_jobs()
                if type(jobs) is not int or jobs < 0:
                    raise Refusal("generator job count is invalid")
                if probe.request_context() is not None:
                    raise Refusal("nested generator iteration is unsupported")
                if not jobs:
                    return original(generator, *args, **kwargs)
                probe.iteration_number += 1
                context = {"iteration": probe.iteration_number, "jobs_at_entry": jobs,
                           "thread_name": threading.current_thread().name,
                           "thread_native_id": threading.get_native_id(),
                           "phase": "nonempty_generator_iterate", "target_tp": False}
                probe.request_phase.context = context
                try:
                    if probe.iteration_number == 1:
                        probe.writer({"event": "kda_request_phase_started", **context})
                    return original(generator, *args, **kwargs)
                finally:
                    probe.request_phase.context = None
        return iterate

    def eligible(self, layer, x, params):
        if self.request_context() is None:
            return False
        if params.get("prefill", False) or params.get("is_prefill", False):
            return False
        return (bool(params.get("recurrent_history", False)) and
                tuple(x.shape[:2]) == (1, 8) and x.ndim == 3 and
                bool(getattr(layer, "kda", False)) and
                bool(getattr(layer, "bc_split", False)) and
                bool(getattr(layer, "ba_weight_filled", False)) and
                getattr(layer, "bc", None) is not None and
                getattr(layer, "num_k_heads", 0) > 0 and
                bool(params.get("recurrent_states")))

    def state(self, layer, params):
        rsg = params["recurrent_states"]
        if len(rsg) != 1:
            raise Refusal("diagnostic requires one serialized recurrent job")
        if rsg[0].exported:
            raise Refusal("exported TP recurrent state is unsupported by the LS-only diagnostic")
        slots = self.get_for_device(params, "recurrent_slots", layer.device)
        if slots is None or tuple(slots.shape) != (1,) or slots.dtype != self.torch.int:
            raise Refusal("recurrent slots must be one int32 device index")
        instance = (layer.layer_idx, params.get("layer_instance", 0))
        if rsg[0].exported:
            rsl = layer.tp_recurrent_lookup[rsg[0].cache]
        else:
            rsl = rsg[0].cache.get_recurrent_layer(instance)
        conv, recurrent = rsl.get_state_tensors()
        index = int(slots.detach().cpu().item())
        if (conv.ndim != 3 or recurrent.ndim != 5 or conv.shape[0] != recurrent.shape[0] or
                not 0 <= index < conv.shape[0] or recurrent.shape[1] < 8):
            raise Refusal("recurrent state geometry/slot cannot support a Q8 history window")
        if conv.device != recurrent.device or slots.device != conv.device:
            raise Refusal("recurrent state and slots must share a device")
        return rsl, conv, recurrent, slots, index, instance

    def snapshot_buffers(self, layer, seqlen):
        snapshots = {}
        for tag, (shape, dtype, axis) in buffer_specs(layer, self.torch, seqlen).items():
            key = self.cache.make_key(layer.device, shape, dtype, tag)
            if key not in self.cache.cache:
                raise Refusal(f"configured native static is missing: {tag}")
            tensor = self.cache.cache[key][1]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise Refusal(f"configured native static geometry differs: {tag}")
            first = tensor.narrow(axis, 0, 1)
            snapshots[tag] = (cpu_copy(first), {**metadata(tensor), "sequence_axis": axis,
                                              "first_row_shape": list(first.shape)})
        return snapshots

    def compare(self, layer, x, params):
        torch = self.torch
        rsl, conv, recurrent, slots, index, instance = self.state(layer, params)
        if not x.is_contiguous():
            raise Refusal("actual Q8 input must be contiguous")
        clone_bytes = 2 * (conv.numel() * conv.element_size() +
                           recurrent.numel() * recurrent.element_size())
        clone_bytes += (x[:, :1].numel() + x.numel()) * x.element_size()
        if clone_bytes > self.config["max_clone_bytes"]:
            raise Refusal("cloned-state/output estimate exceeds the diagnostic budget")
        original_conv, original_recurrent = cpu_copy(conv), cpu_copy(recurrent)
        original_x, original_slots = cpu_copy(x), cpu_copy(slots)
        original_pointers = (conv.data_ptr(), recurrent.data_ptr())
        c1, r1 = conv.clone(), recurrent.clone()
        c8, r8 = conv.clone(), recurrent.clone()
        x1 = x[:, :1].contiguous()
        y1 = torch.empty_like(x1, dtype=layer.out_dtype or torch.half)
        y8 = torch.empty_like(x, dtype=layer.out_dtype or torch.half)
        if layer.bc.needs_configure(1, 1, False):
            layer._bc_configure_slot_kda(1, 1, False)
        if layer.bc.needs_configure(1, 8, True):
            layer._bc_configure_slot_kda(1, 8, True)
        layer.bc.run_bszN(x1, y1, c1, r1, slots, False)
        first = self.snapshot_buffers(layer, 1)
        output1, state1 = cpu_copy(y1), cpu_copy(r1[index, 0])
        layer.bc.run_bszN(x, y8, c8, r8, slots, True)
        eighth = self.snapshot_buffers(layer, 8)
        output8, state8 = cpu_copy(y8[:, :1]), cpu_copy(r8[index, 1])
        now_conv, now_recurrent = rsl.get_state_tensors()
        untouched = (original_pointers == (now_conv.data_ptr(), now_recurrent.data_ptr()) and
            byte_equal(torch, original_conv, cpu_copy(now_conv)) and
            byte_equal(torch, original_recurrent, cpu_copy(now_recurrent)) and
            byte_equal(torch, original_x, cpu_copy(x)) and
            byte_equal(torch, original_slots, cpu_copy(slots)))
        if not untouched:
            raise Refusal("diagnostic changed original state/input/slot storage or contents")
        projection = None
        projection_key = (id(layer), instance[1])
        if (self.config.get("probe_output_projection", False) and
                projection_key not in self.projection_seen and
                len(self.projection_seen) < self.config["max_projection_layers"]):
            from kda_projection_probe import compare_projection
            shape, dtype, axis = buffer_specs(layer, torch, 8)["s_caof"]
            origin = self.cache.cache[self.cache.make_key(layer.device, shape, dtype, "s_caof")][1]
            first_row = origin[:, :1].reshape(1, shape[-1])
            def protected():
                current_conv, current_recurrent = rsl.get_state_tensors()
                values = {"original_conv": current_conv, "original_recurrent": current_recurrent,
                          "original_x": x, "original_slots": slots}
                for length in (1, 8):
                    for tag, (buffer_shape, buffer_dtype, _) in buffer_specs(layer, torch, length).items():
                        key = self.cache.make_key(layer.device, buffer_shape, buffer_dtype, tag)
                        if key not in self.cache.cache:
                            raise Refusal("protected native static disappeared during projection diagnostic")
                        values[f"static:q{length}:{tag}"] = self.cache.cache[key][1]
                return values
            projection = compare_projection(torch, self.native, layer, first_row, protected, self.config)
            projection["request_context"] = dict(self.request_context())
            self.projection_seen.add(projection_key)
        comparisons = {tag: {**metrics(torch, first[tag][0], eighth[tag][0]),
                            "q1_metadata": first[tag][1], "q8_metadata": eighth[tag][1],
                            "scratch_activity_unproven": tag in ("s_qkv_xh", "s_o_xh")}
                       for tag in BUFFER_ORDER}
        meaningful = [tag for tag in BUFFER_ORDER if tag not in ("s_qkv_xh", "s_o_xh")]
        record = {"event": "kda_layer_compare", "layer_idx": layer.layer_idx,
                  "request_context": dict(self.request_context()),
                  "layer_instance": instance[1], "device": str(x.device), "slot": index,
                  "exported_state": bool(params["recurrent_states"][0].exported),
                  "query_shapes": [list(x1.shape), list(x.shape)],
                  "history_flags": [False, True], "clone_output_bytes_estimate": clone_bytes,
                  "original_storage_and_contents_unchanged": untouched,
                  "output_first": metrics(torch, output1, output8),
                  "q1_canonical_vs_q8_first_intermediate": metrics(torch, state1, state8),
                  "buffers": comparisons,
                  "first_non_scratch_difference": next((tag for tag in meaningful if not comparisons[tag]["bitwise_exact"]), None),
                  "scope": "Extra actual native layer invocations on equal cloned state; graph warming/counts changed. No speed or full-model equivalence claim."}
        if projection is not None:
            record["output_projection"] = projection
        # Keep all cloned GPU objects alive until the original native call has
        # repatched its Q8 pointers; the wrapper releases them afterward.
        return record, (c1, r1, c8, r8, x1, y1, y8)

    def wrap(self, original):
        probe = self
        def forward(layer, x, params, out_dtype=None):
            with probe.lock:
                key = (id(layer), params.get("layer_instance", 0))
                if key in probe.seen or len(probe.seen) >= probe.config["max_layers"] or not probe.eligible(layer, x, params):
                    return original(layer, x, params, out_dtype)
                with probe.torch.inference_mode():
                    record = None
                    try:
                        record, clones = probe.compare(layer, x, params)
                        probe.seen.add(key)
                        out = original(layer, x, params, out_dtype)
                        record["original_forward_completed"] = True
                        probe.writer(record)
                        del clones
                        return out
                    except BaseException as error:
                        probe.writer({"event": "kda_probe_error", "layer_idx": layer.layer_idx,
                                      "error_type": type(error).__name__, "error": str(error),
                                      "projection_diagnostic": getattr(error, "diagnostic", None) or
                                          (record.get("output_projection") if record else None)})
                        raise
        return forward


def install_config(config, loader=None):
    global _installed
    check_config(config)
    if _installed is not None:
        return _installed
    if loader is None:
        import torch
        import sys
        import exllamav3_ext as native
        from exllamav3.modules.gated_delta_net import GatedDeltaNet
        from exllamav3.generator.generator import Generator
        from exllamav3.util.tensor import g_tensor_cache, get_for_device
        for relative, expected in SOURCE_PINS.items():
            if not relative.endswith(".py"):
                continue
            module_name = relative[:-3].replace("/", ".")
            module = sys.modules.get(module_name)
            desired = Path(config["engine_root"]).resolve() / relative
            if module is None or Path(module.__file__).resolve() != desired or sha(module.__file__) != expected:
                raise Refusal(f"loaded source differs: {relative}")
        if Path(native.__file__).resolve() != Path(config["native_path"]).resolve():
            raise Refusal("loaded native extension path differs")
        if sha(native.__file__) != config["native_sha256"]:
            raise Refusal("loaded native extension hash differs")
        if hasattr(GatedDeltaNet, "_glm_actual_kda_probe"):
            raise Refusal("KDA class is already wrapped")
        if hasattr(Generator, "_glm_actual_kda_probe"):
            raise Refusal("Generator class is already wrapped")
    else:
        torch, native, GatedDeltaNet, g_tensor_cache, get_for_device, Generator = loader()
    if config.get("probe_output_projection", False):
        from kda_projection_probe import validate_abi
        validate_abi(native)
    output = Path(config["trace_directory"]) / f"kda-{os.getpid()}.jsonl"
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stream = os.fdopen(fd, "w", buffering=1)
    def writer(row):
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
    writer({"event": "kda_probe_header", "schema": 3, "pid": os.getpid(),
            "source_base": "16a49792a3c93d8432d72e6c4bce800841566577",
            "source_pins": SOURCE_PINS, "api_sha256": config["api_sha256"],
            "native_sha256": config["native_sha256"], "max_layers": config["max_layers"],
            "max_clone_bytes": config["max_clone_bytes"], "request_gate": "nonempty Generator.iterate in the same thread",
            "probe_output_projection": config.get("probe_output_projection", False),
            "max_projection_layers": config.get("max_projection_layers"),
            "max_projection_bytes": config.get("max_projection_bytes"),
            "target_mode": "layer_split_only", "scope": "Private intrusive request-scoped correctness diagnostic; serialized actual-layer cloned-state calls, no performance claim."})
    probe = ActualKDAProbe(config, torch, g_tensor_cache, get_for_device, writer, native)
    GatedDeltaNet.forward = probe.wrap(GatedDeltaNet.forward)
    GatedDeltaNet._glm_actual_kda_probe = probe
    Generator.iterate = probe.wrap_iterate(Generator.iterate)
    Generator._glm_actual_kda_probe = probe
    _installed = probe
    return probe
