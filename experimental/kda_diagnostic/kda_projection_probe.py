"""Owned-scratch, bounded output-projection diagnostic. No engine imports."""
from contextlib import nullcontext
import os

from kda_actual_hook import Refusal, cpu_copy, metadata, metrics
import hashlib


REQUIRED_ABI = ("exl3_gemm", "exl3_gemm_shape_compat",
                "exl3_gemm_num_kernel_shapes", "g_get_num_sms", "g_get_cc")


class ProjectionRefusal(Refusal):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def tensor_hash(torch, tensor):
    value = cpu_copy(tensor).reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def snapshot(torch, tensors):
    return {name: {**metadata(value), "data_ptr": value.data_ptr(),
                   "sha256": tensor_hash(torch, value)}
            for name, value in tensors.items() if value is not None}


def validate_abi(native):
    for name in REQUIRED_ABI:
        if not callable(getattr(native, name, None)):
            raise Refusal(f"output projection native ABI missing: {name}")


def projection_fields(layer, first, torch, *, allow_cpu=False):
    wrapper = getattr(layer, "o_proj", None)
    inner = getattr(wrapper, "inner", None)
    if (getattr(wrapper, "quant_type", None) != "exl3" or
            getattr(inner, "quant_type", None) != "exl3"):
        raise Refusal("loaded output projection is not Linear/LinearEXL3")
    if first.device.type != "cuda" and not allow_cpu:
        raise Refusal("output projection requires the actual CUDA layer device")
    if first.ndim != 2 or first.shape[0] != 1 or first.dtype != torch.half:
        raise Refusal("output projection origin must be one FP16 s_caof row")
    if not first.is_contiguous():
        raise Refusal("output projection origin must be contiguous")
    trellis = getattr(inner, "trellis", None)
    if (not isinstance(trellis, torch.Tensor) or trellis.ndim != 3 or
            trellis.dtype != torch.int16 or not trellis.is_contiguous()):
        raise Refusal("output projection trellis geometry/type/contiguity differs")
    k, n, tile = int(trellis.shape[0]) * 16, int(trellis.shape[1]) * 16, int(trellis.shape[2])
    mcg, mul1 = getattr(inner, "mcg", None), getattr(inner, "mul1", None)
    if type(mcg) is not bool or type(mul1) is not bool or (mcg and mul1):
        raise Refusal("output projection codebook flags are invalid")
    bits, half_k = tile // 16, bool(tile % 16)
    if not 1 <= bits <= 8 or (half_k and (tile % 16 != 8 or not mul1 or bits >= 8)):
        raise Refusal("output projection bitrate is unsupported")
    if (first.shape[1] != k or k <= 0 or n <= 0 or k % 128 or n % 128 or
            getattr(inner, "in_features", None) != k or getattr(inner, "out_features", None) != n or
            getattr(layer, "hidden_size", None) != n):
        raise Refusal("output projection widths differ from native GDN geometry")
    dtype = getattr(layer, "out_dtype", None) or torch.half
    if dtype not in (torch.half, torch.float):
        raise Refusal("output projection output dtype is unsupported")
    fields = {name: getattr(inner, name, None) for name in
              ("trellis", "suh", "svh", "bias", "mcg_tensor", "mul1_tensor")}
    for name, width in (("suh", k), ("svh", n), ("bias", n)):
        value = fields[name]
        if value is None:
            if name != "bias":
                raise Refusal(f"loaded LinearEXL3 projection has no {name}")
            continue
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.half or
                value.ndim != 1 or value.numel() != width or not value.is_contiguous()):
            raise Refusal(f"output projection {name} geometry/type/contiguity differs")
    for name, value in fields.items():
        if name in ("mcg_tensor", "mul1_tensor"):
            if (value is not None) != getattr(inner, name.removesuffix("_tensor")):
                raise Refusal("output projection marker/boolean disagrees")
            if value is not None and (not isinstance(value, torch.Tensor) or value.dtype != torch.int32 or value.numel() != 1):
                raise Refusal("output projection codebook marker is not an int32 scalar")
        elif value is not None and (not isinstance(value, torch.Tensor) or value.device != first.device):
            raise Refusal(f"output projection {name} device differs")
    return inner, fields, k, n, bits, half_k, dtype


def compare_projection(torch, native, layer, first, protected_tensors, config,
                       *, allow_cpu=False):
    """CPU allowance is a Python-only test parameter, never a launcher option.

    Only the direct native binding sees borrowed weights. Its A, C and A_had
    arguments are all newly owned storage. Neither BC pointers nor graph slots
    are passed to it. Four direct calls can warm the native autotuner/cache.
    """
    row = {"event": "kda_output_projection_compare", "layer_idx": layer.layer_idx,
           "device": str(first.device), "origin": "actual Q8 cloned-call s_caof first row",
           "process_environment": {name: os.environ.get(name) for name in
               ("EXL3_GEMV", "EXL3_INT8_GEMV", "EXLLAMAV3_TUNE_CACHE")},
           "calls": [], "scope": "Direct non-graph projection of one repeated actual input row; no full-model fix, parity or performance claim."}
    try:
        validate_abi(native)
        if any(row["process_environment"][name] != "0" for name in ("EXL3_GEMV", "EXL3_INT8_GEMV")):
            raise Refusal("direct projection requires actual EXL3_GEMV=0 and EXL3_INT8_GEMV=0")
        inner, fields, k, n, bits, half_k, dtype = projection_fields(layer, first, torch, allow_cpu=allow_cpu)
        # Four inputs/outputs/scratch: reuse the owned 1/8-row allocations for
        # the forced pair; retain only CPU copies of the automatic outputs.
        owned_bytes = 9 * (k * 2 + k * 2 + n * (4 if dtype == torch.float else 2))
        if owned_bytes > config["max_projection_bytes"]:
            raise Refusal("owned projection allocation estimate exceeds its budget")
        if not bool(torch.isfinite(cpu_copy(first).float()).all().item()):
            raise Refusal("actual projection input contains nonfinite values")
        row.update({"input_origin": metadata(first), "input_sha256": tensor_hash(torch, first),
                    "in_features": k, "out_features": n, "bits": bits, "half_integer_bits": half_k,
                    "codebook": {"mcg": inner.mcg, "mul1": inner.mul1},
                    "output_dtype": str(dtype), "bias_present": fields["bias"] is not None,
                    "owned_projection_bytes_estimate": owned_bytes})
        before_inner = inner
        before_codebook = (inner.mcg, inner.mul1)
        def all_protected():
            current = getattr(getattr(layer, "o_proj", None), "inner", None)
            if current is not before_inner or (current.mcg, current.mul1) != before_codebook:
                raise Refusal("output projection object/codebook changed during diagnostic")
            return {**protected_tensors(), **{"weight:" + name: getattr(current, name, None)
                                              for name in fields}}
        # Retain references so an illicit replacement cannot free/recycle an
        # old allocation and accidentally regain its original pointer address.
        before_references = all_protected()
        before = snapshot(torch, before_references)
        row["protected_before"] = before
        a1, a8 = first.clone(), first.repeat(8, 1).contiguous()
        c1 = torch.empty((1, n), dtype=dtype, device=first.device)
        c8 = torch.empty((8, n), dtype=dtype, device=first.device)
        h1, h8 = torch.empty_like(a1), torch.empty_like(a8)
        owned = (a1, a8, c1, c8, h1, h8)
        original_pointers = {value.data_ptr() for value in all_protected().values() if value is not None}
        if any(value.data_ptr() in original_pointers for value in owned):
            raise Refusal("owned projection scratch aliases protected storage")
        a_before = snapshot(torch, {"a1": a1, "a8": a8})
        row["repeated_input_first_row_bitwise_exact"] = all(
            tensor_hash(torch, a8[index:index+1]) == row["input_sha256"] for index in range(8))
        if not row["repeated_input_first_row_bitwise_exact"]:
            raise Refusal("owned repeated rows do not equal the actual origin")
        guard = torch.cuda.device(first.device) if first.device.type == "cuda" else nullcontext()
        with guard:
            device = first.device.index if first.device.type == "cuda" else 0
            sms, cc = native.g_get_num_sms(device), native.g_get_cc(device)
            if type(sms) is not int or sms < 1 or type(cc) is not int:
                raise Refusal("native device metadata is invalid")
            row.update({"native_device_sms": sms, "native_cc": cc})
            def call(label, a, c, scratch, shape, force_sms):
                details = {"label": label, "rows": a.shape[0], "requested_shape": shape,
                           "requested_sms": force_sms}
                try:
                    tag = native.exl3_gemm(a, fields["trellis"], c, fields["suh"], scratch,
                        fields["svh"], shape, inner.mcg, inner.mul1, force_sms)
                except BaseException as error:
                    row["calls"].append({**details, "actual_tag": None,
                        "error_type": type(error).__name__, "error": str(error)})
                    raise
                result = cpu_copy(c)
                finite = bool(torch.isfinite(result.float()).all().item())
                row["calls"].append({**details, "actual_tag": tag,
                    "output_finite": finite, "output_sha256": tensor_hash(torch, result)})
                if type(tag) is not int or not finite:
                    raise Refusal("projection native tag/output is invalid or nonfinite")
                return tag, result
            tag1, auto1 = call("auto_q1", a1, c1, h1, -1, 0)
            tag8, auto8 = call("auto_q8", a8, c8, h8, -1, 0)
            count = native.exl3_gemm_num_kernel_shapes()
            if type(count) is not int or count != 7:
                raise Refusal("native cooperative shape count differs from the pinned ABI")
            if not all(1 <= tag <= count for tag in (tag1, tag8)):
                raise Refusal("automatic projection returned a noncooperative tag")
            common = None
            row["common_shape_candidates"] = []
            for tag in dict.fromkeys((tag1, tag8)):
                if not 1 <= tag <= count:
                    row["common_shape_candidates"].append({"tag": tag, "cooperative": False})
                    continue
                compatible = [bool(native.exl3_gemm_shape_compat(tag, m, k, n, bits, half_k)) for m in (1, 8)]
                row["common_shape_candidates"].append({"tag": tag, "cooperative": True,
                                                       "native_q1_q8_compatible": compatible})
                if all(compatible) and common is None:
                    common = tag
            if common is None:
                raise Refusal("no automatic tag is a common native-validated Q1/Q8 cooperative shape")
            row.update({"common_shape": common, "common_requested_sms": 1,
                        "sms_scope": "Forced request is one SM; pinned dispatch clamps it to one for positive compatible dimensions. Binding returns shape tag only."})
            forced_tag1, force1 = call("common_q1", a1, c1, h1, common, 1)
            forced_tag8, force8 = call("common_q8", a8, c8, h8, common, 1)
            if forced_tag1 != common or forced_tag8 != common:
                raise Refusal("native returned a different forced projection tag")
        after = snapshot(torch, all_protected())
        row["protected_after"] = after
        row["input_sha256_after"] = tensor_hash(torch, first)
        row["protected_storage_and_contents_unchanged"] = before == after
        row["owned_inputs_unchanged"] = a_before == snapshot(torch, {"a1": a1, "a8": a8})
        if not row["protected_storage_and_contents_unchanged"] or not row["owned_inputs_unchanged"]:
            raise Refusal("direct projection changed protected storage/contents or owned inputs")
        row["automatic_first"] = metrics(torch, auto1, auto8[:1])
        row["common_first"] = metrics(torch, force1, force8[:1])
        row["q1_automatic_vs_common"] = metrics(torch, auto1, force1)
        row["q8_automatic_vs_common_first"] = metrics(torch, auto8[:1], force8[:1])
        if fields["bias"] is not None:
            # The actual GDN uses native add_gr, so do not substitute Torch
            # arithmetic and report it as that native post-op's behavior.
            row["bias_scope"] = "Bias excluded; native raw projection only. Full GDN adds it with add_gr."
        row["success"] = True
        return row
    except BaseException as error:
        row.update({"success": False, "error_type": type(error).__name__, "error": str(error)})
        raise ProjectionRefusal(str(error), row) from error
