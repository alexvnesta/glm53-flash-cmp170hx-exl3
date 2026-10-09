# Optional KDA output-projection diagnostic

This standalone diagnostic samples loaded layer-split KDA state and raw output projection calls in ExLlamaV3. It is inert unless its launcher is used. It changes no API, engine source, service startup, or default configuration. This branch starts from the stable prefill-cap change and does not contain the experimental RAM features.

The six hook, projection, launcher, sitecustomize, HTTP runner, and CPU-runner files are byte-identical to the frozen v3 tools. Two tests and the original saved fixtures are also preserved. `source_manifest.json` records those hashes and the external engine source pins. No engine tree, native library, weights, service controls, private trace, or machine receipt is included.

## What it measures

The inherited request gate acts only inside a nonempty `Generator.iterate` call on the same thread. Loading and empty iterations do not consume samples. TP models and exported recurrent state are refused. The hook clones the real convolution/recurrent states, compares Q1 and Q8 calls, checks all 14 static buffer groups, and restores the original forward path.

With output projection explicitly enabled, it takes the actual Q8 clone's first `s_caof` row, clones it, and repeats it into eight identical rows. Four direct `exl3_gemm` calls use the loaded EXL3 weights with newly owned A, C, and Hadamard scratch. Automatic calls record their returned cooperative tags; native shape compatibility is checked for both row counts under the actual layer device guard. A common tag is then requested for both with one SM. The binding records returned shape tags, not actual SM-count telemetry.

Inputs, slots, states, statics, projection weights/markers, and any bias are checked before and after. Original references are retained so storage replacement cannot hide mutation. Errors, nonfinite outputs, wrong/incompatible tags, mutation, and memory-limit violations refuse before normal forward. This is an intrusive diagnostic: extra calls can warm autotuning or capture state and invalidate performance comparisons.

The raw comparison excludes the native bias add. Direct non-graph calls on a repeated row do not reproduce full graph execution, every layer/output row, or autoregressive continuation. [The bounded observed result](OBSERVATIONS.md) does not establish a full-model parity fix or prove that shape selection alone causes the difference.

## External prerequisites

Use an external, complete ExLlamaV3 source tree at `16a49792a3c93d8432d72e6c4bce800841566577` from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3/tree/16a49792a3c93d8432d72e6c4bce800841566577). Every source file pinned in `source_manifest.json` must exist and match. A partial mirror is insufficient. These sources are read as fixtures and are not vendored or imported by the CPU runner.

Actual native execution additionally requires an explicitly supplied compatible compiled extension, an API entry point, and independently verified SHA256 values for both. Matching source hashes and mocked calls do not establish compiled ABI compatibility. Fresh processes are required because native environment flags may be cached. An owner must provide idle/exclusive execution, bounded requests, and restoration; these tools do not manage services or GPUs.

## CPU contracts

A Python environment with CPU-capable PyTorch is required. Run the preserved runner with explicit paths; the tests' legacy sibling fallback is not part of the public invocation.

```sh
/ABS/CPU-PYTHON experimental/kda_diagnostic/run_cpu_tests.py \
  --engine-root /ABS/EXLLAMAV3-16a-SOURCE \
  --output /ABS/CPU-VALIDATION
```

The 68 contracts use CPU tensors, native/HTTP doubles, and pinned source methods. They cover request-phase nesting/reset, history slots and static shapes, mutation/storage reuse, pre-import source/ABI checks, exact request preservation, native argument layout, cooperative tags, common-shape compatibility, owned scratch, failure retention, and layer/memory limits. The runner hides CUDA and requires CUDA to remain uninitialized. CPU contracts do not independently prove real DMA, native numerical behavior, or full-model parity.

## Controlled native launch

Use a fresh process and a fresh trace directory. `sitecustomize.py` is only for the launcher's subprocess import path; do not add this diagnostic directory to unrelated Python environments.

```sh
EXL3_GEMV=0 EXL3_INT8_GEMV=0 GLM53_PORT=8013 \
  /ABS/DEPLOYED-PYTHON experimental/kda_diagnostic/launch_actual_kda.py \
  --engine-root /ABS/EXLLAMAV3-16a-SOURCE \
  --api /ABS/glm_api.py --api-sha256 API_SHA256 \
  --native /ABS/exllamav3_ext.so --native-sha256 NATIVE_SHA256 \
  --trace-directory /ABS/FRESH-PRIVATE-TRACE \
  --max-layers 3 --max-clone-bytes 83886080 \
  --probe-output-projection --max-projection-layers 3 \
  --max-projection-bytes 1048576 -- API_ARGUMENTS
```

Projection defaults off. Its layer limit defaults to one and cannot exceed three. The default 1MiB projection estimate has a 4MiB maximum, separate from clone/output accounting. These estimates exclude normal model allocations, native static slots, global locks/autotuner memory, and CPU snapshots. The launcher checks source/API/native pins before imports and checks loaded paths and hashes again.

For a bounded already-running condition, the preserved HTTP runner accepts the synthetic one-case `projection_fixture.json` or four-case `kda_request_fixture.json`. It sends the saved bodies unchanged, normalizes only the supported `glm` health alias, and refuses a busy or mismatched condition. Use `kda_trial_benchmark.py --help` for its required output/fixture arguments. It performs no service control, cancellation, retry, or request rewriting. The hook samples once per eligible layer, rather than each later HTTP request.

Trace/config files include local paths, process addresses, and tensor provenance. Keep them private. Publish reviewed numerical summaries and synthetic request fixtures only. See [NOTICE.md](NOTICE.md) for licensing and attribution.
