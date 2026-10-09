# Experimental selected-latent RAM overflow

The separate `experimental/active_host` launcher defaults to `--mode normal`, which installs no host adapter. Active mode must be requested explicitly. Normal mode preserves the existing runtime path. Optional combined mode is described in [combined_memory.md](combined_memory.md). The new adapter and native sources use this repository's MIT license; the original Flun copyright and license are preserved. ExLlamaV3 and PyTorch remain external dependencies. No engine source copies, third-party headers, model weights, binaries, service units or machine receipts are redistributed.

This prototype moves all eleven Q8 MLA latent arenas to pinned host RAM after a sufficiently long GPU prefill. GPU selection gathers only the requested packed rows and tails into fixed staging buffers for decode. The indexer, KDA states, drafter and ordinary prefill remain on GPU. The original GPU arenas are allocated first, and are restored before idle, cancellation, defragmentation or another prefill. It therefore demonstrates release of active latent storage during decode, not a larger maximum context, more concurrent requests or allocation only when global GPU memory is exhausted. RAM transfers happen once the explicit context threshold is reached, rather than only under allocator pressure.

The current host budget counts explicit power-of-two pinned allocation requests; tensor payload is reported separately. Cached free allocator blocks, native checkpoints, driver bookkeeping and process RSS remain separate. Budget decode staging and GPU restoration headroom separately. Earlier standalone measurements used payload accounting; their historical results do not retroactively qualify this new accounting or combined policy.

The current source-bound adapter requires the stock ExLlamaV3 revision `16a49792a3c93d8432d72e6c4bce800841566577`, one serialized layer-split request, eleven Q8 NoPE MLA layers, DFlash K7 and an 8192-token draft ring. Standalone active mode rejects TP, prefix/session retention, generic CPU page tiers, requeue and full-attention captured cache owners. Explicit combined mode permits only its owned fixed-page session/tier coordinator. `EXL3_BC_ATTN=0` must be set before imports. The actual loaded attention flags and cached owners are checked as well as the environment. Source drift refuses rather than silently adopting a different ownership contract.

The native mapped-host alias verifies CPU/pinned/contiguous storage, requested GPU, Host pointer type, registration flags, device mapping and absence of stream capture before constructing a device tensor. It retains the CPU storage owner and uses PyTorch's explicit TensorMaker target device. Device or managed allocations have no accepted relabel path. Cross-device access and slab reuse still require completed lifetime fences. This uses external public APIs, without copying their implementation: [PyTorch from_blob.h](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/ops/from_blob.h), [CUDA memory API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html).

## Build and input attestation

Use the intended CUDA-capable Python environment and a fresh build directory. Compilation requires an explicit Ampere architecture and checks the installed TensorMaker header. It asserts CUDA remains uninitialized. The resulting native build receipt records local source/toolchain/header hashes and must not be committed.

```text
CUDA_VISIBLE_DEVICES='' TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 \
  python experimental/active_host/build_native.py \
  --build-directory /ABS/FRESH_HOST_BUILD
```

There is no machine-specific input template. Generate the probe manifest from explicit installed paths. The generator verifies known stock DSA, MLA and attention source hashes without importing Torch or the engine. The engine extension hash must come from an independent local source/build attestation; a supplied hash alone does not establish its provenance. Output creation is exclusive.

```text
python experimental/active_host/engine_inputs.py \
  --engine-root /ABS/ENGINE_PACKAGE_PARENT \
  --engine-extension /ABS/EXLLAMAV3_EXTENSION.so \
  --engine-extension-sha256 ATTESTED_SHA256 \
  --output /ABS/FRESH_ENGINE_INPUTS.json
```

Only an operator who owns an isolated two-GPU service trial should invoke the explicit native GPU probe. It refuses foreign requested GPU owners, source/binary drift and existing output before imports. This command does not manage service restoration.

```text
python experimental/active_host/validate_gpu.py --run-gpu-probe \
  --build-receipt /ABS/FRESH_HOST_BUILD/build_receipt.json \
  --input-manifest /ABS/FRESH_ENGINE_INPUTS.json \
  --devices 0,1 --output /ABS/FRESH_NATIVE_RESULT.json
```

The build name and source hashes must match this v3 code; older alias builds refuse. Two actual GPUs are required, including cross-registration-device mapping. The probe verifies mapped reads/writes, storage ownership, Q1/Q8 selected union, packed bytes/scales, unchanged DSA, graph replay, cancellation draining, stale epochs and deployed quant-append/rejected-tail overwrite. These small checks do not establish a full-model optimization by themselves.

## Opt-in service trial

Use a fresh isolated API process on a private loopback port. The operator must arrange ownership, shutdown and restoration outside this launcher. Active mode requires explicit absence acknowledgements and rejects forwarded cache options. Omit these experimental arguments to use the normal API path.

```text
EXL3_BC_ATTN=0 python experimental/active_host/run_trial.py \
  --api /ABS/GLM_API/glm_api.py --mode active-host \
  --build-receipt /ABS/FRESH_HOST_BUILD/build_receipt.json \
  --threshold 8192 --host-budget-bytes 1073741824 \
  --no-prefix-cache --no-session-cache -- API_ARGUMENTS
```

The strict HTTP runner uses one saved deterministic 256-output request, then a bounded real SSE disconnect and fresh prefill. It refuses production port 8012, non-loopback endpoints and stale output directories. Supply the same fixture for separately started `normal` and `active-host` arms; the script has no fixture tied to a private machine. Its current live-tested geometry is fixed at cache 65536, public context 65280 and threshold 8192.

```text
python experimental/active_host/run_active_host_http.py run --run \
  --condition normal --fixture /ABS/SAVED_FIXTURE.json \
  --base-url http://127.0.0.1:8013 --out /ABS/FRESH_NORMAL
```

Repeat unchanged for the active arm with a fresh output directory. After both arms, compare entire choices and semantic usage, exact request hashes and three journal migration/restoration epochs:

```text
python experimental/active_host/run_active_host_http.py compare \
  --normal /ABS/FRESH_NORMAL/report.json \
  --active /ABS/FRESH_ACTIVE/report.json \
  --normal-journal /ABS/NORMAL_JOURNAL.txt \
  --active-journal /ABS/ACTIVE_JOURNAL.txt \
  --output /ABS/FRESH_COMPARISON.json
```

## Measured scope and portable checks

A controlled 64Ki initial-cache campaign passed this exact synthetic first-request, cancellation and fresh-prefill recovery lifecycle. Entire first/recovery choices and semantic usage matched the fresh normal arm. Three real migration/restoration epochs each replaced 392167424 B (374 MiB) of latent storage, with 286021120 B (272.77 MiB) reduction in live tensor occupancy reported by `torch.cuda.memory_allocated`. The indexer stayed on GPU. Separate v3 native checks passed on both devices, including actual allocator pointer reuse across allocating devices. This is bounded evidence for those measured requests and lifetimes. It does not prove general losslessness, larger context capacity, simultaneous sessions, a throughput benefit or production readiness.

A separate full 384Ki initial-cache campaign also passed. Its first synthetic ledger rendered 133153 prompt tokens; the fresh-prefill recovery rendered 133162. Both normal and active arms returned 28 completion tokens with finish reason `stop`, and their entire choices and usage matched. A real SSE text delta preceded deliberate disconnect, idle recovery and the fresh request. Three ordered migration/restoration epochs each replaced and restored 2353004544 B (2244 MiB) of latent storage. The reduction in `torch.cuda.memory_allocated` was 2246632960 B (2142.55615234375 MiB) per epoch, derived from both devices' before/after live-tensor counters. `memory_reserved`, driver-free memory and nvidia-smi release were not measured; allocator blocks may stay reserved. This is not a claim that an external process gains that much available VRAM. Hot/staging buffers and headroom needed to restore GPU arenas are separate from the host-arena budget. Capacity stayed 393216 cache tokens and 392960 public context tokens. These are two separate fresh process arms of one saved synthetic case, not repeated performance measurements or proof beyond the original context capacity.

The separate full-pool runner `experimental/active_host/qualify_active_host_full.py` and its pure synthetic `matched_fixture_full.json` preserve that tested request byte for byte (SHA256 `3e0b21ec682a1064419c5f40f2055e0ff7171f986b0d042c67510e2de10bb42a`). It requires that fixture hash, 133153 first-prompt tokens, the exact Q8/chunk/ring health profile, cache 393216 and threshold 131072. Per-request calls are bounded at 300 seconds and the full lifecycle at 600 seconds; nested deadlines cannot extend the enclosing bound. The first/recovery output cap is 128 and early stop is valid. The original 64Ki runner remains separate.

```text
python experimental/active_host/qualify_active_host_full.py run --run \
  --condition normal \
  --fixture experimental/active_host/matched_fixture_full.json \
  --base-url http://127.0.0.1:8013 --out /ABS/FRESH_FULL_NORMAL
```

Repeat for a separately started active arm using threshold 131072 and a host budget that can hold 2244 MiB of latent arenas plus separate staging/metadata. Use the same runner's `compare` subcommand and the two complete service journals. No runtime adapter/native sources changed between the 64Ki and full-pool campaigns. Raw machine receipts, journal files and binaries stay outside Git.

The portable suite uses externally supplied pinned engine source through AST extraction. It does not import the engine, Torch or a native extension, initialize CUDA or access a service. The public tree excludes the private source-input copies; exact stock hashes are checked when deriving the test inputs.

```text
python scripts/run_active_host_cpu_tests.py --engine-root /ABS/EXLLAMAV3_CHECKOUT
```

CPU contracts cover active/default-off ownership gates, migration rollback and release, completion leases, selection/scatter oracle, mapped-alias source safety, source/binary drift and HTTP lifecycle doubles. They do not replace native compilation or real asynchronous DMA and cancellation tests. Rollback consists of launching the unchanged normal API without this wrapper's active mode; no engine files or service configuration are patched by this feature.
