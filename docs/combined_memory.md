# Experimental combined LS memory mode

The standalone active-host adapter previously excluded retained sessions because
page/checkpoint operations could encounter mapped CPU aliases instead of native
GPU cache tensors. `CombinedMemoryCoordinator` makes the layout transition
explicit. It restores real GPU latent tensors before session/page operations,
rebinds only the original target tier's segments, and fences generator, cache,
page table, worker thread, stream and layout epochs. Unrelated indexer/cache
planes stay on GPU. Failure to complete a layout transaction requires a fresh
process; ownership or source drift refuses further reuse.

`glm_pin_budget.py` requests explicit power-of-two pinned extents and keeps tensor
payload accounting separate. The active host budget now bounds those owned
requests, not process RSS, native checkpoint RAM, cached free allocator blocks
or driver bookkeeping. Session and active budgets remain distinct. Decode
staging and enough VRAM to restore the original arenas must also fit.

The new mode uses stock source contracts at ExLlamaV3
`16a49792a3c93d8432d72e6c4bce800841566577`, Q8 NoPE MLA, eleven target MLA
layers, one serialized LS request, fixed pages, DFlash K7 and the existing draft
ring. It excludes TP, generic/draft CPU tiers, requeue and full-attention BC
cache owners. `EXL3_BC_ATTN=0`, session retention and referenced-duplicate
recycling must be explicit. Ordinary `--mode normal` installs no active hook;
standalone `--mode active-host` retains its session exclusion.

A bounded operator invocation is:

```sh
EXL3_BC_ATTN=0 GLM53_CPU_DUPLICATE_RECYCLE=1 \
  python experimental/active_host/run_trial.py \
  --api glm/glm_api.py --mode combined-memory \
  --build-receipt /PATH/ATTESTED_NATIVE_BUILD.json \
  --threshold 131072 --host-budget-bytes 3221225472 --no-prefix-cache -- \
  --dflash-session-cache --target-cpu-cache-gib 1 OTHER_API_ARGUMENTS
```

Use the native build and input-attestation commands in
[active_host_kv.md](active_host_kv.md). A context threshold is the default
activation policy. `--activation-mode context_and_pressure` additionally checks
bounded free memory before migration using `--pressure-free-bytes`; neither
policy enlarges initial cache capacity or simultaneous jobs. The operator owns
private-port isolation, shutdown and restoration outside this launcher. No
service-unit changes or model files are included in this branch.

## Validation scope

```sh
python scripts/run_cpu_tests.py --engine-root /PATH/STOCK_16A_CHECKOUT
python scripts/run_active_host_cpu_tests.py --engine-root /PATH/STOCK_16A_CHECKOUT
python scripts/run_http_cpu_tests.py
```

The clean publication copy passes 134 LS contracts using CPU tensors, 80 active
stdlib contracts without importing Torch, and the HTTP checks documented
separately. CPU tests use explicit external source through AST extraction and
keep CUDA uninitialized. They cover pinned-request accounting, native
page/checkpoint bodies, layout rollback/rebinding, same-worker ownership and
executor cancellation. Source hashes remain strict; do not broaden them to run
a newer engine. Compilation and real DMA/full-model evidence are distinct from
these CPU results.

Earlier standalone 64Ki/384Ki measurements in `active_host_kv.md` remain
historical, separate-arm results. Combining two individually tested policies
requires its own exact-source live lifecycle qualification: cold A/B, retained
A/B, real unfinished SSE cancellation, idle recovery, and a final fresh request.
Compare full choices and semantic usage, actual host spill/restore and all
layout epochs. A bounded synthetic pass does not establish general
losslessness, speedup, higher context capacity, concurrent sessions or reboot
recovery. Live qualification summaries should identify exact source/native
pins and tested geometry before this optional mode is promoted.

## Bounded live result, 2026-10-09

A full 393216-token initial-cache comparison used the same completed-session
retention and target CPU tier in both arms, with active latent migration enabled
only in the candidate. Six pressure-lifecycle responses and five short/16K
semantic probes passed; their full choices and semantic usage matched between
arms. The large A/B sessions each rendered 219992 prompt tokens, and retained
requests reported 219904 cached tokens. Five ordered migration/return epochs
replaced/restored 2353004544 B of latent payload, requesting 3137339392 B of
pinned extents within a 3221225472 B active budget. Each epoch reduced live
`torch.cuda.memory_allocated` by 2246632960 B (about 2.09 GiB). Reserved memory
and externally available driver-free memory are separate quantities.

The measured activation policy was `context_threshold`, threshold 131072.
The short unfinished cancellation D followed by cached B recovery passed, but
D did not qualify cancellation while latent host aliases were active.
Driver-pressure admission, active-layout cancellation and live checkpoint
failure paths remain separate pending trials. This bounded pass establishes
neither general losslessness nor a throughput/capacity or boot guarantee.
The controller owner supplied this result and reports its independent offline
audit passed. [qualification_summary.json](qualification_summary.json) records
the sanitized geometry; raw journals, process metadata and binaries stay out
of Git. Runtime files in this clean copy match the tested candidate hashes.
