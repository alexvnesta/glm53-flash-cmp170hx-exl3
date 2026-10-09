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
D did not qualify cancellation while latent host aliases were active. The
separate lifecycle supplement below covers active-layout cancellation, a normal
recurrent-checkpoint return and both measured pressure-policy branches. A later
pre-copy fail-stop supplement below qualifies one deliberate return failure.
Mid-transfer faults and broader eviction/allocation-failure paths remain pending.
This bounded pass establishes neither general losslessness nor a
throughput/capacity or boot guarantee.
The controller owner supplied this result and reports its independent offline
audit passed. [qualification_summary.json](qualification_summary.json) records
the sanitized geometry; raw journals, process metadata and binaries stay out
of Git. Runtime files in this clean copy match the tested candidate hashes.

## Separate lifecycle supplement, 2026-10-09

Three fresh diagnostic processes retained the exact source-pinned runtime and
native inputs. A separate offline receipt review passed 76 of 76 checks,
including immutable sources, actual mapped native libraries, process birth and
unit ownership, complete owned shutdown and restoration of the original
service. The reviewer authored the adapter and instrumentation; the controller
owner alone operated the live trial. Instrumentation observed scalar lifecycle
state around the original methods without changing kernels, but its method
wrappers and import order can perturb execution. Raw machine evidence and the
diagnostic service controller are excluded from this repository.

The lifecycle process used a 16528-token synthetic ledger and a context
threshold of 8192. During an unfinished SSE reply, a real content or reasoning
delta arrived while health simultaneously reported busy, `HOST_ACTIVE` and
suspended target transfers. Disconnect cleanup returned to idle GPU layout,
resumed the target tier and invalidated the cancelled paired checkpoint. Fresh
recovery matched the baseline's full choices and semantic usage exactly.

A separate ignore-EOS decode completed 2304 tokens, reporting 18832 total tokens
and finish reason `length`. At actual KV position 18432, the journal observed
`job_maybe_stash_recurrent` restoring host-resident epoch 4 to GPU on the same
persistent generation thread. Decode migrated again into epoch 5 and returned
before idle. There were exactly five complete, ordered migration/return pairs.
This tests the native 2048-token recurrent-checkpoint boundary, not an injected
checkpoint failure or the semantic quality of an ignore-EOS reply.

The other processes used `context_and_pressure`, minimum context 8192, and the
actual minimum driver-free memory across both devices. Threshold 1 byte
refused all six observed admission attempts and produced no host migration.
Threshold 100 GiB admitted once and completed one migration/return. These
thresholds force both real predicate branches; the trial does not establish a
useful production pressure threshold. The ordinary ledger reply's full choices
and semantic usage matched across all three processes.

Each of the six new epochs replaced and restored 2353004544 B of latent tensor
payload, requested 3137339392 B of pinned extents inside the 3221225472 B active
budget, and reduced summed `torch.cuda.memory_allocated` by 2246632960 B. The
indexer stayed on GPU. These are live-tensor allocation measurements, not
reserved-memory or externally available VRAM measurements. Initial full GPU
cache allocation, context capacity and one-request serialization are unchanged.
Owned pinned-request budgets remain separate from process RSS, cached allocator
capacity and the native recurrent/snapshot accounts.

The fresh restoration receipt verified the original enabled service, idle
normal prefix-cache profile, unchanged sources and native input, original argv,
no shadow Python path, exclusive GPU/listener ownership and a visible smoke
reply. This is a timestamped post-trial snapshot, not persistent experimental
activation or reboot evidence. There was one run per condition, with no repeated
performance or general losslessness claim. The pre-copy fail-stop supplement
below separately qualifies one intentional return fault. The earlier lifecycle
processes did not exercise a failure, and broader failure paths remain separate.

## Separate pre-copy fail-stop supplement, 2026-10-09

A fourth fresh diagnostic process deliberately failed its first GPU return
before the original adapter's drain, allocations or copies. Its source-pinned
runtime and native inputs were unchanged. The full offline receipt/custody audit
passed 51 of 51 checks; 16 CPU negative contracts against copied actual receipts
also pass. The reviewer authored the adapter and diagnostic wrapper; the
controller owner alone operated the live test. Raw machine evidence and the
process-only injection are excluded from this repository.

The exact 16528-token ledger migrated once into `HOST_ACTIVE` epoch 1. At its
first return, the one-shot wrapper recorded `original_deactivate_called=false`,
kept the same host epoch/worker and suspended target tier, and published
`FAILED`. A retry on that failed owner was actually refused before delegation.
There was no successful GPU return or additional migration after the fault.

The first generation POST returned HTTP 500. Health then returned HTTP 503,
and the next generation POST returned HTTP 503. The pinned API's failed-worker
gate precedes `AsyncJob` creation; no direct native job-construction counter was
captured. The rejection and source-bound gate are the stated refusal proof.
The failed layout requires process stop and restart, not in-process continuation.

Application shutdown reported the expected failed-layout close error and
`Application shutdown failed. Exiting.` The raw journal preserves that error;
this is not a graceful adapter close. The captured process family nevertheless
exited under the reviewed owned stop, and the pre-registered independent guard
restored the original service. Fresh source/native/argv, process ownership,
exclusive GPU/listener, idle normal prefix-cache and visible `4`/`stop` checks
passed after restoration. No persistent activation or reboot was tested.

The initial health independently proved full cache 393216/context 392960,
chunk/checkpoint 2048, workspace margin 128, Q8, K7/ring 8192, combined LS mode
and request-accounted budgets. Actual environment receipts captured BC0,
fixed-draft/statistics settings, cache mode and request admission settings;
they do not observe every declared environment variable. The lone migration
again replaced 2353004544 B of latent payload, requested 3137339392 B inside a
3221225472 B budget and reduced summed live tensor allocation by 2246632960 B.
The GPU indexer stayed resident. Failed aliases were held until process exit.

This qualifies only one explicit failure before copies. It does not test faults
midway through DMA, return-allocation exhaustion, checkpoint eviction failure,
general losslessness, speedup, increased context capacity or concurrent GPU
decode. Budgets remain separate owned requests/accounts, not a hard process-RAM
or allocator-cache ceiling. Active host migration remains an opt-in LS mode;
TP active-host support is not implemented.
