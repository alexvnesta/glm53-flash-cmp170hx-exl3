# Experimental completed-session retention

The API can retain several completed paired prefixes rather than discarding the preceding request on every unrelated miss. This is opt-in, disabled by default and currently restricted to layer split with one active DFlash2 sequence. The existing previous-request helper remains unchanged.

Enable the new mode with `--dflash-session-cache`. When using `scripts/serve.sh`, set `GLM53_PREFIX_CACHE=0` first: its existing previous-request default must not be combined with the new mode. `--target-cpu-cache-gib 0` preserves only GPU-resident target chains; a positive integer, such as 1 or 16, enables inactive target page spill. The independent `--session-cache-gib` default 1 bounds paired draft/token plus native-checkpoint payload. `--recurrent-cache-gib` preserves the native 4 GiB default. `--session-cache-max-checkpoints` defaults to 8 retained paired boundaries, not 8 simultaneously active jobs.

Only complete, unreferenced target pages are transferred to RAM when the native GPU page allocator actually evicts them. The lazy tier has no background pinning thread. Its initial pinned allocation is zero regardless of the configured ceiling; one whole raw page slab is pinned on each new overflow slot. Existing slabs are recycled and remain allocated until tier close. Active attention and all weights remain GPU-resident. RAM extends inactive-session prefix retention, not active token streaming, concurrency or maximum single-request context.

The generic draft CPU tier is excluded. The draft’s physical 32-page ring and the target’s full page IDs do not have the same geometry. Instead, each retained native KDA checkpoint is paired with the raw last 2048 draft tokens and exact prefix IDs. Epoch cookies fence native HostPool recycling; the adapter never keeps raw references to native stash buffers after their eviction. Unrelated completed prefixes survive misses, cancellation and generation errors. Current unpublished snapshots are discarded on cancellation. Native hash-chain liveness is checked across both GPU and CPU tiers, and a missing/recreated checkpoint or allocation mismatch falls back to cold replay.

The combined paired limit counts raw snapshot/token and paired native-checkpoint payload. The target argument bounds raw pinned slabs. Health separately reports `slab_budget_bytes`, actual `pinned_bytes`, `token_snapshot_bytes`, entry counts and spill/restore/fallback counters. The compatibility field `reserved_budget_bytes` means the slab ceiling, not an upfront reservation. Native HostPool retention, unpaired recurrent snapshots and Python metadata add memory outside these payload bounds; there is no total-RSS cap. The tier’s token IDs add 2048 B per 256-token entry.

Optional host pin allocation-capacity failures disable further growth and skip prefix-preservation spill, allowing cold replay when needed. Existing slabs remain usable. CUDA/context/transfer errors propagate instead of being classified as host capacity failures.

The adapter checks exact engine source hashes before installation. Its supported ownership/queue contracts are ExLlamaV3 source 16a49792a3c93d8432d72e6c4bce800841566577: recurrent.py, pagetable.py, job.py, generator.py and cpu_cache.py. Source drift refuses rather than silently adopting an untested contract. TP, MTP, concurrent sequences and requeue use are excluded.

## Qualification

Run the CPU contracts with an explicit pinned ExLlama checkout:

```text
python scripts/run_cpu_tests.py --engine-root PATH_TO_EXLLAMAV3_CHECKOUT
```

The tests read native methods through AST extraction and execute them with CPU tensors. They do not import the engine/native extension, initialize CUDA or load model weights. Model output equivalence, actual GPU/CPU transfer ordering, async disconnect cleanup and global-hook scheduling still require a separate guarded inference campaign.

`scripts/qualify_session_cache.py` is an explicit private-loopback HTTP campaign for an already running isolated 65536-token cache service. It refuses port 8012 and stale output directories. It saves an exact calibrated fixture before inference: two distinct approximately 40000-token sessions, resume A, unrelated cold C, resume B, bounded SSE cancel D and recover B. Initial slabs must be zero, B must cause real spill, and A must perform CPU restores. Restored A/B and B after cancellation must match their cold message, completion count and finish reason. Early stop is valid. Requests, raw HTTP/headers/health and a pass/fail report are written only when the operator explicitly invokes `--run`.

No broad losslessness or speed claim is made. Start with adaptive DFlash disabled. A private pilot does not qualify 384K geometry until a matched full-size campaign passes. Roll back by omitting all new session/CPU options and using the unchanged previous-request mode.

## Controlled 64Ki result and larger campaign

A controlled layer-split Q8/DFlash K7 campaign used a 65536-token target cache and two 40018-token rendered prompts. Each cold reply completed in 5 tokens. Initial pinned target KV slabs were zero; the second session caused 57 GPU-page spills and 179773440 B pinned slabs. Session A restored 57 CPU pages and reused 39936 input tokens. After an unrelated cold request, session B also reused 39936 tokens; cumulative CPU restores reached 131. Entire message/count/finish signatures matched cold/restored A and cold/restored/post-cancel B. A real SSE delta preceded cancellation, and the candidate returned idle. This validates those measured repeat cases, not arbitrary follow-ups, model quality or general losslessness.

The CPU-tier eviction counter stayed zero. Host-capacity eviction, allocation faults and stale-epoch behavior therefore remain CPU contracts rather than that live proof. Slabs remained allocated after use, separate from paired checkpoint payload and loading memory. No total process RSS limit is claimed.

The qualification script now accepts `--cache-tokens` (default 65536), `--session-content-tokens` (default 40000) and `--request-timeout` (default 180 seconds). Sessions must exceed half the usable cache and leave context headroom. Token calibration allows max(2000,target//10) rows with 14 attempts per label. Full 384Ki geometry uses 393216/220000/300 respectively. These larger values are covered by offline doubles; full-size model qualification is still pending. A 1 GiB host ceiling at that geometry is a capacity-stress test, because native fetch retains redundant host copies and protects the incoming chain during restore. A larger ceiling can isolate geometry without making an upfront reservation.

## Optional eviction of redundant host images

`GLM53_CPU_DUPLICATE_RECYCLE=1` enables a separate policy within the lazy target tier. It defaults to 0; other values refuse. It needs the session mode and a positive target CPU budget. No API defaults or native raw copy methods change.

At actual host capacity pressure, the policy first considers host images whose complete GPU page is currently referenced. PageTable/generator/cache ownership, physical page index, hash, parent, token contents and non-revertible completion state must match. Such a GPU page cannot be repurposed by native eviction. Its redundant host image can therefore be recycled even if its hash belongs to the incoming allocation's protected chain. Unreferenced GPU duplicates are excluded. If every host entry is protected and no eligible duplicate exists, the spill is skipped to preserve unrestored entries. Otherwise the native chain/LRU policy runs with a fresh order snapshot.

The optional mode binds the first transfer's thread and current stream identity per device and refuses any later thread/stream, storage, geometry or owner drift before transferring or reusing a slot. Each host byte range belongs to a fixed cache tensor/device. Native H2D reads and subsequent recycled-slot D2H writes stay ordered on that same device's stream; different devices use disjoint ranges. PyTorch 2.14.1 selects the CUDA-side device for CPU/GPU copies and submits nonblocking transfers on its current stream. Current stream selection is thread-local and indexed by device. The ordering argument is an inference from those implementations plus the enforced fixed ownership; CPU tests do not prove CUDA DMA timing. [Copy.cu](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Copy.cu), [CUDAStream.cpp](https://github.com/pytorch/pytorch/blob/v2.14.1/c10/cuda/CUDAStream.cpp).

Stream/owner drift is a fatal refusal requiring a clean process restart. The native page-allocation transaction cannot reliably recover partial claims through a cold retry at that point. Ordinary host capacity skips remain recoverable. Keep one serialized LS worker and defragmentation disabled. `duplicate_evictions` is a subset of tier `evictions`; `protected_spill_skips` also contributes to `skipped_spills`. Neither recycling nor a lower entry count releases already pinned slabs.

The source tests execute native PageTable/CPUPageCache methods with CPU tensors labeled as two fake devices. At 32 GPU pages, two 20-page sessions and 17 host slots, stock eviction resumes A then loses B's checkpoint; the optional policy retains both. Tests cover all raw cache planes, unrelated misses, cancellation, paired recurrent/draft recovery, protected-only host capacity, current streams, foreign threads and owner/layout faults. Real transfers and AsyncGenerator cancellation still require guarded model testing.

A completed 384Ki/1GiB capacity trial of the original policy did not qualify warm B: A reused 219904 tokens, but B fell back to a successful cold HTTP response. Its entire message/count/finish signature matched cold B. Host evictions reached 26 after restoring A and 43 after an unrelated C; cancellation steps were not reached. These counters and the native retention policy support the redundant-copy explanation, but that run did not capture per-page residency. The new optional policy remains GPU-unqualified. It improves the verified referenced-duplicate case, not an optimal exclusive GPU+CPU capacity sum for every request order.
