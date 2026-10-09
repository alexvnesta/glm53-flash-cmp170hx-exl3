# Experimental TP DFlash2 and completed-session overflow

Use explicit `-tp --experimental-dflash2-tp`. The typed engine configuration
remains default off. `--dflash-prefix-cache` retains the previous request;
`--dflash-session-cache` enables bounded completed-session retention. Choose one
of those two options. Only one request runs at a time.

The native engine stashes target recurrent checkpoints in every TP rank. The
parent holds the native integer handle and an epoch cookie, paired with the
exact token prefix and raw packed draft-ring window. It does not clone the
worker checkpoint buffers. Recreated native hashes cannot reuse old draft
snapshots. Budget replacement dispatches deletion of the exact handle to every
rank after draining pending replies. Aliased native keys keep their shared
handle until the last key is removed. An unrelated miss preserves completed
sessions. Cancellation/errors discard unpublished work. Native state or target
page eviction can cause a cold retry.

`--target-cpu-cache-gib N` requires session retention. The target-only tier is
attached after a fresh native generator is validated and defragmentation is
disabled in an owned installation transaction. Generic `--cpu_cache_size` is
refused for TP DFlash2: it would also attach the small modulo draft ring using
target page identities. No draft pages enter this CPU tier.

No host mapping is allocated at installation. Native eviction of a complete,
unreferenced GPU page prepares a slot on every rank. Each rank uses
`PinnedArena(required=True)`, an anonymous mapping registered with CUDA and
rounded to `mmap.PAGESIZE`. This avoids Torch's pinned-allocator power-of-two
slab charges. Registration failure closes partly prepared peer slabs, disables
further growth, and lets missing prefixes run cold. Existing stored pages
remain usable. Budget counts the sum of rank slab sizes, including replicated
planes on every rank. Health reports rank sizes, logical slots and committed
mapping bytes. Compare allocation counters at an idle boundary. Token copies,
Python metadata, driver bookkeeping, draft snapshots and native recurrent
HostPool RAM are separate costs; this is not a process-RAM cap. Paired native
checkpoint accounting overlaps native cache accounting and is not additive.

Every slot byte range belongs to a fixed rank/device/current CUDA stream. A
previous H2D read completes before its next D2H overwrite on that same stream;
different devices operate on disjoint mappings. Transfers refuse tensor
identity/pointer/layout or thread/stream drift. The parent pins the generator,
cache-owner serial, page-table geometry and physical page objects. A duplicate
host image is recycled only after exact token, parent, native page ownership,
complete referenced GPU residency and non-revertible state checks. Sole
protected host images remain intact; an incoming spill may be skipped when all
host entries are protected. Teardown synchronizes recorded rank streams before
unregistering mappings.

Shutdown shields the whole responsive async close until its native step and
executor settle. It releases session state, then separately attempts ack drain
and model unload. A failed ack still permits native worker destruction, and
cleanup never hides an original lifespan exception. Use the existing attested
ExecStop/ExecStopPost guard with KillMode=mixed, bounded stop time and before/
after IPC capture. Do not signal TP children concurrently with the parent.

Base is official dev `6cd89a908d957afba1cd1659fcf7d99a9e56d776`. It includes
published default-off TP, typed loader plumbing, reviewed graph/BLAS patches,
and sparse NoPE query-lifetime cleanup. The new slot transport is Python-only.
Use an extension built from the matching C++ tree, rather than historical
stock1.5.4 binaries. The source manifest identifies exact inputs, and this TP
helper gates nine relevant installed engine files before attaching.

The existing LS experimental tier keeps its separate stock1.5.4 source fences.
They have not been broadened to current dev. LS tests use the original external
fixture; TP tests use this new engine tree. Ordinary default-off LS serving is
unchanged.

CPU contracts use real native page/checkpoint/queue bodies and two rank doubles
with CPU tensors. They establish ownership, hook order and raw-copy contracts,
not asynchronous DMA, graphs, full-model parity or performance. The TP HTTP
pilot saves exact requests and raw responses, then performs cold A/B, resume A,
unrelated cold C, resume B, real SSE cancellation D and recovery B. It requires
actual host spill/restoration and matching full choices/semantic token counts.
Start with65,536 target-cache tokens and two approximately40,000-token sessions.
For393,216/220,000 use a2GiB target budget initially and confirm that reported
slot capacity exceeds the required overflow pages. No live qualification of
this combined current-dev candidate has been performed by this agent.

Active sparse-latent host migration remains unsupported in TP. The LS adapter
binds actual layer tensors/native attention objects in one process; TP places
those objects in ranks and exposes only exports in the parent. It needs a
per-rank adapter/coordinator, rank-local registered latent arenas and pointer
leases, return/rebind before checkpoint/page lifecycle operations,
collective-safe failures and separate live qualification. This inactive tier
preserves completed sessions while target working tensors remain on GPU. It
does not increase simultaneous active requests or the configured context limit.
