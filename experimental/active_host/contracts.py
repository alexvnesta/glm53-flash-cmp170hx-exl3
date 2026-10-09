"""CPU-only ownership contracts for a future GLM latent host tier.

This module does not import Torch or an inference engine. The event receipts are
an executable contract for an adapter, not evidence of CUDA event completion.
"""
from __future__ import annotations

from dataclasses import dataclass, field


PAGE_SIZE = 256
PACKED_ROW_BYTES = 512
SCALE_ROW_BYTES = 32
LATENT_ROW_BYTES = PACKED_ROW_BYTES + SCALE_ROW_BYTES
MLA_LAYERS = 11


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Identity:
    namespace: str
    page_hash: bytes
    logical_page: int
    generation: int
    layout_epoch: int


@dataclass
class Spill:
    identity: Identity
    slot: int
    nonce: int
    queued: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)
    cancelled: bool = False


@dataclass(frozen=True)
class StepLease:
    nonce: int
    layout_epoch: int
    identities: tuple[Identity, ...]
    start_position: int
    query_rows: int


class HostPageOwner:
    """Bounded target-latent pages, with enqueue/publication and read leases.

    Page bytes charge all eleven layers. Indexer and KDA/draft storage are not
    included: their separate owners must reserve and pair those budgets. Host
    slot allocation occurs only when begin_spill is explicitly needed.
    """

    def __init__(self, *, namespace: str, layout_epoch: int, budget_bytes: int,
                 indexer_capacity_tokens: int, layers: int = MLA_LAYERS):
        if not namespace or layout_epoch <= 0 or budget_bytes < 0:
            raise ValueError("namespace, positive epoch and nonnegative budget required")
        if layers != MLA_LAYERS or indexer_capacity_tokens <= 0:
            raise ValueError("prototype supports the eleven-layer GLM indexer layout")
        self.namespace, self.layout_epoch = namespace, layout_epoch
        self.capacity = indexer_capacity_tokens
        self.page_bytes = PAGE_SIZE * LATENT_ROW_BYTES * layers
        self.max_pages = budget_bytes // self.page_bytes
        self.free_slots = list(reversed(range(self.max_pages)))
        self.pending: dict[int, Spill] = {}
        self.valid: dict[Identity, int] = {}
        self.retired: set[Identity] = set()
        self.step: StepLease | None = None
        self.step_cancelled = False
        self.visible_position = 0
        self._nonce = 0

    @property
    def reserved_bytes(self):
        return (len(self.pending) + len(self.valid)) * self.page_bytes

    def admission(self, *, GPU_prefix_fits: bool, required_host_pages: int,
                  requested_tokens: int):
        if requested_tokens < 0 or requested_tokens > self.capacity:
            raise ContractError("GPU indexer capacity remains a hard context bound")
        if required_host_pages < 0:
            raise ValueError("negative page count")
        if GPU_prefix_fits:
            return "resident"  # no spill, arena allocation or host-copy receipt
        if required_host_pages > len(self.free_slots):
            raise ContractError("insufficient explicit host budget")
        return "host_decode_trial"

    def _check_identity(self, identity):
        if (identity.namespace != self.namespace or
                identity.layout_epoch != self.layout_epoch or
                identity.generation <= 0 or not identity.page_hash or
                identity.logical_page < 0 or
                identity.logical_page * PAGE_SIZE >= self.capacity):
            raise ContractError("stale or incompatible page identity")

    def begin_spill(self, identity: Identity):
        self._check_identity(identity)
        if self.step is not None:
            raise ContractError("cannot migrate or replace pages during a read step")
        if identity in self.valid or any(p.identity == identity for p in self.pending.values()):
            raise ContractError("duplicate spill transaction")
        if any(i.logical_page == identity.logical_page for i in self.valid):
            raise ContractError("retire the previous logical-page generation first")
        if any(p.identity.logical_page == identity.logical_page for p in self.pending.values()):
            raise ContractError("logical-page migration already pending")
        if not self.free_slots:
            raise ContractError("host budget exhausted before releasing GPU source")
        self._nonce += 1
        transaction = Spill(identity, self.free_slots.pop(), self._nonce)
        self.pending[transaction.nonce] = transaction
        return transaction.nonce

    def queued(self, nonce: int, layer: int):
        transaction = self.pending[nonce]
        if transaction.cancelled or not 0 <= layer < MLA_LAYERS or layer in transaction.queued:
            raise ContractError("invalid or repeated layer enqueue receipt")
        transaction.queued.add(layer)

    def source_releasable_in_stream_order(self, nonce: int):
        """Source reuse MUST follow every layer's actual copy on its own stream."""
        return len(self.pending[nonce].queued) == MLA_LAYERS

    def completed(self, nonce: int, layer: int):
        transaction = self.pending[nonce]
        if layer not in transaction.queued or layer in transaction.completed:
            raise ContractError("completion requires a unique matching enqueue receipt")
        transaction.completed.add(layer)
        if (len(transaction.completed) == MLA_LAYERS or
                (transaction.cancelled and transaction.completed == transaction.queued)):
            del self.pending[nonce]
            if transaction.cancelled:
                self.free_slots.append(transaction.slot)
                return "discarded_after_completion"
            self.valid[transaction.identity] = transaction.slot
            return "published"
        return "pending"

    def cancel_spill(self, nonce: int):
        transaction = self.pending[nonce]
        transaction.cancelled = True
        # Once any copy is queued its destination remains leased until completion.
        if not transaction.queued:
            del self.pending[nonce]
            self.free_slots.append(transaction.slot)
            return "discarded_before_enqueue"
        # Drain every actually submitted copy; do not submit remaining layers.
        # Cancellation never assumes CUDA work which was already queued stopped.
        return "drain_required"

    def retire(self, identity: Identity):
        if identity not in self.valid:
            raise ContractError("unknown published identity")
        if self.step is not None and identity in self.step.identities:
            self.retired.add(identity)
            return "deferred_until_step_completion"
        self.free_slots.append(self.valid.pop(identity))
        return "retired"

    def begin_step(self, *, identities: tuple[Identity, ...], start_position: int,
                   query_rows: int):
        if self.step is not None or self.pending:
            raise ContractError("step requires a stable published source layout")
        if not 1 <= query_rows <= 8 or not 0 <= start_position < self.capacity:
            raise ContractError("decode prototype supports one to eight query rows")
        if start_position + query_rows > self.capacity:
            raise ContractError("verification exceeds GPU indexer capacity")
        if len(set(identities)) != len(identities):
            raise ContractError("duplicate read lease")
        for identity in identities:
            self._check_identity(identity)
            if identity not in self.valid:
                raise ContractError("unpublished or evicted source page")
        # The engine adapter must lease ALL host pages of the active prefix.
        # No selected-index CPU readback is required. Pages outside the lease
        # may not appear in the corresponding GPU block table.
        self._nonce += 1
        self.step = StepLease(self._nonce, self.layout_epoch, identities,
                              start_position, query_rows)
        self.step_cancelled = False
        return self.step

    def cancel_step(self, nonce):
        self._check_step(nonce)
        self.step_cancelled = True

    def _check_step(self, nonce):
        if self.step is None or self.step.nonce != nonce:
            raise ContractError("stale step completion")

    def finish_step(self, nonce, *, actual_event_complete: bool,
                    checked_kernel_error_mask: int, accepted_tokens: int,
                    target_checkpoint_epoch: int = 0,
                    paired_draft_checkpoint_epoch: int = 0):
        self._check_step(nonce)
        if not actual_event_complete:
            raise ContractError("no publication or hot-buffer reuse before GPU completion")
        step = self.step
        if not 0 <= accepted_tokens <= step.query_rows:
            raise ContractError("invalid speculative acceptance")
        # Most decode steps do not create a resumable checkpoint. If a caller
        # publishes one here, the native target epoch must match its draft pair.
        paired = ((target_checkpoint_epoch == paired_draft_checkpoint_epoch == 0) or
                  (target_checkpoint_epoch > 0 and
                   paired_draft_checkpoint_epoch == target_checkpoint_epoch))
        publish = not self.step_cancelled and checked_kernel_error_mask == 0 and paired
        if publish:
            self.visible_position = step.start_position + accepted_tokens
        # Rejected speculative rows remain inaccessible beyond visible_position;
        # the next append must overwrite them under the same stream/lease rules.
        self.step = None
        for identity in self.retired:
            self.free_slots.append(self.valid.pop(identity))
        self.retired.clear()
        self.step_cancelled = False
        if not publish:
            raise ContractError("cancelled, failed or unpaired step was not committed")
        return self.visible_position
