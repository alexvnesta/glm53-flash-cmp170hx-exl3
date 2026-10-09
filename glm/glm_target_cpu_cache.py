"""Demand-allocated LS target tier using pinned installed CPUPageCache methods.

Constructor policy and optional host-allocation failure handling differ. Page
hashing, stream-ordered raw copies, LRU/chain eviction, fetch and close use upstream.
The pinned source hash refuses an incompatible engine before host allocation.
"""
from collections import deque
import hashlib
from pathlib import Path
import threading
import torch

import exllamav3.generator.cpu_cache as native

SOURCE_SHA256 = "4ba765dff08ac77e40ee1849b581385b6a98727c560c536910019b4657bf5906"


def align(n, a):
    return (n + a - 1) // a * a


class HostAllocationUnavailable(MemoryError):
    pass


class LazyTargetCPUPageCache(native.CPUPageCache):
    reservation_policy = "pressure_demand_pinned_slabs"

    def __init__(self, caches, max_size):
        source = Path(native.__file__).read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("Lazy target tier requires the pinned installed CPUPageCache source")
        if len(caches) != 1 or caches[0].model.loaded_tp:
            raise ValueError("Lazy target tier supports exactly one LS target cache")
        if type(max_size) is not int or max_size <= 0:
            raise ValueError("Lazy target byte budget must be a positive integer")
        self.tp_groups = []
        self.tp = False
        self.segments = []
        offset = 0
        for layer in caches[0].layers.values():
            for t in layer.get_tensors():
                if t is None:
                    continue
                if t.device.type != "cuda":
                    raise ValueError("Load target cache tensors before attaching the host tier")
                page_shape = tuple(t.shape[1:])
                nbytes = t[0].numel() * t.element_size()
                self.segments.append((t, offset, page_shape, t.dtype))
                offset = align(offset + nbytes, 256)
        if not offset:
            raise ValueError("No paged target cache tensors")
        self.slab_size = self.slot_size = align(offset, 4096)
        self.max_slots = max_size // self.slot_size
        if self.max_slots < 2:
            raise ValueError("Target CPU tier must hold at least two complete page images")
        self.pagetable = None
        self.entries = {}
        self.slot_slabs = []
        self.slot_views = []
        self.free_slots = deque()
        self.num_slots = 0
        self._order = deque()
        self._order_pops = 0
        self._order_rebuild = max(64, self.max_slots // 8)
        self.metrics = {"pushes": 0, "dedup_hits": 0, "restores": 0,
                        "evictions": 0, "cold_allocs": 0, "pin_failures": 0, "skipped_spills": 0}
        self._growth_disabled = False
        self._local_cold_allocs = self._tp_cold_allocs = 0
        # Upstream _new_slot already has a synchronous no-spare fallback. With
        # no producer thread, exactly that path pins one page only on its first
        # actual eviction. Recycled slots reuse existing pinned slabs.
        self._spare = deque()
        self._spare_cond = threading.Condition()
        self._stop_event = threading.Event()
        self._alloc_thread = None

    @property
    def pinned_bytes(self):
        return len(self.slot_slabs) * self.slab_size

    def _make_slab(self):
        if self._growth_disabled:
            raise HostAllocationUnavailable("Host pin growth disabled after an allocation failure")
        try:
            slab = torch.empty((self.slab_size,), dtype=torch.uint8, pin_memory=True)
        except (MemoryError, RuntimeError) as error:
            message = str(error).lower()
            if (not isinstance(error, MemoryError) and not any(s in message for s in
                    ("out of memory", "cannot allocate memory", "memory allocation", "pin memory", "malloc"))):
                raise  # CUDA/context/shape faults must still fail the engine.
            self._growth_disabled = True
            self.metrics["pin_failures"] += 1
            raise HostAllocationUnavailable("Pinned host allocation unavailable") from error
        views = []
        for t, offset, page_shape, dtype in self.segments:
            nbytes = t[0].numel() * t.element_size()
            views.append(slab[offset:offset+nbytes].view(dtype).view(page_shape))
        return slab, views

    def _new_slot(self, protect):
        if self._growth_disabled and self.num_slots:
            return self.free_slots.popleft() if self.free_slots else self._evict_one(protect)
        return super()._new_slot(protect)

    def store(self, page, serial, protect=None):
        try:
            return super().store(page, serial, protect)
        except HostAllocationUnavailable:
            # Optional prefix preservation may degrade to cold replay. Throwing
            # mid-PageTable.allocate_pages would leave its partial claim list
            # inaccessible to the Job's cleanup, so host capacity failure must
            # not interrupt that native allocation transaction.
            self.metrics["skipped_spills"] += 1
