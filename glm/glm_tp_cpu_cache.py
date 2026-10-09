"""Experimental demand-allocated, target-only TP inactive KV overflow tier."""
from collections import deque
import threading
import uuid
import torch
import json
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.util.tp_lazy_cache import (
    mp_tp_lazy_init, mp_tp_lazy_prepare, mp_tp_lazy_drop,
    mp_tp_lazy_store, mp_tp_lazy_fetch, mp_tp_lazy_close,
)


class LazyTPTargetCPUPageCache(CPUPageCache):
    reservation_policy = "lazy_exact_registered_rank_slabs_on_gpu_eviction"

    def __init__(self, caches, max_size):
        if len(caches) != 1 or not caches[0].model.loaded_tp:
            raise ValueError("Lazy TP tier accepts exactly the actual TP target cache")
        if type(max_size) is not int or max_size <= 0:
            raise ValueError("Lazy TP target byte budget must be positive")
        self.cache = caches[0]
        self.model = self.cache.model
        self.owner = uuid.uuid4().hex
        self.cache_id = id(self.cache)
        self.max_pages = self.cache.max_num_tokens // 256
        self.cache_owner_serial = getattr(self.cache, "owner_serial", None)
        self.thread = None
        self._generator_owner = None
        self.pagetable = None
        self.segments, self.slot_views = [], []
        self.arena = None
        self.tp = True
        self.tp_groups = [(self.model, [self.cache_id])]
        self.entries = {}
        self.free_slots, self._order = deque(), deque()
        self.num_slots = self._order_pops = 0
        self._growth_disabled = self.closed = False
        self.metrics = dict(pushes=0, dedup_hits=0, restores=0, evictions=0, protected_spill_skips=0,
                            duplicate_evictions=0, pin_failures=0, skipped_spills=0)
        sizes = self._call(mp_tp_lazy_init, self.owner, self.cache_id, 0, self.max_pages)
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("Every TP rank must expose target paged tensors")
        self.rank_slot_bytes = tuple(sizes)
        self.slot_size = self.slab_size = sum(sizes)
        self.max_slots = max_size // self.slot_size
        if self.max_slots < 2:
            raise ValueError("Lazy TP target byte budget must hold at least two whole pages")
        self._order_rebuild = max(64, self.max_slots // 8)
        try:
            self._call(mp_tp_lazy_init, self.owner, self.cache_id, self.max_slots, self.max_pages)
        except BaseException as error:
            try:
                self._call(mp_tp_lazy_close, self.owner, self.cache_id)
            except BaseException as cleanup_error:
                error.add_note(f"Partial TP tier initialization cleanup failed: {cleanup_error!r}")
            raise

    def _call(self, function, *args):
        """Drain every rank even when one fails; never leave stale pipe replies."""
        self.model.tp_drain_acks()
        devices = tuple(self.model.active_devices)
        dispatched, values, failures = [], [], []
        for device in devices:
            try:
                self.model.tp_worker_dispatch(device, function, args)
                dispatched.append(device)
            except BaseException as error:
                failures.append(error)
        for device in dispatched:
            try:
                values.append(self.model.tp_worker_result(device))
            except BaseException as error:
                failures.append(error)
        if failures:
            for extra in failures[1:]:
                failures[0].add_note(f"Another TP rank failed: {extra!r}")
            raise failures[0]
        return values

    def attach(self, pagetable):
        g = pagetable.generator
        if (g.model is not self.model or g.cache is not self.cache or g.max_batch_size != 1
                or g.enable_defrag is not False or g.num_remaining_jobs()
                or getattr(g, "job_serial", 0) or pagetable.referenced_pages
                or pagetable.cpu_tier is not None
                or pagetable.cache is not self.cache or pagetable.max_pages != self.max_pages
                or len(pagetable.all_pages) != self.max_pages):
            raise ValueError("Lazy TP tier attach requires its fresh serialized fixed-page generator: "
                             + json.dumps(dict(enable_defrag=g.enable_defrag,
                                      max_batch_size=g.max_batch_size,
                                      target_tp=g.model.loaded_tp,
                                      target_cache_matches=g.cache is self.cache,
                                      expected_pages=self.max_pages,
                                      pagetable_pages=pagetable.max_pages,
                                      physical_page_count=len(pagetable.all_pages)), sort_keys=True))
        self._generator_owner = g
        self._page_geometry = (pagetable.all_pages, tuple(id(p) for p in pagetable.all_pages))
        self.pagetable = pagetable

    def _check_owner(self):
        g = self.pagetable.generator if self.pagetable is not None else None
        if (self.closed or g is not self._generator_owner or g.model is not self.model
                or g.cache is not self.cache
                or getattr(g.cache, "owner_serial", None) != self.cache_owner_serial
                or g.enable_defrag is not False
                or g.max_batch_size != 1 or g.cpu_page_cache is not self
                or self.pagetable.cpu_tier is not self or self.pagetable.cache is not self.cache
                or self.cache.max_num_tokens != self.max_pages * 256
                or self.pagetable.max_pages != self.max_pages
                or self.pagetable.all_pages is not self._page_geometry[0]
                or tuple(id(p) for p in self.pagetable.all_pages) != self._page_geometry[1]):
            raise ValueError("Lazy TP tier lost its generator/page-table ownership")
        thread = threading.get_ident()
        if self.thread is None:
            self.thread = thread
        elif self.thread != thread:
            raise ValueError("Lazy TP transfers require their original serialized engine thread")

    def _new_slot(self, protect):
        if self.free_slots:
            return self.free_slots.popleft()
        if self.num_slots >= self.max_slots:
            return self._evict_one(protect)
        if self._growth_disabled:
            return None
        slot = self.num_slots
        results = self._call(mp_tp_lazy_prepare, self.owner, self.cache_id, slot)
        if not all(result is True for result in results):
            self._call(mp_tp_lazy_drop, self.owner, self.cache_id, slot)
            self._growth_disabled = True
            self.metrics["pin_failures"] += 1
            return None
        self.num_slots += 1
        return slot

    def _referenced_duplicate(self, phash, entry):
        pt = self.pagetable
        page = pt.referenced_pages.get(phash)
        return (page is not None and page.pagetable is pt and page.ref_count > 0
                and page.phash == phash and page.kv_position == 256 and not page.can_revert
                and phash[:8] != bytes(8) and page.prev_hash == entry["prev_hash"]
                and type(page.page_index) is int and 0 <= page.page_index < pt.max_pages
                and pt.all_pages[page.page_index] is page
                and page.sequence.device.type == "cpu" and entry["tokens"].device.type == "cpu"
                and torch.equal(page.sequence, entry["tokens"]))

    def _evict_one(self, protect):
        self._check_owner()
        # Complete referenced GPU duplicates are protected from native eviction.
        # Each rank's fixed stream orders a prior H2D read before slot reuse D2H.
        candidates = sorted(tuple(self.entries.items()),
                            key=lambda item: (item[1]["access_serial"], item[1]["slot"], item[0]))
        for phash, entry in candidates:
            if self._referenced_duplicate(phash, entry):
                if self.entries.get(phash) is not entry:
                    raise ValueError("TP duplicate host entry changed")
                del self.entries[phash]
                self.metrics["evictions"] += 1
                self.metrics["duplicate_evictions"] += 1
                self._order.clear()
                return entry["slot"]
        if protect and all(phash in protect for phash in self.entries):
            self.metrics["protected_spill_skips"] += 1
            return None
        self._build_order()
        return super()._evict_one(protect)

    def store(self, page, serial, protect=None):
        self._check_owner()
        entry = self.entries.get(page.phash)
        if entry is not None:
            entry["access_serial"] = serial
            self.metrics["dedup_hits"] += 1
            return
        slot = self._new_slot(protect)
        if slot is None:
            self.metrics["skipped_spills"] += 1
            return
        self._call(mp_tp_lazy_store, self.owner, self.cache_id, slot, page.page_index)
        self.entries[page.phash] = dict(slot=slot, prev_hash=page.prev_hash,
                                       access_serial=serial, tokens=page.sequence.clone())
        self.metrics["pushes"] += 1

    def fetch(self, phash, page_index, serial):
        self._check_owner()
        entry = self.entries[phash]
        self._call(mp_tp_lazy_fetch, self.owner, self.cache_id, entry["slot"], page_index)
        entry["access_serial"] = serial
        self.metrics["restores"] += 1
        return entry

    def stats(self):
        return dict(budget_slots=self.max_slots, slot_bytes=self.slot_size,
                    rank_slot_bytes=list(self.rank_slot_bytes),
                    slab_budget_bytes=self.max_slots * self.slot_size,
                    reserved_budget_bytes=self.max_slots * self.slot_size,
                    pinned_bytes=self.num_slots * self.slot_size,
                    registered_mapping_bytes=self.num_slots * self.slot_size,
                    token_snapshot_bytes=sum(e["tokens"].numel() * e["tokens"].element_size()
                                             for e in tuple(self.entries.values())),
                    pin_growth_disabled=self._growth_disabled, resident_entries=len(self),
                    metrics=dict(self.metrics), reservation_policy=self.reservation_policy,
                    transfer_policy="unreferenced_page_eviction_only")

    def close(self):
        if not self.closed:
            self._call(mp_tp_lazy_close, self.owner, self.cache_id)
            self.closed = True
            self.entries.clear()
            self.num_slots = 0
            self.pagetable = self._generator_owner = None
