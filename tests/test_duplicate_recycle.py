"""Run pinned native PageTable/CPUPageCache methods on CPU tensors only."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# The existing portable fixture executes the pinned native methods from the
# explicit EXLLAMAV3_ENGINE_ROOT checkout without importing the engine.
import test_sessions as base
pt_stub = ModuleType("exllamav3.generator.pagetable")
pt_stub.__file__ = str(base.ENGINE / "generator/pagetable.py")
pt_stub.PageTable = base.native["PageTable"]
stubs = {base.engine_stub.__name__: base.engine_stub,
         base.generator_stub.__name__: base.generator_stub,
         base.cache_stub.__name__: base.cache_stub, pt_stub.__name__: pt_stub}
spec = importlib.util.spec_from_file_location("duplicate_tier", REPO / "glm/glm_target_cpu_cache.py")
tier_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tier_module
with patch.dict(sys.modules, stubs):
    spec.loader.exec_module(tier_module)
TRANSFER_SNAPSHOT = tier_module.current_transfer_streams


class FakeCuda(base.FakeCuda):
    counter = 0
    def __init__(self, tensor):
        super().__init__(tensor)
        self.device_index = FakeCuda.counter % 2
        FakeCuda.counter += 1
    @property
    def device(self): return torch.device("cuda", self.device_index)
    def data_ptr(self): return self.tensor.data_ptr()


class Fixture(base.Fixture):
    def __init__(self, enabled=True, slots=10, **kwargs):
        self.extra_patches = [
            patch.dict(os.environ, {"GLM53_CPU_DUPLICATE_RECYCLE": "1" if enabled else "0"}),
            patch.dict(sys.modules, stubs),
            patch.object(base, "FakeCuda", FakeCuda),
            patch.object(base.lazy, "LazyTargetCPUPageCache", tier_module.LazyTargetCPUPageCache),
            patch.object(tier_module, "current_transfer_streams", return_value=(("cuda:0", 11), ("cuda:1", 22)))
        ]
        for p in self.extra_patches: p.start()
        try:
            FakeCuda.counter = 0
            super().__init__(tier_budget=slots*8192, **kwargs)
        except BaseException:
            for p in reversed(self.extra_patches): p.stop()
            raise
    def close(self):
        if getattr(self, "closed", False): return
        try: super().close()
        finally:
            for p in reversed(self.extra_patches): p.stop()


class DuplicateRecycle(unittest.TestCase):
    def setUp(self): self.f = Fixture()
    def tearDown(self): self.f.close()

    def duplicate(self):
        job = self.f.start(1)
        self.f.checkpoint(job)
        page = job.sequences[0].allocated_pages[0]
        tier = self.f.g.cpu_page_cache
        tier.store(page, 10)
        return job, page, tier, tier.entries[page.phash]

    def test_default_off_preserves_native_policy(self):
        self.f.close(); self.f = Fixture(enabled=False)
        job, page, tier, entry = self.duplicate()
        other = job.sequences[0].allocated_pages[1]
        tier.store(other, 20)
        slot = tier._evict_one({page.phash})
        self.assertEqual(slot, 1)
        self.assertIn(page.phash, tier.entries)
        self.assertEqual(tier.metrics["duplicate_evictions"], 0)
        self.f.finish(job)

    def test_protected_referenced_duplicate_is_recycled(self):
        job, page, tier, entry = self.duplicate()
        self.assertEqual(tier._evict_one({page.phash}), entry["slot"])
        self.assertNotIn(page.phash, tier.entries)
        self.assertIs(self.f.g.pagetable.referenced_pages[page.phash], page)
        self.assertEqual(tier.metrics["duplicate_evictions"], 1)
        self.assertEqual(tier.metrics["evictions"], 1)
        self.f.finish(job)

    def test_unreferenced_complete_gpu_page_is_not_eligible(self):
        job, page, tier, entry = self.duplicate()
        self.f.finish(job)
        self.assertFalse(tier._referenced_duplicate(page.phash, entry))
        with self.assertRaises(tier_module.ProtectedHostEntries):
            tier._evict_one({page.phash})
        self.assertIs(tier.entries[page.phash], entry)
        self.assertEqual(tier.metrics["duplicate_evictions"], 0)

    def test_incomplete_revert_foreign_parent_tokens_and_owner_are_ineligible(self):
        job, page, tier, entry = self.duplicate()
        for field, value in (("kv_position", 255), ("can_revert", True), ("ref_count", 0),
                             ("prev_hash", b"foreign"), ("pagetable", object()),
                             ("page_index", -1), ("page_index", 32)):
            old = getattr(page, field)
            setattr(page, field, value)
            self.assertFalse(tier._referenced_duplicate(page.phash, entry), field)
            setattr(page, field, old)
        old = page.sequence.clone(); page.sequence.add_(1)
        self.assertFalse(tier._referenced_duplicate(page.phash, entry))
        page.sequence.copy_(old)
        self.assertTrue(tier._referenced_duplicate(page.phash, entry))
        self.f.finish(job)

    def test_hash_reassignment_and_page_alias_are_ineligible(self):
        job, page, tier, entry = self.duplicate()
        h = page.phash; page.phash = bytes(16)
        self.assertFalse(tier._referenced_duplicate(h, entry))
        page.phash = h
        old = self.f.g.pagetable.all_pages[page.page_index]
        self.f.g.pagetable.all_pages[page.page_index] = object()
        self.assertFalse(tier._referenced_duplicate(h, entry))
        self.f.g.pagetable.all_pages[page.page_index] = old
        self.f.finish(job)

    def test_all_protected_sole_host_copies_are_preserved(self):
        job, page, tier, entry = self.duplicate()
        self.f.finish(job)
        with self.assertRaises(tier_module.ProtectedHostEntries):
            tier._evict_one(set(tier.entries))
        self.assertIs(tier.entries[page.phash], entry)
        self.assertEqual(tier.metrics["protected_spill_skips"], 1)

    def test_all_protected_capacity_skips_spill_without_losing_host_chain(self):
        self.f.close(); self.f = Fixture(slots=2)
        job = self.f.start(1); self.f.checkpoint(job)
        pages = job.sequences[0].allocated_pages
        tier = self.f.g.cpu_page_cache
        tier.store(pages[0], 10); tier.store(pages[1], 20)
        self.f.finish(job)
        entries = dict(tier.entries)
        tier.store(pages[2], 30, set(entries))
        self.assertEqual(tier.entries, entries)
        self.assertEqual(tier.metrics["skipped_spills"], 1)
        self.assertEqual(tier.metrics["protected_spill_skips"], 1)
        self.assertEqual(tier.pinned_bytes, 2*8192)
        self.assertEqual(tier.metrics["pin_failures"], 0)

    def test_raw_slot_reuse_preserves_prior_fetch_for_both_device_segments(self):
        self.f.close(); self.f = Fixture(slots=2)
        job = self.f.start(1); self.f.checkpoint(job)
        a, b, c = job.sequences[0].allocated_pages[:3]
        tier = self.f.g.cpu_page_cache
        for plane, tensor in enumerate(self.f.g.cache.tensors):
            tensor[a.page_index].fill_(100 + plane)
            tensor[c.page_index].fill_(900 + plane)
        tier.store(a, 10); tier.store(b, 20)
        old_slot = tier.entries[a.phash]["slot"]
        tier.fetch(a.phash, 31, 15)
        tier.store(c, 30, {a.phash, b.phash})
        self.assertNotIn(a.phash, tier.entries)
        self.assertEqual(tier.entries[c.phash]["slot"], old_slot)
        ranges = []
        for plane, (tensor, offset, _, _) in enumerate(tier.segments):
            self.assertTrue(torch.equal(tensor[31], torch.full_like(tensor[31], 100 + plane)))
            self.assertTrue(torch.equal(tier.slot_views[old_slot][plane], torch.full_like(tensor[31], 900 + plane)))
            ranges.append((offset, offset + tensor[0].numel()*tensor.element_size(), str(tensor.device)))
        self.assertEqual({r[2] for r in ranges}, {"cuda:0", "cuda:1"})
        self.assertTrue(all(left[1] <= right[0] for left,right in zip(ranges,ranges[1:])))
        self.f.finish(job)

    def test_native_lru_fallback_uses_unprotected_entry(self):
        job, page, tier, entry = self.duplicate()
        other = job.sequences[0].allocated_pages[1]
        tier.store(other, 20)
        self.f.finish(job)
        slot = tier._evict_one({page.phash})
        self.assertEqual(slot, 1)
        self.assertIn(page.phash, tier.entries)
        self.assertEqual(tier.metrics["duplicate_evictions"], 0)

    def test_fresh_native_order_includes_new_unprotected_entry(self):
        job, page, tier, entry = self.duplicate()
        tier._build_order()
        other = job.sequences[0].allocated_pages[1]
        tier.store(other, 20)
        self.f.finish(job)
        self.assertNotIn(other.phash, tier._order)
        self.assertEqual(tier._evict_one({page.phash}), 1)
        self.assertIn(page.phash, tier.entries)

    def test_stream_drift_refuses_fetch_and_store_before_mutation(self):
        job, page, tier, entry = self.duplicate()
        before = dict(tier.metrics); image = tier.slot_slabs[0].clone()
        with patch.object(tier_module, "current_transfer_streams", return_value=(("cuda:0", 99), ("cuda:1", 22))):
            with self.assertRaisesRegex(ValueError, "thread/stream"):
                tier.fetch(page.phash, 1, 20)
            with self.assertRaisesRegex(ValueError, "thread/stream"):
                tier.store(job.sequences[0].allocated_pages[1], 30)
        self.assertEqual(before, tier.metrics)
        self.assertTrue(torch.equal(image, tier.slot_slabs[0]))
        self.assertIs(tier.entries[page.phash], entry)
        self.f.finish(job)

    def test_other_transfer_thread_refuses_before_mutation(self):
        job, page, tier, entry = self.duplicate()
        result = []
        def other():
            try: tier.fetch(page.phash, 1, 20)
            except ValueError as error: result.append(str(error))
        worker = threading.Thread(target=other); worker.start(); worker.join()
        self.assertEqual(len(result), 1)
        self.assertIn("thread/stream", result[0])
        self.assertEqual(tier.metrics["restores"], 0)
        self.f.finish(job)

    def test_geometry_and_storage_drift_refuse_before_slot_reuse(self):
        job, page, tier, entry = self.duplicate()
        pt = self.f.g.pagetable
        for owner, field, value in ((pt, "max_pages", 31), (pt.generator, "enable_defrag", True),
                                    (pt.generator, "max_batch_size", 2)):
            old = getattr(owner, field); setattr(owner, field, value)
            with self.assertRaisesRegex(ValueError, "geometry"):
                tier._evict_one({page.phash})
            setattr(owner, field, old)
        tensor = self.f.g.cache.tensors[0]
        old = tensor.tensor; tensor.tensor = old.clone()
        with self.assertRaisesRegex(ValueError, "geometry"):
            tier.fetch(page.phash, 1, 20)
        tensor.tensor = old
        self.assertIn(page.phash, tier.entries)
        self.f.finish(job)

    def test_target_tensor_and_generator_owner_drift_refuse_before_transfer(self):
        job, page, tier, entry = self.duplicate()
        original = self.f.g.cache.tensors
        self.f.g.cache.tensors = tuple(FakeCuda(t.tensor.clone()) for t in original)
        with self.assertRaisesRegex(ValueError, "geometry"):
            tier.fetch(page.phash, 1, 20)
        self.f.g.cache.tensors = original
        for field, value in (("cache", object()), ("model", object())):
            old = getattr(self.f.g, field); setattr(self.f.g, field, value)
            with self.assertRaisesRegex(ValueError, "geometry"):
                tier.store(job.sequences[0].allocated_pages[1], 20)
            setattr(self.f.g, field, old)
        self.assertEqual(tier.metrics["restores"], 0)
        self.assertIs(tier.entries[page.phash], entry)
        self.f.finish(job)

    def test_replaced_generator_with_same_cache_and_model_refuses_before_transfer(self):
        job, page, tier, entry = self.duplicate()
        pt = self.f.g.pagetable
        original = pt.generator
        pt.generator = SimpleNamespace(**original.__dict__)
        try:
            with self.assertRaisesRegex(ValueError, "geometry"):
                tier.fetch(page.phash, 1, 20)
            with self.assertRaisesRegex(ValueError, "geometry"):
                tier.store(job.sequences[0].allocated_pages[1], 30)
            self.assertEqual(tier.metrics["restores"], 0)
            self.assertIs(tier.entries[page.phash], entry)
        finally: pt.generator = original
        self.f.finish(job)

    def test_pinned_pagetable_source_refuses_before_attachment(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            p = Path(tmp)/"changed.py"; p.write_text("changed")
            with patch.object(pt_stub, "__file__", str(p)):
                tier = tier_module.LazyTargetCPUPageCache([self.f.g.cache], 100000)
                try:
                    with self.assertRaisesRegex(ValueError, "pinned"):
                        tier.attach(self.f.g.pagetable)
                    self.assertIsNone(tier.pagetable)
                    self.assertEqual(tier.pinned_bytes, 0)
                finally: tier.close()

    def test_serialized_fixed_page_geometry_required_at_attach(self):
        self.f.g.enable_defrag = True
        tier = tier_module.LazyTargetCPUPageCache([self.f.g.cache], 100000)
        try:
            with self.assertRaisesRegex(ValueError, "geometry"):
                tier.attach(self.f.g.pagetable)
            self.assertEqual(tier.pinned_bytes, 0)
        finally:
            tier.close(); self.f.g.enable_defrag = False

    def test_invalid_opt_in_refuses_before_allocation(self):
        with patch.dict(os.environ, {"GLM53_CPU_DUPLICATE_RECYCLE": "yes"}):
            with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
                tier_module.LazyTargetCPUPageCache([self.f.g.cache], 100000)

    def test_stream_snapshot_reads_current_stream_once_per_fixed_device(self):
        with patch.object(torch.cuda, "current_stream", side_effect=lambda device: type("S",(),{"cuda_stream":device.index+10})()) as current:
            result = TRANSFER_SNAPSHOT(self.f.g.cpu_page_cache.segments)
        self.assertEqual(result, (("cuda:0", 10), ("cuda:1", 11)))
        self.assertEqual([call.args[0] for call in current.call_args_list], [torch.device("cuda:0"),torch.device("cuda:1")])

    def test_no_pressure_allocates_no_host_slab(self):
        self.f.completed(1)
        self.f.completed(2)
        tier = self.f.g.cpu_page_cache
        self.assertEqual(tier.pinned_bytes, 0)
        self.assertEqual(tier.metrics["duplicate_evictions"], 0)
        self.assertIsNone(tier._alloc_thread)

    def test_two_session_pressure_retains_raw_planes_and_paired_states(self):
        a, _, ka, da = self.f.completed(1, 5120)
        b, _, kb, db = self.f.completed(2, 5120)
        tier = self.f.g.cpu_page_cache
        self.assertEqual(tier.metrics["pushes"], 9)
        job = self.f.start(1, 5120)
        self.assertEqual(job.cached_pages, 20)
        self.assertGreater(tier.metrics["duplicate_evictions"], 0)
        self.assertTrue(torch.equal(self.f.g.cache.layer.tensor, ka))
        self.assertIn(b.identity, self.f.manager.registry)
        self.f.finish(job)
        job = self.f.start(2, 5120)
        self.assertEqual(job.cached_pages, 20)
        self.assertTrue(torch.equal(self.f.g.cache.layer.tensor, kb))
        for p, page in enumerate(job.sequences[0].allocated_pages[:20]):
            v = int(job.sequences[0].sequence_ids.torch()[0,p*256]) % 500
            for plane, tensor in enumerate(self.f.g.cache.tensors):
                self.assertTrue(torch.equal(tensor[page.page_index], torch.full_like(tensor[page.page_index], v + plane*10)))
        for row, index in enumerate(b.page_indices):
            self.assertTrue(torch.equal(self.f.ring[index], db[0][row]))
        self.assertLessEqual(tier.pinned_bytes, 10*8192)
        self.assertEqual(tier.metrics["pin_failures"], 0)
        self.f.finish(job)

    def test_default_policy_loses_second_session_at_same_small_budget(self):
        self.f.close(); self.f = Fixture(enabled=False, slots=17)
        self.f.completed(1, 5120)
        b, _, _, _ = self.f.completed(2, 5120)
        job = self.f.start(1, 5120)
        self.assertEqual(job.cached_pages, 20)
        self.f.finish(job)
        self.assertNotIn(b.identity, self.f.manager.registry)
        job = self.f.start(2, 5120)
        self.assertEqual(job.cached_pages, 0)
        self.f.finish(job)

    def test_optional_policy_preserves_second_session_at_matching_budget(self):
        self.f.close(); self.f = Fixture(slots=17)
        self.f.completed(1, 5120)
        b, _, _, _ = self.f.completed(2, 5120)
        job = self.f.start(1, 5120)
        self.assertEqual(job.cached_pages, 20)
        self.f.finish(job)
        self.assertIn(b.identity, self.f.manager.registry)
        job = self.f.start(2, 5120)
        self.assertEqual(job.cached_pages, 20)
        self.assertGreater(self.f.g.cpu_page_cache.metrics["duplicate_evictions"], 0)
        self.f.finish(job)

    def test_pressure_unrelated_miss_cancel_then_paired_recovery(self):
        self.f.close(); self.f = Fixture(slots=17)
        self.f.completed(1, 5120)
        b, _, kb, db = self.f.completed(2, 5120)
        job = self.f.start(1, 5120)
        self.assertEqual(job.cached_pages, 20); self.f.finish(job)
        self.f.completed(3, 1024)
        job = self.f.start(2, 5120)
        self.assertEqual(job.cached_pages, 20); self.f.finish(job)
        cancelled = self.f.start(4, 1024); self.f.checkpoint(cancelled)
        self.f.manager.invalidate("cancelled"); cancelled.deallocate_pages()
        self.f.g.recurrent_cache.prune_stranded()
        job = self.f.start(2, 5120)
        self.assertEqual(job.cached_pages, 20)
        self.assertTrue(torch.equal(self.f.g.cache.layer.tensor, kb))
        for row, index in enumerate(b.page_indices):
            self.assertTrue(torch.equal(self.f.ring[index], db[0][row]))
        self.assertGreater(self.f.g.cpu_page_cache.metrics["duplicate_evictions"], 0)
        self.f.finish(job)


if __name__ == "__main__":
    assert not torch.cuda.is_initialized()
    result = unittest.main(exit=False)
    assert not torch.cuda.is_initialized()
    print("cuda_initialized=False")
    raise SystemExit(not result.result.wasSuccessful())
