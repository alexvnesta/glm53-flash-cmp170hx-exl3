"""CPU source-level integration, with real native page/cache/state methods.

Torch uses CPU tensors only. Fake device labels and dropping pin_memory in the
test allocator let the installed CPUPageCache run without initializing CUDA.
No attention/model kernels or inference are exercised.
"""
import ast
from collections import OrderedDict, deque, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import importlib.util
from itertools import pairwise
import os
from pathlib import Path
import sys
import threading
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Explicit source input, never a workspace path or an engine import.
ENGINE = Path(os.environ["EXLLAMAV3_ENGINE_ROOT"]).resolve() / "exllamav3"
FULL = ENGINE
sys.path.insert(0, str(REPO / "glm"))
import glm_dflash_sessions as sessions
from glm_session_options import session_options


def extract(path, names, ns, methods=False):
    tree = ast.parse(path.read_text())
    source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Job").body if methods else tree.body
    nodes = [n for n in source if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), ns)


module = ModuleType("cpu_native_session_fixture")
sys.modules[module.__name__] = module
native = module.__dict__
native.update(torch=torch, OrderedDict=OrderedDict, deque=deque, defaultdict=defaultdict,
              PAGE_SIZE=256, threading=threading, weakref=weakref, heapq=heapq, dataclass=dataclass,
              pairwise=pairwise, hashlib=hashlib, os=os, _uniquehash=0,
              _TRIM_THRESHOLD=256 * 1024**2, _freed_bytes=0, malloc_trim=lambda:None)
extract(ENGINE / "cache/recurrent.py", {"HostPool", "RecurrentCache", "host_copy", "note_freed"}, native)
native["host_pool"] = native["HostPool"]()
extract(ENGINE / "generator/pagetable.py", {"_tensor_blake2b_checksum", "_randomhash", "is_content_hash",
                                          "CachePage", "PageTable", "Sequence"}, native)
native["tensor_hash_checksum"] = native["_tensor_blake2b_checksum"]
extract(ENGINE / "generator/cpu_cache.py", {"_align", "_alloc_worker", "_stop_worker", "CPUPageCache"}, native)
extract(FULL / "modules/gated_delta_net.py", {"GDNState"}, native)
extract(ENGINE / "generator/job.py", {"allocate_pages", "deallocate_pages", "free_recurrent_state",
                                     "maybe_stash_recurrent", "is_checkpoint_boundary"}, native, methods=True)

cache_stub=ModuleType("exllamav3.generator.cpu_cache")
cache_stub.__file__=str(ENGINE / "generator/cpu_cache.py")
cache_stub.CPUPageCache=native["CPUPageCache"]
generator_stub=ModuleType("exllamav3.generator")
generator_stub.__path__=[]
generator_stub.cpu_cache=cache_stub
engine_stub=ModuleType("exllamav3")
engine_stub.__path__=[]
engine_stub.generator=generator_stub
with patch.dict(sys.modules,{engine_stub.__name__:engine_stub,
                             generator_stub.__name__:generator_stub,cache_stub.__name__:cache_stub}):
    import glm_target_cpu_cache as lazy


class TokenIds:
    def __init__(self, tensor): self.tensor = tensor
    def __len__(self): return self.tensor.shape[-1]
    def torch(self): return self.tensor
    def torch_slice(self, lo, hi): return self.tensor[:, lo:hi]


class FakeCuda:
    """A label wrapper only; every slice is an actual CPU tensor."""
    def __init__(self, tensor): self.tensor = tensor
    @property
    def shape(self): return self.tensor.shape
    @property
    def dtype(self): return self.tensor.dtype
    @property
    def device(self): return NS(type="cuda")
    def __getitem__(self, key): return self.tensor[key]
    def numel(self): return self.tensor.numel()
    def element_size(self): return self.tensor.element_size()


class Layer:
    def __init__(self): self.tensor = torch.zeros(4)
    def clear(self, slot): self.tensor.zero_()
    def get_checkpoint_size(self): return 16
    def stash(self, slot): return native["host_copy"](self.tensor)
    def unstash(self, slot, saved): self.tensor.copy_(saved)


class Cache:
    def __init__(self, model, pages=32):
        self.model = model
        self.max_num_tokens = pages * 256
        self.layer = Layer()
        # Packed values, scales, indexer plane and pooled auxiliary plane.
        tensors = (torch.zeros(pages,256,2,dtype=torch.int32),
                   torch.zeros(pages,256,1,dtype=torch.float16),
                   torch.zeros(pages,256,4,dtype=torch.float16),
                   torch.zeros(pages,64,2,dtype=torch.float16))
        self.tensors = tuple(FakeCuda(t) for t in tensors)
        self.layers = {0:NS(get_tensors=lambda:self.tensors)}
    def get_all_tensors(self): return self.tensors
    def get_all_recurrent_layers(self): return {0:self.layer}
    def get_new_state(self): return native["GDNState"](self, 0, 0)
    def new_from_stashed(self, stash, position): return native["GDNState"](self, 0, position, stashed=stash)
    def release_state(self, state): pass


class Job:
    allocate_pages = native["allocate_pages"]
    deallocate_pages = native["deallocate_pages"]
    free_recurrent_state = native["free_recurrent_state"]
    maybe_stash_recurrent = native["maybe_stash_recurrent"]
    is_checkpoint_boundary = native["is_checkpoint_boundary"]
    def __init__(self, generator, seed, position=2048):
        self.generator = generator
        self.pagetable = generator.pagetable
        values = torch.arange(position + 1).reshape(1,-1) + seed * 10000
        seq = native["Sequence"].__new__(native["Sequence"])
        seq.input_ids = seq.sequence_ids = TokenIds(values)
        seq.kv_position = 0
        seq.max_cached_pages = None
        seq.allocated_pages = []
        seq.prefill_complete = False
        self.sequences = [seq]
        self.is_requeued = False
        self.embeddings = self.filters = self.banned_strings = None
        self.prefix_token = self.orig_max_rq_tokens = None
        self.recurrent_state = None
        self.cached_pages = self.cached_tokens = self.total_pages = self.non_sequential_pages = 0
        self.last_recurrent_checkpoint_pos = None
        self.all_unique_hashes = []
        self.expected_position = position


class Fixture:
    def __init__(self, target_cpu=True, budget=1024**2, max_checkpoints=8, tier_budget=64*8192):
        self.inference_context=torch.inference_mode()
        self.inference_context.__enter__()
        self.closed=False
        self.empty_original = torch.empty
        self.empty_patch = patch.object(torch, "empty", self.cpu_empty)
        self.empty_patch.start()
        model = NS(loaded_tp=False)
        cache = Cache(model)
        self.ring = torch.zeros(17,256,1)
        draft = NS(dflash_ring_tokens=4352, get_all_tensors=lambda:(self.ring,))
        self.g = NS(model=model, cache=cache, draft_cache=draft, dflash_draft=True,
                    max_batch_size=1, max_chunk_size=256, cpu_page_cache=None, enable_defrag=False,
                    recurrent_checkpoint_interval=2048, recurrent_checkpoint_interval_pp=32768,
                    active_jobs=[], pending_jobs=[], num_remaining_jobs=lambda:0)
        self.g.pagetable = native["PageTable"](self.g, cache)
        self.g.recurrent_cache = native["RecurrentCache"](model, 1024**2)
        self.g.recurrent_cache.pagetable = self.g.pagetable
        if target_cpu:
            sessions.attach_target_cpu_tier(self.g, tier_budget, lazy.LazyTargetCPUPageCache)
        self.manager = sessions.MultiSessionCache(self.g, budget, max_checkpoints, "fixture")
        rc_module = ModuleType("exllamav3.cache.recurrent")
        rc_module.host_pool = native["host_pool"]
        rc_module.note_freed = native["note_freed"]
        self.modules_patch = patch.dict(sys.modules, {rc_module.__name__:rc_module})
        self.modules_patch.start()

    def cpu_empty(self, *args, **kwargs):
        kwargs.pop("pin_memory", None)
        return self.empty_original(*args, **kwargs)

    def close(self):
        if self.closed: return
        self.closed=True
        self.manager.close()
        if self.g.cpu_page_cache is not None: self.g.cpu_page_cache.close()
        self.modules_patch.stop()
        self.empty_patch.stop()
        native["host_pool"].release()
        self.inference_context.__exit__(None,None,None)

    def start(self, seed, position=2048, before_allocate=None, unsupported=False):
        job = Job(self.g, seed, position)
        if unsupported: job.filters = [object()]
        self.manager.begin(job)
        seq = job.sequences[0]
        hashes, unique = seq.prepare(False, 8)
        job.all_unique_hashes = list(hashes)
        self.manager.restrict(job)
        if before_allocate: before_allocate(job)
        job.allocate_pages()
        self.manager.restore(job, Job.allocate_pages)
        return job

    def checkpoint(self, job):
        seq = job.sequences[0]
        for p, page in enumerate(seq.allocated_pages[:job.expected_position // 256]):
            page.kv_position = 256
            page.prev_hash = seq.allocated_pages[p-1].phash if p else None
            page.sequence.copy_(seq.sequence_ids.torch_slice(p*256,(p+1)*256))
            h=seq.page_hashes[p]
            duplicate=self.g.pagetable.unreferenced_pages.get(h)
            if duplicate is not None and duplicate is not page:
                duplicate.clear()
            if page.phash != h:
                page.update_hash(h)
            v = int(page.sequence[0,0]) % 500
            for plane, tensor in enumerate(self.g.cache.tensors): tensor[page.page_index].fill_(v + plane * 10)
        seq.kv_position = job.expected_position
        job.recurrent_state.position = job.expected_position
        self.g.cache.layer.tensor.fill_(int(seq.sequence_ids.torch()[0,0]) + job.expected_position)
        self.ring.fill_(int(seq.sequence_ids.torch()[0,0]) % 500 + job.expected_position)
        self.manager.before_stash(job, 256)
        job.maybe_stash_recurrent(self.g.recurrent_cache, 256)
        self.manager.capture(job)

    def finish(self, job):
        job.deallocate_pages()
        # The native idle transition prunes before wrapper finish publishes.
        self.g.recurrent_cache.prune_stranded()
        self.manager.finish([{"job":job,"eos":True}])

    def completed(self, seed, position=2048):
        job = self.start(seed, position)
        self.checkpoint(job)
        cp = self.manager.prompt_checkpoint
        target = tuple(t.tensor.clone() for t in self.g.cache.tensors)
        recurrent = self.g.cache.layer.tensor.clone()
        draft = tuple(t.clone() for t in cp.tensors)
        self.finish(job)
        return cp, target, recurrent, draft


class Sessions(unittest.TestCase):
    def setUp(self): self.f = Fixture()
    def tearDown(self): self.f.close()

    def test_two_sessions_resume_without_spill(self):
        a, _, ka, da = self.f.completed(1)
        b, _, _, _ = self.f.completed(2)
        self.assertEqual(len(self.f.manager.registry), 2)
        self.assertEqual(self.f.g.cpu_page_cache.metrics["pushes"], 0)
        self.f.ring.fill_(-999)
        job = self.f.start(1)
        self.assertEqual(job.cached_pages, 8)
        self.assertTrue(torch.equal(self.f.g.cache.layer.tensor,ka))
        for row, index in enumerate(a.page_indices): self.assertTrue(torch.equal(self.f.ring[index],da[0][row]))
        self.f.finish(job)
        self.assertIn(b.identity,self.f.manager.registry)

    def test_unrelated_miss_preserves_previous_session(self):
        a, _, _, _ = self.f.completed(1)
        before = [p.phash for p in self.f.g.pagetable.all_pages]
        job = self.f.start(3)
        self.assertEqual(job.cached_pages,0)
        self.assertIn(a.identity,self.f.manager.registry)
        self.assertTrue(self.f.g.pagetable.is_resumable(a.key))
        self.assertGreater(len(set(before)&{p.phash for p in self.f.g.pagetable.all_pages}),8)
        self.f.manager.invalidate("cancelled")
        job.deallocate_pages()

    def test_cancellation_discards_only_unpublished_pair(self):
        a, _, _, _ = self.f.completed(1)
        job = self.f.start(2)
        self.f.checkpoint(job)
        unfinished = self.f.manager.prompt_checkpoint
        self.f.manager.invalidate("cancelled")
        job.deallocate_pages()
        self.assertIn(a.identity,self.f.manager.registry)
        self.assertNotIn(unfinished.identity,self.f.manager.registry)
        self.assertNotIn(unfinished.key,self.f.g.recurrent_cache)

    def test_error_discards_only_unpublished_pair(self):
        a, _, _, _ = self.f.completed(1)
        job=self.f.start(2)
        self.f.checkpoint(job)
        self.f.manager.finish([{"job":job,"stage":"error"}])
        job.deallocate_pages()
        self.assertIn(a.identity,self.f.manager.registry)
        self.assertIsNone(self.f.manager.current_job)

    def test_unsupported_request_disables_unpaired_native_reuse(self):
        a, _, _, _=self.f.completed(1)
        job=self.f.start(1,unsupported=True)
        self.assertEqual(job.cached_pages,0)
        self.assertEqual(job.sequences[0].max_cached_pages,0)
        self.f.finish(job)
        self.assertIn(a.identity,self.f.manager.registry)

    def test_native_epoch_recreation_rejects_old_ring(self):
        cp, _, _, _=self.f.completed(1)
        old=self.f.g.recurrent_cache.pop(cp.key)
        self.f.g.recurrent_cache[cp.key]={**old,sessions.COOKIE:"foreign-new-epoch"}
        job=self.f.start(1)
        self.assertEqual(job.cached_pages,0)
        self.assertNotIn(cp.identity,self.f.manager.registry)
        self.f.finish(job)

    def test_checkpoint_eviction_prunes_draft_pair(self):
        cp, _, _, _=self.f.completed(1)
        self.f.g.recurrent_cache.pop(cp.key)
        job=self.f.start(1)
        self.assertEqual(job.cached_pages,0)
        self.assertEqual(len(self.f.manager.registry),0)
        self.f.finish(job)

    def test_cpu_spill_restores_all_raw_planes_and_checkpoint(self):
        cp, _, ka, da=self.f.completed(1,4096)
        self.f.completed(2,4096)
        self.f.completed(3,4096)
        tier=self.f.g.cpu_page_cache
        self.assertGreater(tier.metrics["pushes"],0)
        self.assertTrue(self.f.g.pagetable.is_resumable(cp.key))
        self.assertIn(cp.identity,self.f.manager.registry)
        job=self.f.start(1,4096)
        self.assertEqual(job.cached_pages,16)
        self.assertGreater(tier.metrics["restores"],0)
        self.assertTrue(torch.equal(self.f.g.cache.layer.tensor,ka))
        for p,page in enumerate(job.sequences[0].allocated_pages[:16]):
            v=int(job.sequences[0].sequence_ids.torch()[0,p*256])%500
            for plane,t in enumerate(self.f.g.cache.tensors):
                self.assertTrue(torch.equal(t[page.page_index],torch.full_like(t[page.page_index],v+plane*10)))
        for row,index in enumerate(cp.page_indices): self.assertTrue(torch.equal(self.f.ring[index],da[0][row]))
        self.f.finish(job)

    def test_active_pages_never_enter_native_eviction_order(self):
        job=self.f.start(1)
        self.f.checkpoint(job)
        order=self.f.g.pagetable.build_eviction_order()
        self.assertTrue(all(p.ref_count==0 for p in order))
        self.assertFalse({id(p) for p in order}&{id(p) for p in job.sequences[0].allocated_pages})
        self.f.finish(job)

    def test_allocation_race_falls_back_cold(self):
        cp,_,_,_=self.f.completed(1)
        def replace(job): self.f.g.recurrent_cache[cp.key][sessions.COOKIE]="replaced"
        job=self.f.start(1,before_allocate=replace)
        self.assertEqual(job.cached_pages,0)
        self.assertEqual(job.sequences[0].kv_position,0)
        self.assertEqual(self.f.manager.last_reason,"allocation_epoch_miss")
        self.f.finish(job)

    def test_lru_checkpoint_bound_releases_owned_native_state(self):
        self.f.manager.max_checkpoints=2
        a,_,_,_=self.f.completed(1)
        self.f.completed(2)
        self.f.completed(3)
        self.assertEqual(len(self.f.manager.registry),2)
        self.assertNotIn(a.identity,self.f.manager.registry)
        self.assertNotIn(a.key,self.f.g.recurrent_cache)
        self.assertGreater(self.f.manager.native_releases,0)

    def test_byte_bound_evicts_before_new_capture(self):
        self.f.manager.max_bytes=150000
        self.f.completed(1,4096)
        self.f.completed(2,4096)
        self.f.completed(3,4096)
        self.f.completed(4,4096)
        self.assertLessEqual(self.f.manager.stats()["paired_host_bytes"],150000)
        self.assertLessEqual(len(self.f.manager.registry),3)
        self.assertGreater(self.f.manager.evictions,0)

    def test_stale_pair_does_not_free_recreated_checkpoint(self):
        cp,_,_,_=self.f.completed(1)
        stash=self.f.g.recurrent_cache[cp.key]
        replacement={**stash,sessions.COOKIE:"replacement"}
        self.f.g.recurrent_cache[cp.key]=replacement
        self.f.manager._prune()
        self.assertIs(self.f.g.recurrent_cache[cp.key],replacement)

    def test_namespace_mismatch_cannot_restore(self):
        cp,_,_,_=self.f.completed(1)
        self.f.manager.namespace="another-owner"
        job=self.f.start(1)
        self.assertEqual(job.cached_pages,0)
        self.f.finish(job)

    def test_serialize_jobs_before_state_change(self):
        job=self.f.start(1)
        with self.assertRaisesRegex(ValueError,"serialized"):
            self.f.manager.begin(Job(self.f.g,2))
        self.f.finish(job)

    def test_retention_without_cpu_tier(self):
        self.f.close()
        self.f=Fixture(target_cpu=False)
        a,_,_,_=self.f.completed(1)
        self.f.completed(2)
        job=self.f.start(1)
        self.assertEqual(job.cached_pages,8)
        self.assertIsNone(self.f.manager.stats()["target_cpu_tier"])
        self.f.finish(job)

    def test_zero_pinned_allocation_without_pressure(self):
        tier=self.f.g.cpu_page_cache
        self.assertIsNone(tier._alloc_thread)
        self.assertEqual(tier.pinned_bytes,0)
        self.f.completed(1)
        self.f.completed(2)
        self.assertEqual(tier.pinned_bytes,0)
        self.assertEqual(tier.num_slots,0)

    def test_lazy_pin_growth_is_exact_and_bounded(self):
        self.f.completed(1,4096)
        self.f.completed(2,4096)
        tier=self.f.g.cpu_page_cache
        self.assertGreater(tier.pinned_bytes,0)
        self.assertEqual(tier.pinned_bytes,tier.num_slots*tier.slab_size)
        self.assertLessEqual(tier.pinned_bytes,tier.max_slots*tier.slot_size)
        self.assertEqual(self.f.manager.stats()['target_cpu_tier']['reservation_policy'],
                         'pressure_demand_pinned_slabs')

    def test_exhausted_host_tier_falls_back_cold(self):
        self.f.close()
        self.f=Fixture(tier_budget=2*8192)
        cp,_,_,_=self.f.completed(1,4096)
        self.f.completed(2,4096)
        self.f.completed(3,4096)
        tier=self.f.g.cpu_page_cache
        self.assertEqual(tier.pinned_bytes,2*8192)
        self.assertGreater(tier.metrics['evictions'],0)
        self.assertFalse(self.f.g.pagetable.is_resumable(cp.key))
        job=self.f.start(1,4096)
        self.assertEqual(job.cached_pages,0)
        self.f.finish(job)

    def test_native_pair_reuse_keeps_original_raw_window(self):
        cp,_,_,_=self.f.completed(1)
        job=self.f.start(1)
        self.f.ring.fill_(-1000)
        self.f.manager.capture(job)
        self.assertIs(self.f.manager.prompt_checkpoint,cp)
        self.f.finish(job)

    def test_partial_snapshot_failure_cannot_publish_pair(self):
        original=torch.empty
        job=self.f.start(2)
        def fail(*args,**kwargs):
            if kwargs.get('device')=='cpu': raise MemoryError('injected snapshot failure')
            return original(*args,**kwargs)
        with patch.object(torch,'empty',fail),self.assertRaises(MemoryError):
            self.f.checkpoint(job)
        self.f.manager.invalidate('generation_error')
        job.deallocate_pages()
        self.assertFalse(self.f.manager.registry)

    def test_source_epoch_change_refuses_during_active_request(self):
        job=self.f.start(1)
        with self.assertRaisesRegex(ValueError,'idle'):
            self.f.manager.invalidate_all('changed_layout')
        self.f.finish(job)

    def test_lazy_constructor_source_drift_refuses_before_allocation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path=Path(tmp)/'cpu_cache.py';path.write_text('changed')
            with patch.object(cache_stub,'__file__',str(path)),self.assertRaisesRegex(ValueError,'pinned'):
                lazy.LazyTargetCPUPageCache([self.f.g.cache],100000)

    def test_target_only_attachment_never_passes_draft_cache(self):
        self.f.close()
        self.f=Fixture(target_cpu=False)
        seen=[]
        def factory(caches,budget):
            seen.extend(caches)
            return lazy.LazyTargetCPUPageCache(caches,budget)
        sessions.attach_target_cpu_tier(self.f.g,100000,factory)
        self.assertEqual(seen,[self.f.g.cache])
        self.assertNotIn(self.f.g.draft_cache,seen)

    def test_double_attachment_refuses_without_overwriting_owner(self):
        owner=self.f.g.cpu_page_cache
        with self.assertRaisesRegex(ValueError,'existing tier'):
            sessions.attach_target_cpu_tier(self.f.g,100000,lazy.LazyTargetCPUPageCache)
        self.assertIs(self.f.g.cpu_page_cache,owner)

    def test_host_pin_failure_degrades_to_cold_without_reference_leak(self):
        cp,_,_,_=self.f.completed(1,4096)
        original=torch.empty
        def fail_pin(*args,**kwargs):
            if kwargs.get('pin_memory'): raise MemoryError('injected host capacity failure')
            return original(*args,**kwargs)
        with patch.object(torch,'empty',fail_pin):
            self.f.completed(2,4096)
            # B's unwritten output claim can revert to A on deallocation.
            # C forces a written replacement of A's original target chain.
            self.f.completed(3,4096)
        tier=self.f.g.cpu_page_cache
        self.assertEqual(tier.metrics['pin_failures'],1)
        self.assertGreater(tier.metrics['skipped_spills'],0)
        self.assertEqual(tier.pinned_bytes,0)
        self.assertEqual(len(self.f.g.pagetable.referenced_pages),0)
        self.assertFalse(self.f.g.pagetable.is_resumable(cp.key))
        job=self.f.start(1,4096)
        self.assertEqual(job.cached_pages,0)
        self.f.finish(job)

    def test_transfer_fault_is_not_swallowed_as_host_allocation_failure(self):
        self.f.completed(1,4096)
        with patch.object(lazy.LazyTargetCPUPageCache,'_make_slab',side_effect=RuntimeError('illegal memory access')):
            with self.assertRaisesRegex(RuntimeError,'illegal memory'):
                self.f.start(2,4096)

    def test_source_drift_and_tp_constructor_gates(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):
            lazy.LazyTargetCPUPageCache([self.f.g.cache,self.f.g.draft_cache],100000)
        self.f.g.model.loaded_tp=True
        with self.assertRaisesRegex(ValueError,'exactly one'):
            lazy.LazyTargetCPUPageCache([self.f.g.cache],100000)
        self.f.g.model.loaded_tp=False

    def test_pre_stash_replaces_unpaired_existing_native_state(self):
        job=self.f.start(1)
        seq=job.sequences[0]
        # Complete/hash target pages before the native checkpoint exists.
        for p,page in enumerate(seq.allocated_pages[:8]):
            page.kv_position=256
            page.prev_hash=seq.allocated_pages[p-1].phash if p else None
            page.sequence.copy_(seq.sequence_ids.torch_slice(p*256,(p+1)*256))
            page.update_hash(seq.page_hashes[p])
        seq.kv_position=2048;job.recurrent_state.position=2048
        self.f.g.cache.layer.tensor.fill_(-99)
        job.maybe_stash_recurrent(self.f.g.recurrent_cache,256)
        key=seq.allocated_pages[7].phash
        self.assertNotIn(sessions.COOKIE,self.f.g.recurrent_cache[key])
        job.last_recurrent_checkpoint_pos=None
        self.f.g.cache.layer.tensor.fill_(123)
        self.f.manager.before_stash(job,256)
        job.maybe_stash_recurrent(self.f.g.recurrent_cache,256)
        self.f.manager.capture(job)
        self.assertTrue(torch.equal(self.f.g.recurrent_cache[key][0],torch.full((4,),123.)))
        self.assertIn(sessions.COOKIE,self.f.g.recurrent_cache[key])
        self.f.finish(job)


class Configuration(unittest.TestCase):
    def test_runtime_native_source_pins_match_deployed_fixture(self):
        paths = {name: ENGINE / (name.removeprefix('exllamav3.').replace('.', '/') + '.py')
                 for name in sessions.ENGINE_SOURCE_PINS}
        modules = {name: NS(__file__=str(path)) for name, path in paths.items()}
        with patch.object(sessions.importlib, 'import_module', side_effect=modules.__getitem__):
            sessions.validate_engine_sources()

    def test_runtime_native_source_drift_refuses_before_manager_creation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path = Path(tmp) / 'changed.py'; path.write_text('changed')
            with patch.object(sessions.importlib, 'import_module', return_value=NS(__file__=str(path))):
                with self.assertRaisesRegex(ValueError, 'pinned installed source'):
                    sessions.enable_session_cache(NS(num_remaining_jobs=lambda:0))

    def test_native_source_path_missing_refuses(self):
        with patch.object(sessions.importlib, 'import_module', return_value=NS()):
            with self.assertRaisesRegex(ValueError, 'pinned installed source'):
                sessions.validate_engine_sources()

    def args(self,**changes):
        a=NS(dflash_session_cache=False,target_cpu_cache_gib=0,recurrent_cache_gib=4,
             session_cache_gib=1,session_cache_max_checkpoints=8,dflash_prefix_cache=False,
             mtp=False,draft_model_dir="drafter",tensor_parallel=False)
        a.__dict__.update(changes);return a

    def test_default_is_gpu_only_and_native_budget_unchanged(self):
        o=session_options(self.args())
        self.assertFalse(o.enabled)
        self.assertEqual(o.target_cpu_bytes,0)
        self.assertEqual(o.recurrent_bytes,4*1024**3)

    def test_explicit_sixteen_gib_target_tier(self):
        o=session_options(self.args(dflash_session_cache=True,target_cpu_cache_gib=16))
        self.assertEqual(o.target_cpu_bytes,16*1024**3)

    def test_previous_request_rollback_mode_is_preserved(self):
        self.assertFalse(session_options(self.args(dflash_prefix_cache=True)).enabled)

    def test_conflicting_modes_refuse(self):
        with self.assertRaisesRegex(ValueError,"Choose"):
            session_options(self.args(dflash_prefix_cache=True,dflash_session_cache=True))

    def test_budget_validation(self):
        for key,value in (("target_cpu_cache_gib",-1),("target_cpu_cache_gib",65),
                          ("recurrent_cache_gib",0),("session_cache_gib",True),
                          ("session_cache_max_checkpoints",1),("session_cache_max_checkpoints",129)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                session_options(self.args(**{key:value}))

    def test_cpu_tier_requires_paired_session_mode(self):
        with self.assertRaisesRegex(ValueError,"requires --dflash-session"):
            session_options(self.args(target_cpu_cache_gib=16))

    def test_tp_or_mtp_refuses(self):
        for key in ("tensor_parallel","mtp"):
            with self.subTest(key=key),self.assertRaises(ValueError):
                session_options(self.args(dflash_session_cache=True,**{key:True}))

    def test_api_uses_underlying_async_job_for_stats(self):
        tree=ast.parse((REPO/'glm/glm_api.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
               isinstance(n.func,ast.Name) and n.func.id=='summarize_draft_stats']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].args[0]),'job.job.draft_stats')


if __name__=="__main__": unittest.main()
