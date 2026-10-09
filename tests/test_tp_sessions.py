"""CPU contracts execute native checkpoint/page methods with two rank doubles.

Real CPU tensors represent rank planes. No CUDA/device/native extension import.
Actual DMA and full-model continuation require a guarded live qualification.
"""
from collections import defaultdict
import importlib.util
from pathlib import Path
import sys
import mmap
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
import test_sessions as base

ROOT = Path(__file__).resolve().parents[1]
import glm_tp_sessions as sessions
import glm_tp_prefix as prefix

base.extract(base.ENGINE / 'cache/recurrent.py',
             {'new_checkpoint_handle','mp_cache_recurrent_clear','mp_cache_recurrent_stash',
              'mp_cache_recurrent_unstash','mp_cache_recurrent_del','_stashed_bytes'},base.native)
base.native['_next_checkpoint_handle']=0

class FakeCuda(base.FakeCuda):
    def __init__(self,tensor,rank): super().__init__(tensor); self.rank=rank
    @property
    def device(self): return NS(type='cuda', index=self.rank)
    @property
    def ndim(self): return self.tensor.ndim
    def data_ptr(self): return self.tensor.data_ptr()
    def is_contiguous(self): return self.tensor.is_contiguous()

class Arena:
    allocations=[]
    fail=False
    def __init__(self,size,*args,**kwargs):
        if self.fail: raise RuntimeError('pin refused')
        self.size=size; self.tensor=torch.empty(size,dtype=torch.uint8); self.closed=False
        self.allocations.append(self)
    def close(self): self.tensor=None; self.closed=True

class Stream:
    def __init__(self,rank): self.cuda_stream=rank+100; self.synced=0
    def synchronize(self): self.synced+=1

pool_module=ModuleType('exllamav3.util.tp_lazy_cache')
pool_module.__file__=str(base.ENGINE/'util/tp_lazy_cache.py')
pool_module.__dict__.update(threading=base.threading,mmap=mmap,torch=torch,PinnedArena=Arena)
base.extract(base.ENGINE/'util/tp_lazy_cache.py',
             {'_signature','LazyRankSlotPool','mp_tp_lazy_init','_pool','mp_tp_lazy_prepare',
              'mp_tp_lazy_drop','mp_tp_lazy_store','mp_tp_lazy_fetch','mp_tp_lazy_close'},pool_module.__dict__)
pool_module._tensors=lambda ctx,cache_id:tuple(ctx['tensors'])
with patch.dict(sys.modules, {'exllamav3.generator.cpu_cache':base.cache_stub,
                             'exllamav3.util.tp_lazy_cache':pool_module}):
    import glm_tp_cpu_cache as tier_module

class RankLayer(base.Layer):
    def stash(self, slot, position=0): return super().stash(slot)
    def unstash(self, slot, saved, position=0): return super().unstash(slot,saved)

class Model:
    loaded_tp=True
    active_devices=(0,1)
    def __init__(self,cache):
        self.contexts=[]; self.responses={}; self.commands=[]; self.drain_count=0
        for rank in self.active_devices:
            rank_cache=base.Cache(NS(loaded_tp=False))
            rank_cache.tensors=tuple(FakeCuda(t.tensor,rank) for t in rank_cache.tensors)
            layer=RankLayer()
            rank_cache.layer=layer
            module=NS(tp_recurrent_lookup={id(cache):layer},
                      tp_cache_lookup={id(cache):NS(get_tensors=lambda c=rank_cache:c.tensors)})
            self.contexts.append(dict(recurrent_modules=[module], recurrent_cache={},
                                      kv_modules=[module], tensors=rank_cache.tensors,layer=layer))
    def tp_drain_acks(self): self.drain_count+=1
    def tp_dispatch_all(self,fn,args):
        self.tp_drain_acks()
        for rank in self.active_devices: fn(self.contexts[rank],*args);self.commands.append((rank,fn.__name__,args))
    def tp_worker_dispatch(self,rank,fn,args):
        self.commands.append((rank,fn.__name__,args))
        try: self.responses[rank]=fn(self.contexts[rank],*args)
        except BaseException as error: self.responses[rank]=error
    def tp_worker_result(self,rank):
        value=self.responses.pop(rank)
        if isinstance(value,BaseException): raise value
        return value

class Fixture(base.Fixture):
    def __init__(self,target_cpu=False,tier_slots=64):
        super().__init__(target_cpu=False)
        self.manager.close()
        model=Model(self.g.cache)
        self.g.model=self.g.cache.model=self.g.recurrent_cache.model=model
        self.g.cache.recurrent_state_cls=base.native['GDNState']
        self.g.draft_cache.model=NS(loaded_tp=False)
        self.g.draft_model=NS(config=NS(experimental_tensor_parallel=True))
        self.streams={0:Stream(0),1:Stream(1)}
        self.stream_patch=patch.object(torch.cuda,'current_stream',side_effect=lambda device:self.streams[device.index])
        self.stream_patch.start()
        self.manager=sessions.MultiSessionCache(self.g,1024**2,8,'tpfixture')
        rc=sys.modules['exllamav3.cache.recurrent']
        rc.mp_cache_recurrent_del=base.native['mp_cache_recurrent_del']
        if target_cpu:
            with patch.dict(sys.modules,{'exllamav3.util.tp_lazy_cache':pool_module}):
                sessions.attach_target_cpu_tier(self.g,tier_slots*16384,tier_module.LazyTPTargetCPUPageCache)
    def checkpoint(self,job):
        value=int(job.sequences[0].sequence_ids.torch()[0,0])+job.expected_position
        for rank,ctx in enumerate(self.g.model.contexts): ctx['layer'].tensor.fill_(value+rank)
        super().checkpoint(job)
        # Fixture checkpoints also materialize each rank's paged byte planes.
        seq=job.sequences[0]
        for p,page in enumerate(seq.allocated_pages[:job.expected_position//256]):
            value=int(seq.sequence_ids.torch()[0,p*256])%500
            for rank,ctx in enumerate(self.g.model.contexts):
                for plane,tensor in enumerate(ctx['tensors']): tensor[page.page_index].fill_(value+plane*10+rank)
    def close(self):
        try: super().close()
        finally: self.stream_patch.stop()

class TPSessions(unittest.TestCase):
    def setUp(self): self.f=Fixture()
    def tearDown(self): self.f.close()
    def test_two_sessions_restore_both_rank_states_and_draft(self):
        cp,_,_,draft=self.f.completed(1)
        expected=[ctx['layer'].tensor.clone() for ctx in self.f.g.model.contexts]
        self.f.completed(2)
        for ctx in self.f.g.model.contexts:ctx['layer'].tensor.fill_(-999)
        job=self.f.start(1)
        self.assertEqual(job.cached_pages,8)
        for ctx,saved in zip(self.f.g.model.contexts,expected):self.assertTrue(torch.equal(ctx['layer'].tensor,saved))
        for row,index in enumerate(cp.page_indices):self.assertTrue(torch.equal(self.f.ring[index],draft[0][row]))
        self.f.finish(job)
    def test_cold_miss_preserves_two_sessions(self):
        a,*_=self.f.completed(1);b,*_=self.f.completed(2)
        job=self.f.start(3)
        self.assertEqual(job.cached_pages,0)
        self.assertEqual({a.identity,b.identity},set(self.f.manager.registry))
        self.f.manager.invalidate('cancelled');job.deallocate_pages()
    def test_cancel_unpublished_releases_exact_handle_on_both_ranks(self):
        a,*_=self.f.completed(1)
        job=self.f.start(2);self.f.checkpoint(job)
        cp=self.f.manager.prompt_checkpoint
        handle=self.f.g.recurrent_cache[cp.key]['tp_handle']
        self.f.manager.invalidate('cancelled');job.deallocate_pages()
        self.assertIn(a.identity,self.f.manager.registry)
        for ctx in self.f.g.model.contexts:self.assertNotIn(handle,ctx['recurrent_cache'])
        deletes=[r for r,fn,args in self.f.g.model.commands if fn=='mp_cache_recurrent_del' and args[1]==handle]
        self.assertEqual(deletes,[0,1])
    def test_checkpoint_lru_releases_ranks(self):
        self.f.manager.max_checkpoints=2
        a,*_=self.f.completed(1);handle=self.f.g.recurrent_cache[a.key]['tp_handle']
        self.f.completed(2);self.f.completed(3)
        for ctx in self.f.g.model.contexts:self.assertNotIn(handle,ctx['recurrent_cache'])
    def test_alias_key_removal_preserves_shared_rank_image(self):
        cp,*_=self.f.completed(1)
        stash=self.f.g.recurrent_cache[cp.key];handle=stash['tp_handle']
        self.f.g.recurrent_cache[b'alias']=stash
        self.f.manager._release_native(cp)
        for ctx in self.f.g.model.contexts:self.assertIn(handle,ctx['recurrent_cache'])
        self.f.manager._discard_native_key(b'alias')
        for ctx in self.f.g.model.contexts:self.assertNotIn(handle,ctx['recurrent_cache'])
    def test_stale_cookie_does_not_delete_recreated_handle(self):
        cp,*_=self.f.completed(1)
        stash=self.f.g.recurrent_cache[cp.key];stash[sessions.COOKIE]='changed'
        self.f.manager._release_native(cp)
        self.assertIn(cp.key,self.f.g.recurrent_cache)
    def test_missing_or_boolean_handle_cannot_publish(self):
        job=self.f.start(1);self.f.checkpoint(job)
        cp=self.f.manager.prompt_checkpoint;self.f.manager.prompt_checkpoint=None
        stash=self.f.g.recurrent_cache[cp.key];handle=stash['tp_handle'];stash['tp_handle']=True
        self.f.manager.capture(job)
        self.assertIsNone(self.f.manager.prompt_checkpoint)
        stash['tp_handle']=handle;self.f.finish(job)
    def test_unsupported_features_force_zero_native_reuse(self):
        self.f.completed(1)
        job=self.f.start(1,unsupported=True)
        self.assertEqual(job.cached_pages,0);self.f.finish(job)
    def test_epoch_change_between_plan_and_allocate_retries_cold(self):
        cp,*_=self.f.completed(1)
        job=self.f.start(1,before_allocate=lambda j:self.f.g.recurrent_cache[cp.key].update({sessions.COOKIE:'foreign'}))
        self.assertEqual(job.cached_pages,0);self.f.finish(job)
    def test_tp_draft_refused(self):
        self.f.g.draft_cache.model.loaded_tp=True
        with self.assertRaisesRegex(ValueError,'custody'):sessions.MultiSessionCache(self.f.g)
        self.f.g.draft_cache.model.loaded_tp=False

class TPLazyTier(unittest.TestCase):
    def setUp(self):
        Arena.allocations.clear();Arena.fail=False
        self.f=Fixture(target_cpu=True)
    def tearDown(self):Arena.fail=False;self.f.close()
    def test_zero_registration_until_native_pressure(self):
        self.f.completed(1);self.f.completed(2)
        self.assertEqual(Arena.allocations,[])
        self.assertEqual(self.f.g.cpu_page_cache.stats()['pinned_bytes'],0)
    def test_real_page_pressure_restores_every_plane_and_both_rank_states(self):
        cp,*_=self.f.completed(1,4096)
        recurrent=[ctx['layer'].tensor.clone() for ctx in self.f.g.model.contexts]
        self.f.completed(2,4096);self.f.completed(3,4096)
        tier=self.f.g.cpu_page_cache
        self.assertGreater(tier.metrics['pushes'],0)
        self.assertTrue(self.f.g.pagetable.is_resumable(cp.key))
        job=self.f.start(1,4096)
        self.assertEqual(job.cached_pages,16)
        self.assertGreater(tier.metrics['restores'],0)
        for ctx,saved in zip(self.f.g.model.contexts,recurrent):self.assertTrue(torch.equal(ctx['layer'].tensor,saved))
        for p,page in enumerate(job.sequences[0].allocated_pages[:16]):
            value=int(job.sequences[0].sequence_ids.torch()[0,p*256])%500
            for rank,ctx in enumerate(self.f.g.model.contexts):
                for plane,tensor in enumerate(ctx['tensors']):
                    self.assertTrue(torch.equal(tensor[page.page_index],torch.full_like(tensor[page.page_index],value+plane*10+rank)))
        self.assertLessEqual(tier.stats()['registered_mapping_bytes'],tier.max_slots*tier.slot_size)
        self.f.finish(job)
    def test_pin_failure_degrades_to_cold_and_drains_both_ranks(self):
        self.f.completed(1,4096);Arena.fail=True;self.f.completed(2,4096)
        tier=self.f.g.cpu_page_cache
        self.assertEqual(tier.metrics['pin_failures'],1)
        self.assertFalse(self.f.g.model.responses)
        self.assertEqual(len(tier),0)
        job=self.f.start(1,4096)
        self.assertEqual(job.cached_pages,0);self.f.finish(job)
    def test_foreign_generator_identity_refuses_transfer(self):
        tier=self.f.g.cpu_page_cache;g=self.f.g.pagetable.generator
        self.f.g.pagetable.generator=NS(**vars(g))
        with self.assertRaisesRegex(ValueError,'ownership'):tier._check_owner()
        self.f.g.pagetable.generator=g
    def test_different_rank_stream_refuses_before_overwrite(self):
        self.f.completed(1,4096);self.f.completed(2,4096)
        self.f.streams[1]=Stream(99)
        with self.assertRaisesRegex(ValueError,'CUDA streams'):
            self.f.g.cpu_page_cache.fetch(next(iter(self.f.g.cpu_page_cache.entries)),0,0)
        self.assertFalse(self.f.g.model.responses)
    def test_complete_gpu_duplicate_recycles_even_if_protected(self):
        tier=self.f.g.cpu_page_cache
        job=self.f.start(1);self.f.checkpoint(job)
        page=job.sequences[0].allocated_pages[0]
        tier.entries[page.phash]=dict(slot=0,prev_hash=None,access_serial=0,tokens=page.sequence.clone())
        self.assertEqual(tier._evict_one({page.phash}),0)
        self.assertEqual(tier.metrics['duplicate_evictions'],1);self.f.finish(job)
    def test_all_protected_sole_host_copies_skip_spill(self):
        tier=self.f.g.cpu_page_cache
        for i in range(2):
            tier.entries[bytes([i+1])*16]=dict(slot=i,prev_hash=None,access_serial=i,tokens=torch.zeros(1,256,dtype=torch.long))
        self.assertIsNone(tier._evict_one(set(tier.entries)))
        self.assertEqual(len(tier.entries),2)
        self.assertEqual(tier.metrics['protected_spill_skips'],1)
    def test_inexact_gpu_duplicates_do_not_destroy_protected_host_image(self):
        tier=self.f.g.cpu_page_cache
        job=self.f.start(1);self.f.checkpoint(job)
        page=job.sequences[0].allocated_pages[0]
        entry=dict(slot=0,prev_hash=page.prev_hash,access_serial=0,tokens=page.sequence.clone())
        tier.entries[page.phash]=entry
        entry['tokens'][0,0]+=1
        self.assertIsNone(tier._evict_one({page.phash}))
        self.assertIn(page.phash,tier.entries)
        entry['tokens']=page.sequence.clone();page.can_revert=True
        self.assertIsNone(tier._evict_one({page.phash}))
        page.can_revert=False;self.f.finish(job)
    def test_physical_page_table_replacement_refuses(self):
        tier=self.f.g.cpu_page_cache;original=self.f.g.pagetable.all_pages
        self.f.g.pagetable.all_pages=list(original)
        with self.assertRaisesRegex(ValueError,'ownership'):tier._check_owner()
        self.f.g.pagetable.all_pages=original
    def test_one_rank_pin_failure_drops_successful_peer_preparation(self):
        tier=self.f.g.cpu_page_cache
        original=pool_module.LazyRankSlotPool.prepare
        def prepare(pool,slot):
            return False if pool is self.f.g.model.contexts[1]['glm_tp_lazy_cache'] else original(pool,slot)
        with patch.object(pool_module.LazyRankSlotPool,'prepare',prepare):
            self.assertIsNone(tier._new_slot(None))
        self.assertTrue(all(not ctx['glm_tp_lazy_cache'].arenas for ctx in self.f.g.model.contexts))
        self.assertFalse(self.f.g.model.responses)
    def test_tensor_pointer_drift_refuses_rank_transfer(self):
        self.f.completed(1,4096);self.f.completed(2,4096)
        ctx=self.f.g.model.contexts[1]
        original=ctx['tensors']
        ctx['tensors']=tuple(FakeCuda(t.tensor.clone(),1) for t in original)
        with self.assertRaisesRegex(ValueError,'ownership/layout'):
            self.f.g.cpu_page_cache.fetch(next(iter(self.f.g.cpu_page_cache.entries)),0,0)
        ctx['tensors']=original
        self.assertFalse(self.f.g.model.responses)
    def test_parent_transfer_thread_drift_refuses(self):
        tier=self.f.g.cpu_page_cache
        tier._check_owner();tier.thread=-1
        with self.assertRaisesRegex(ValueError,'engine thread'):tier._check_owner()
    def test_out_of_range_slot_preparation_refuses(self):
        pool=self.f.g.model.contexts[0]['glm_tp_lazy_cache']
        for slot in (-1,pool.max_slots,True):
            with self.assertRaisesRegex(ValueError,'slot'):pool.prepare(slot)
    def test_foreign_rank_owner_close_refuses_without_dropping_pool(self):
        context=self.f.g.model.contexts[0]
        with self.assertRaisesRegex(ValueError,'foreign'):
            pool_module.mp_tp_lazy_close(context,'foreign',id(self.f.g.cache))
        self.assertIsNotNone(context['glm_tp_lazy_cache'])
    def test_close_waits_rank_streams_and_drops_all_registered_slabs(self):
        self.f.completed(1,4096);self.f.completed(2,4096)
        tier=self.f.g.cpu_page_cache;tier.close()
        self.assertTrue(all(a.closed for a in Arena.allocations))
        self.assertTrue(all(s.synced for s in self.f.streams.values()))
        self.assertTrue(all(ctx['glm_tp_lazy_cache'] is None for ctx in self.f.g.model.contexts))

if __name__=='__main__':unittest.main()
