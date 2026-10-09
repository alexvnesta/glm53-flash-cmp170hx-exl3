"""Real pinned native page/session/tier methods, CPU-only layout transactions."""
import asyncio
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import test_sessions as base
import test_duplicate_recycle as recycle
from glm_memory_coordinator import CombinedMemoryCoordinator
from glm_pin_budget import pin_request_bytes, allocate_pinned_view
from glm_worker_cleanup import drain_owned_future


class CombinedFixture(recycle.Fixture):
    def __init__(self):
        super().__init__()
        self.g._glm_dflash_prefix_cache = self.manager
        self.g.model.forward = lambda **kwargs: kwargs
        self.g.clear_queue = lambda: None
        self.layer = NS(qk=self.g.cache.tensors[0], sk=self.g.cache.tensors[2])
        self.nonlatent = (self.g.cache.tensors[1], self.g.cache.tensors[3])
        self.g.cache.get_all_tensors = lambda: (
            self.layer.qk, self.nonlatent[0], self.layer.sk, self.nonlatent[1])
        self.tier_identity=patch.dict(__import__('sys').modules,{
            'glm_target_cpu_cache':recycle.tier_module})
        self.tier_identity.start()
        self.coordinator = CombinedMemoryCoordinator(self.g, self.manager, self.g.cpu_page_cache)
        self.adapter = NS(generator=self.g, layers={0:self.layer}, states={},
                          host_bytes=123, host_pin_request_bytes=256)
        self.adapter.deactivate = self.restore
        self.coordinator.bind_adapter(self.adapter)
        self.cuda = [patch.object(torch.cuda, 'device', return_value=NS(
            __enter__=lambda s:None, __exit__=lambda *a:None))]
        # Special methods must be on a class, rather than SimpleNamespace.
        class DeviceContext:
            def __enter__(self): return None
            def __exit__(self,*args): return False
        self.cuda = [patch.object(torch.cuda, 'device', return_value=DeviceContext()),
                     patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False),
                     patch.object(torch.cuda, 'current_stream', return_value=NS(synchronize=lambda:None))]
        for p in self.cuda:p.start()

    def migrate(self):
        c = self.coordinator
        q,s = self.layer.qk,self.layer.sk
        qalias = recycle.FakeCuda(q.tensor.clone()); qalias.device_index=0
        salias = recycle.FakeCuda(s.tensor.clone()); salias.device_index=0
        self.adapter.states={0:NS(layer=self.layer, stager=NS(
            q_alias=qalias,s_alias=salias,device=torch.device('cuda',0)))}
        c.prepare_host_commit()
        self.layer.qk,self.layer.sk=qalias,salias
        self.g.cache.tensors=self.g.cache.get_all_tensors()
        c.host_committed()

    def replacements(self):
        state=self.adapter.states[0]
        q=recycle.FakeCuda(state.stager.q_alias.tensor.clone());q.device_index=0
        s=recycle.FakeCuda(state.stager.s_alias.tensor.clone());s.device_index=0
        return {0:(q,s)}

    def restore(self):
        replacements=self.replacements()
        self.coordinator.prepare_gpu_commit(replacements)
        self.layer.qk,self.layer.sk=replacements[0]
        self.g.cache.tensors=self.g.cache.get_all_tensors()
        self.coordinator.gpu_committed()

    def close(self):
        if getattr(self,'closed',False):return
        try:
            if hasattr(self,'coordinator') and self.coordinator.phase not in ('CLOSED','FAILED'):
                self.coordinator.close()
        finally:
            for p in reversed(getattr(self,'cuda',())):p.stop()
            if hasattr(self,'tier_identity'):self.tier_identity.stop()
            super().close()


class CombinedContracts(unittest.TestCase):
    def setUp(self):self.f=CombinedFixture()
    def tearDown(self):self.f.close()

    def test_three_layout_epochs_preserve_page_identity_and_indexer(self):
        c=self.f.coordinator;tier=self.f.g.cpu_page_cache
        pages=c.page_ids;indexer=tuple(t.data_ptr() for t in self.f.nonlatent)
        old=self.f.layer.qk
        for epoch in (1,2,3):
            self.f.migrate()
            self.assertEqual(tier.segments,[])
            self.assertTrue(tier._layout_suspended)
            self.assertEqual(c.phase,'HOST_ACTIVE')
            c.ensure_gpu('new_prefill')
            self.assertEqual(c.phase,'GPU')
            self.assertEqual(c.epoch,epoch)
            self.assertEqual(c.page_ids,pages)
            self.assertEqual(tuple(t.data_ptr() for t in self.f.nonlatent),indexer)
            self.assertIs(tier.segments[0][0],self.f.layer.qk)
        self.assertIsNot(self.f.layer.qk,old)

    def test_native_store_fetch_uses_restored_planes_and_keeps_checkpoint(self):
        job=self.f.start(1);self.f.checkpoint(job)
        cp=self.f.manager.prompt_checkpoint
        page=job.sequences[0].allocated_pages[0]
        tier=self.f.g.cpu_page_cache
        tier.store(page,10)
        image=tier.slot_slabs[0].clone()
        self.f.migrate();self.f.coordinator.ensure_gpu('resume')
        tier.fetch(page.phash,31,20)
        for tensor,_,_,_ in tier.segments:
            self.assertTrue(torch.equal(tensor[31],tensor[page.page_index]))
        self.assertTrue(torch.equal(image,tier.slot_slabs[0]))
        self.assertTrue(self.f.manager._valid(cp))
        self.f.finish(job)

    def test_raw_copy_refused_until_gpu_return(self):
        self.f.migrate()
        with self.assertRaisesRegex(ValueError,'host latent'):
            self.f.g.cpu_page_cache._check_transfer_owner()
        self.f.coordinator.ensure_gpu('cancel')
        self.f.g.cpu_page_cache._check_transfer_owner()

    def test_return_refused_inside_forward_without_mutation(self):
        self.f.migrate();c=self.f.coordinator;c.forward_depth=1
        with self.assertRaisesRegex(RuntimeError,'during target forward'):c.ensure_gpu('copy')
        self.assertEqual(c.phase,'HOST_ACTIVE')
        c.forward_depth=0

    def test_indexer_pointer_drift_refuses_return(self):
        self.f.migrate();old=self.f.nonlatent
        wrong=recycle.FakeCuda(old[0].tensor.clone());wrong.device_index=1
        self.f.nonlatent=(wrong,old[1])
        with self.assertRaisesRegex(RuntimeError,'non-latent'):self.f.restore()
        self.assertEqual(self.f.coordinator.phase,'HOST_ACTIVE')
        self.f.nonlatent=old

    def test_partial_or_wrong_device_return_refused(self):
        self.f.migrate();c=self.f.coordinator
        with self.assertRaisesRegex(RuntimeError,'replacements'):c.prepare_gpu_commit({})
        values=self.f.replacements();values[0][0].device_index=1
        with self.assertRaisesRegex(RuntimeError,'geometry/device'):c.prepare_gpu_commit(values)
        self.assertEqual(c.phase,'HOST_ACTIVE')

    def test_failed_return_marks_fail_stop(self):
        self.f.migrate()
        self.f.adapter.deactivate=lambda:(_ for _ in ()).throw(MemoryError('restore'))
        with self.assertRaises(MemoryError):self.f.coordinator.ensure_gpu('cancel')
        self.assertEqual(self.f.coordinator.phase,'FAILED')
        with self.assertRaisesRegex(RuntimeError,'restart'):self.f.coordinator.ensure_gpu('prefill')
        # Test fixture repair only; production cannot revive FAILED.
        self.f.coordinator.phase='HOST_ACTIVE';self.f.adapter.deactivate=self.f.restore

    def test_migration_abort_rebinds_original_gpu_fields(self):
        c=self.f.coordinator;self.f.adapter.states={}
        old=self.f.layer.qk;c.prepare_host_commit();c.abort_host_commit()
        self.assertEqual(c.phase,'GPU')
        self.assertIs(self.f.g.cpu_page_cache.segments[0][0],old)

    def test_wrong_worker_refused_before_return(self):
        self.f.migrate();errors=[]
        def run():
            try:self.f.coordinator.ensure_gpu('other_thread')
            except RuntimeError as e:errors.append(str(e))
        t=threading.Thread(target=run);t.start();t.join()
        self.assertEqual(len(errors),1);self.assertIn('persistent worker',errors[0])
        self.assertEqual(self.f.coordinator.phase,'HOST_ACTIVE')

    def test_wrong_stream_refused_before_rebind_publication(self):
        self.f.migrate();replacement=self.f.replacements();c=self.f.coordinator
        c.prepare_gpu_commit(replacement)
        self.f.layer.qk,self.f.layer.sk=replacement[0]
        self.f.g.cache.tensors=self.f.g.cache.get_all_tensors()
        with patch.object(recycle.tier_module,'current_transfer_streams',return_value=(('cuda:0',99),('cuda:1',22))):
            with self.assertRaisesRegex(ValueError,'stream changed'):c.gpu_committed()
        self.assertEqual(self.f.g.cpu_page_cache.segments,[])
        self.assertTrue(self.f.g.cpu_page_cache._layout_suspended)
        c.gpu_committed()

    def test_page_owner_drift_refused_before_migration(self):
        old=self.f.g.pagetable.all_pages;self.f.g.pagetable.all_pages=list(old)
        with self.assertRaisesRegex(RuntimeError,'ownership'):self.f.migrate()
        self.assertEqual(self.f.coordinator.phase,'GPU')
        self.f.g.pagetable.all_pages=old

    def test_lifecycle_checkpoint_noop_keeps_host_then_boundary_restores(self):
        job=self.f.start(1);self.f.checkpoint(job)
        job.last_recurrent_checkpoint_pos=None
        job.sequences[0].prefill_complete=True
        job.sequences[0].kv_position=2049
        native_job=ModuleType('exllamav3.generator.job');native_job.Job=base.Job
        with patch.object(base.Job,'prefill',lambda job:None,create=True),patch.dict(
                __import__('sys').modules,{native_job.__name__:native_job}):
            self.f.coordinator.install_lifecycle();self.f.migrate()
            job.maybe_stash_recurrent(self.f.g.recurrent_cache,2048)
            self.assertEqual(self.f.coordinator.phase,'HOST_ACTIVE')
            job.sequences[0].kv_position=2048;job.recurrent_state.position=2048
            job.maybe_stash_recurrent(self.f.g.recurrent_cache,2048)
            self.assertEqual(self.f.coordinator.phase,'GPU')
            self.f.coordinator.close()
        self.f.finish(job)

    def test_lifecycle_deallocate_and_clear_restore_first(self):
        job=self.f.start(1)
        native_job=ModuleType('exllamav3.generator.job');native_job.Job=base.Job
        with patch.object(base.Job,'prefill',lambda job:None,create=True),patch.dict(
                __import__('sys').modules,{native_job.__name__:native_job}):
            self.f.coordinator.install_lifecycle();self.f.migrate()
            job.deallocate_pages();self.assertEqual(self.f.coordinator.phase,'GPU')
            self.f.migrate();self.f.g.clear_queue();self.assertEqual(self.f.coordinator.phase,'GPU')
            self.f.coordinator.close()

    def test_partial_hook_install_rolls_back(self):
        native_job=ModuleType('exllamav3.generator.job');native_job.Job=base.Job
        before=base.Job.allocate_pages
        with patch.dict(__import__('sys').modules,{native_job.__name__:native_job}):
            # Missing prefill triggers failure after two Job wrappers installed.
            with self.assertRaises(AttributeError):self.f.coordinator.install_lifecycle()
        self.assertIs(base.Job.allocate_pages,before)
        self.assertEqual(self.f.coordinator.hooks,[])


class PinBudgets(unittest.TestCase):
    def test_full_384k_slot_request_ceiling(self):
        payload=3153920;request=pin_request_bytes(payload)
        self.assertEqual(request,4194304)
        self.assertEqual(2**30//request,256)
        self.assertEqual(256*payload,807403520)
        self.assertEqual(11*(pin_request_bytes(192*2**20)+pin_request_bytes(12*2**20)),2992*2**20)

    def test_explicit_extent_is_kept_by_view(self):
        calls=[]
        def empty(*args,**kwargs):
            calls.append(kwargs.pop('pin_memory'))
            return torch.empty(*args,**kwargs)
        fake=NS(empty=empty)
        tensor,request=allocate_pinned_view(fake,(3,7),dtype=torch.int32,element_size=4)
        self.assertEqual(tuple(tensor.shape),(3,7));self.assertEqual(request,128)
        self.assertEqual(tensor.untyped_storage().nbytes(),128)
        self.assertEqual(calls,[True])

    def test_invalid_size_refused(self):
        for value in (0,-1,1.5,True):
            with self.assertRaises(ValueError):pin_request_bytes(value)


class AsyncDrain(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_waits_for_owned_cleanup(self):
        loop=asyncio.get_running_loop();future=loop.create_future();seen=[]
        async def consume():
            try:await drain_owned_future(future)
            except asyncio.CancelledError:seen.append('cancelled_after_done')
        task=asyncio.create_task(consume());await asyncio.sleep(0)
        task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done());self.assertFalse(future.cancelled())
        future.set_result(None);await task
        self.assertEqual(seen,['cancelled_after_done'])

    async def test_worker_error_surfaces_without_cancelling_owner(self):
        future=asyncio.get_running_loop().create_future();future.set_exception(ValueError('worker'))
        with self.assertRaisesRegex(ValueError,'worker'):await drain_owned_future(future)
