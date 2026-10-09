"""Actual pinned native Generator constructor, opt-in TP startup and rollback."""
from pathlib import Path
from types import ModuleType,SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import test_tp_sessions as tp
import test_session_startup as startup

class Fixture(tp.Fixture):
    def __init__(self):
        super().__init__(False)
        self.manager.close()
        old=self.g
        old.model.config=NS(vocab_size=128)
        old.model.caps={'recurrent_states':True}
        old.cache.num_slots=1;old.cache.reset_states=lambda:None
        old.draft_cache.max_num_tokens=old.cache.max_num_tokens
        old.draft_model.caps={'dflash_draft':True,'default_draft_size':7}
        self.Generator=startup.native_generator_type()
        self.g=self.Generator(old.model,old.cache,NS(),draft_model=old.draft_model,
                              draft_cache=old.draft_cache,max_batch_size=1,max_chunk_size=2048)
        self.Job=type('TPStartupJob',(tp.base.Job,),{'prepare_for_queue':lambda self,g,*a,**kw:None})
        modules={}
        for name in tp.prefix.ENGINE_SOURCE_PINS:
            mod=ModuleType(name)
            mod.__file__=str(tp.base.ENGINE/(name.removeprefix('exllamav3.').replace('.','/')+'.py'))
            modules[name]=mod
        modules['exllamav3.generator.generator'].Generator=self.Generator
        modules['exllamav3.generator.job'].Job=self.Job
        modules['exllamav3.cache.recurrent'].mp_cache_recurrent_del=tp.base.native['mp_cache_recurrent_del']
        self.startup_patch=patch.dict(sys.modules,modules);self.startup_patch.start()
        self.manager=None
    def enable(self,bytes=64*8192,factory=None):
        self.manager=tp.sessions.enable_session_cache(self.g,max_bytes=1024**2,
                                                     target_cpu_bytes=bytes,tier_factory=factory or tp.tier_module.LazyTPTargetCPUPageCache)
        return self.manager
    def close(self):
        try:
            if self.manager is not None:self.manager.close()
            if self.g.cpu_page_cache is not None:self.g.cpu_page_cache.close()
            self.g.filter_pool.shutdown()
            tp.base.native['host_pool'].release()
        finally:
            self.startup_patch.stop();self.modules_patch.stop();self.empty_patch.stop()
            self.stream_patch.stop();self.inference_context.__exit__(None,None,None)

class TPStartup(unittest.TestCase):
    def setUp(self):self.f=Fixture()
    def tearDown(self):self.f.close()
    def test_native_true_default_disabled_before_rank_attach(self):
        self.assertIs(self.f.g.enable_defrag,True)
        observed=[]
        def factory(caches,budget):
            observed.append(self.f.g.enable_defrag);return tp.tier_module.LazyTPTargetCPUPageCache(caches,budget)
        manager=self.f.enable(factory=factory)
        self.assertEqual(observed,[False])
        self.assertIs(self.f.g._glm_dflash_prefix_cache,manager)
        self.assertEqual(self.f.g.cpu_page_cache.stats()['pinned_bytes'],0)
        self.assertFalse(any(ctx['glm_tp_lazy_cache'].arenas for ctx in self.f.g.model.contexts))
    def test_zero_cpu_budget_still_retains_fixed_page_sessions(self):
        self.f.enable(bytes=0)
        self.assertIs(self.f.g.enable_defrag,False);self.assertIsNone(self.f.g.cpu_page_cache)
    def test_failed_tier_factory_restores_flag_and_manager(self):
        with self.assertRaisesRegex(RuntimeError,'factory'):
            self.f.enable(factory=lambda *args:(_ for _ in ()).throw(RuntimeError('factory')))
        self.assertIs(self.f.g.enable_defrag,True)
        self.assertFalse(hasattr(self.f.g,'_glm_dflash_prefix_cache'))
    def test_partial_hook_failure_closes_rank_pools_and_restores_owned_state(self):
        with patch.object(tp.prefix,'enable_prefix_cache',side_effect=RuntimeError('hook failed')):
            with self.assertRaisesRegex(RuntimeError,'hook failed'):self.f.enable()
        self.assertIs(self.f.g.enable_defrag,True)
        self.assertIsNone(self.f.g.cpu_page_cache);self.assertIsNone(self.f.g.pagetable.cpu_tier)
        self.assertTrue(all(ctx['glm_tp_lazy_cache'] is None for ctx in self.f.g.model.contexts))
    def test_active_pending_and_preexisting_tier_refuse_before_mutation(self):
        for field in ('active_jobs','pending_jobs'):
            getattr(self.f.g,field).append(object())
            with self.assertRaisesRegex(ValueError,'fresh idle'):self.f.enable()
            getattr(self.f.g,field).clear()
            self.assertIs(self.f.g.enable_defrag,True)
        self.f.g.cpu_page_cache=object()
        with self.assertRaisesRegex(ValueError,'fresh idle'):self.f.enable()
        self.f.g.cpu_page_cache=None
    def test_source_drift_refuses_before_rank_pool_construction(self):
        with patch.object(tp.sessions,'validate_engine_sources',side_effect=ValueError('source drift')):
            with self.assertRaisesRegex(ValueError,'source drift'):self.f.enable()
        self.assertIs(self.f.g.enable_defrag,True)
        self.assertTrue(all('glm_tp_lazy_cache' not in ctx for ctx in self.f.g.model.contexts))
    def test_native_cache_takeover_refuses_old_pair_owner(self):
        manager=self.f.enable(bytes=0)
        self.f.g.cache.owner_serial+=1
        with self.assertRaisesRegex(ValueError,'ownership'):manager._tp_fence()
