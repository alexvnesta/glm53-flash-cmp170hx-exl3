"""Execute the pinned native Generator constructor and the API's startup gate.

Cache/model tensors are CPU storage with fake CUDA labels. Only queue-hook
endpoints are doubles; native __init__, PageTable and RecurrentCache are actual
source bodies. No engine import, native extension, weights or CUDA initialization.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import torch
import test_sessions as base
import test_duplicate_recycle as policies

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import glm_dflash_prefix as prefix
import glm_dflash_sessions as sessions
lazy = base.lazy


def native_generator_type():
    tree = ast.parse((base.ENGINE / "generator/generator.py").read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Generator")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef)
               and n.name in ("__init__", "num_remaining_jobs")]
    cls = ast.ClassDef(name="CPUNativeGenerator", bases=[], keywords=[], body=methods, decorator_list=[])
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[])
    ns = dict(base.native, ThreadPoolExecutor=ThreadPoolExecutor)
    exec(compile(ast.fix_missing_locations(code), str(base.ENGINE / "generator/generator.py"), "exec"), ns)
    generator = ns["CPUNativeGenerator"]
    generator.iterate = lambda self: []
    generator.cancel = lambda self, job: None
    generator.clear_queue = lambda self: None
    return generator


class StartupFixture:
    def __init__(self, enabled=True):
        try:
            self._build(enabled)
        except BaseException:
            if hasattr(self, "g"): self.g.filter_pool.shutdown()
            if hasattr(self, "module_patch"): self.module_patch.stop()
            for p in reversed(getattr(self, "patches", ())): p.stop()
            if hasattr(self, "context"): self.context.__exit__(None, None, None)
            raise

    def _build(self, enabled):
        self.context = torch.inference_mode(); self.context.__enter__()
        self.original_empty = torch.empty
        def cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return self.original_empty(*args, **kwargs)
        self.patches = [patch.object(torch, "empty", cpu_empty),
                        patch.object(base, "FakeCuda", policies.FakeCuda),
                        patch.dict(os.environ, {"GLM53_CPU_DUPLICATE_RECYCLE": "1" if enabled else "0"})]
        for p in self.patches: p.start()
        policies.FakeCuda.counter = 0
        model = NS(loaded_tp=False, config=NS(vocab_size=128), caps={"recurrent_states":True})
        cache = base.Cache(model); cache.num_slots = 1; cache.reset_states = lambda: None
        ring = torch.zeros(17, 256, 1)
        draft_cache = NS(max_num_tokens=cache.max_num_tokens, dflash_ring_tokens=4352,
                         get_all_tensors=lambda:(ring,))
        draft_model = NS(caps={"dflash_draft":True, "default_draft_size":7})
        self.Generator = native_generator_type()
        # Deliberately omit enable_defrag: exercise the native True default.
        self.g = self.Generator(model, cache, NS(), max_batch_size=1, max_chunk_size=2048,
                                draft_model=draft_model, draft_cache=draft_cache)
        self.Job = type("StartupJob", (base.Job,), {"prepare_for_queue":lambda self,g:None})
        modules = {}
        for name in sessions.ENGINE_SOURCE_PINS:
            module = ModuleType(name)
            module.__file__ = str(base.ENGINE / (name.removeprefix("exllamav3.").replace(".", "/") + ".py"))
            modules[name] = module
        modules["exllamav3.generator.generator"].Generator = self.Generator
        modules["exllamav3.generator.job"].Job = self.Job
        modules["exllamav3.generator.pagetable"].PageTable = base.native["PageTable"]
        modules["exllamav3.generator.cpu_cache"] = base.cache_stub
        modules["exllamav3.cache.recurrent"].host_pool = base.native["host_pool"]
        modules["exllamav3.cache.recurrent"].note_freed = base.native["note_freed"]
        self.modules = modules
        self.module_patch = patch.dict(sys.modules, {**policies.stubs, **modules,
                                                    "glm_target_cpu_cache":lazy})
        self.module_patch.start()

    def enable(self, target_bytes=2*8192, factory=None):
        return sessions.enable_session_cache(self.g, max_bytes=1024**2,
                                             target_cpu_bytes=target_bytes, tier_factory=factory)

    def close(self):
        try:
            manager = getattr(self.g, "_glm_dflash_prefix_cache", None)
            if manager is not None: manager.close()
            if self.g.cpu_page_cache is not None: self.g.cpu_page_cache.close()
            self.g.filter_pool.shutdown()
            base.native["host_pool"].release()
        finally:
            self.module_patch.stop()
            for p in reversed(self.patches): p.stop()
            self.context.__exit__(None, None, None)


class SessionStartup(unittest.TestCase):
    def setUp(self): self.f = StartupFixture()
    def tearDown(self): self.f.close()

    def test_native_constructor_true_default_then_fixed_attach(self):
        self.assertIs(self.f.g.enable_defrag, True)
        observed = []
        def factory(caches, budget):
            observed.append(self.f.g.enable_defrag)
            return lazy.LazyTargetCPUPageCache(caches, budget)
        manager = self.f.enable(factory=factory)
        self.assertEqual(observed, [False])
        self.assertIs(self.f.g.enable_defrag, False)
        self.assertIs(self.f.g._glm_dflash_prefix_cache, manager)
        self.assertIs(self.f.g.cpu_page_cache.pagetable, self.f.g.pagetable)
        self.assertIs(self.f.g.cpu_page_cache._generator_owner, self.f.g)
        self.assertEqual(self.f.g.cpu_page_cache.pinned_bytes, 0)

    def test_target_zero_keeps_existing_session_defrag_policy(self):
        self.f.enable(target_bytes=0)
        self.assertIs(self.f.g.enable_defrag, False)
        self.assertIsNone(self.f.g.cpu_page_cache)

    def test_closing_tier_does_not_reenable_defrag_for_installed_manager(self):
        manager = self.f.enable()
        self.f.g.cpu_page_cache.close()
        self.assertIs(self.f.g._glm_dflash_prefix_cache, manager)
        self.assertIs(self.f.g.enable_defrag, False)

    def test_factory_failure_restores_native_defrag_and_no_owner(self):
        with self.assertRaisesRegex(RuntimeError, "factory failed"):
            self.f.enable(factory=lambda *args: (_ for _ in ()).throw(RuntimeError("factory failed")))
        self.assertIs(self.f.g.enable_defrag, True)
        self.assertIsNone(self.f.g.cpu_page_cache)
        self.assertIsNone(self.f.g.pagetable.cpu_tier)
        self.assertFalse(hasattr(self.f.g, "_glm_dflash_prefix_cache"))

    def test_attach_failure_restores_native_defrag_and_closes_owned_tier(self):
        tiers = []
        class Failing(lazy.LazyTargetCPUPageCache):
            def attach(self, table): raise RuntimeError("attach failed")
        def factory(caches, budget):
            tier = Failing(caches, budget); tiers.append(tier); return tier
        with self.assertRaisesRegex(RuntimeError, "attach failed"):
            self.f.enable(factory=factory)
        self.assertIs(self.f.g.enable_defrag, True)
        self.assertIsNone(self.f.g.cpu_page_cache)
        self.assertEqual(tiers[0].segments, [])
        self.assertEqual(tiers[0].pinned_bytes, 0)

    def test_partial_prefix_hook_failure_restores_owned_state(self):
        original = prefix.enable_prefix_cache
        tiers = []
        def factory(caches, budget):
            tier = lazy.LazyTargetCPUPageCache(caches, budget); tiers.append(tier); return tier
        def fail_after_prefix(generator):
            original(generator)
            raise RuntimeError("after prefix hook installation")
        with patch.object(prefix, "enable_prefix_cache", fail_after_prefix), self.assertRaisesRegex(RuntimeError, "after prefix"):
            self.f.enable(factory=factory)
        self.assertIs(self.f.g.enable_defrag, True)
        self.assertIsNone(self.f.g.cpu_page_cache)
        self.assertIsNone(self.f.g.pagetable.cpu_tier)
        self.assertFalse(hasattr(self.f.g, "_glm_dflash_prefix_cache"))
        self.assertEqual(tiers[0].segments, [])
        # Existing prefix class wrappers are instance-gated and inert here.
        self.assertEqual(self.f.g.iterate(), [])

    def test_cleanup_failure_preserves_install_error_and_restores_flags(self):
        class BadClose(lazy.LazyTargetCPUPageCache):
            def close(self):
                super().close(); raise RuntimeError("cleanup failed")
        with patch.object(prefix, "enable_prefix_cache", side_effect=RuntimeError("install failed")):
            with self.assertRaisesRegex(RuntimeError, "install failed") as raised:
                self.f.enable(factory=BadClose)
        self.assertIs(self.f.g.enable_defrag, True)
        self.assertIsNone(self.f.g.cpu_page_cache)
        self.assertIsNone(self.f.g.pagetable.cpu_tier)
        self.assertIn("cleanup failed", str(raised.exception.__notes__))

    def test_active_pending_prior_jobs_and_claimed_pages_refuse_before_flag_change(self):
        for field in ("active_jobs", "pending_jobs"):
            getattr(self.f.g, field).append(object())
            with self.assertRaisesRegex(ValueError, "fresh idle"):
                self.f.enable()
            getattr(self.f.g, field).clear()
            self.assertIs(self.f.g.enable_defrag, True)
        self.f.g.job_serial = 1
        with self.assertRaisesRegex(ValueError, "fresh idle"): self.f.enable()
        self.f.g.job_serial = 0
        self.f.g.pagetable.referenced_pages[b"claimed"] = object()
        with self.assertRaisesRegex(ValueError, "fresh idle"): self.f.enable()
        self.f.g.pagetable.referenced_pages.clear()
        self.assertIs(self.f.g.enable_defrag, True)
        self.assertIsNone(self.f.g.cpu_page_cache)

    def test_foreign_tier_and_manager_refuse_before_flag_change(self):
        for owner, field in ((self.f.g, "cpu_page_cache"), (self.f.g.pagetable, "cpu_tier"),
                             (self.f.g, "_glm_dflash_prefix_cache")):
            setattr(owner, field, object())
            with self.assertRaisesRegex(ValueError, "fresh idle"): self.f.enable()
            setattr(owner, field, None)
            self.assertIs(self.f.g.enable_defrag, True)

    def test_source_drift_refuses_before_flag_change_and_allocation(self):
        with patch.object(sessions, "validate_engine_sources", side_effect=ValueError("source drift")):
            with self.assertRaisesRegex(ValueError, "source drift"): self.f.enable()
        self.assertIs(self.f.g.enable_defrag, True)
        self.assertIsNone(self.f.g.cpu_page_cache)

    def test_direct_tier_attach_reports_actual_true_default_snapshot(self):
        tier = lazy.LazyTargetCPUPageCache([self.f.g.cache], 2*8192)
        try:
            with self.assertRaisesRegex(ValueError, "geometry:") as raised:
                tier.attach(self.f.g.pagetable)
            detail = json.loads(str(raised.exception).split("geometry: ")[1])
            self.assertIs(detail["enable_defrag"], True)
            self.assertEqual(detail["max_batch_size"], 1)
            self.assertEqual(detail["segment_count"], 4)
            self.assertEqual(detail["mismatched_segment_count"], 0)
            self.assertEqual(tier.pinned_bytes, 0)
        finally: tier.close()

    def test_api_retention_gate_leaves_default_off_constructor_unchanged(self):
        tree = ast.parse((REPO / "glm/glm_api.py").read_text())
        lifespan = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "lifespan")
        construction = next(n.value for n in lifespan.body if isinstance(n, ast.Assign)
                            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Call)
                            and isinstance(n.value.func.func, ast.Name) and n.value.func.func.id == "responsive_generator_class")
        self.assertNotIn("enable_defrag", [k.arg for k in construction.keywords])
        gate = next(n for n in lifespan.body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Attribute) and isinstance(n.test.value, ast.Name)
                    and n.test.value.id == "retention" and n.test.attr == "enabled")
        program = ast.Module(body=[gate], type_ignores=[])
        ns = dict(runtime={"generator":NS(generator=self.f.g)},
                  retention=NS(enabled=False, paired_bytes=1024**2, max_checkpoints=8, target_cpu_bytes=2*8192),
                  args=NS(model_dir="/ABS/MODEL"), Path=Path)
        with redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(program), "glm_api.py", "exec"), ns)
        self.assertIs(self.f.g.enable_defrag, True)
        self.assertIsNone(self.f.g.cpu_page_cache)
        ns["retention"].enabled = True
        with redirect_stdout(io.StringIO()):
            exec(compile(program, "glm_api.py", "exec"), ns)
        self.assertIs(self.f.g.enable_defrag, False)
        self.assertEqual(self.f.g.cpu_page_cache.pinned_bytes, 0)


if __name__ == "__main__": unittest.main()
