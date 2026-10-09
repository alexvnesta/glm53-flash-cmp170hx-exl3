"""Exercise the actual combined factory on a persistent CPU executor."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import test_sessions as base
import glm_async
import glm_dflash_sessions

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experimental/active_host'))
import engine_adapter


class CombinedAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls=[];self.entered=threading.Event();self.release=threading.Event()
        self.main=threading.get_ident()
        def native_cancel(job):
            self.calls.append(('cancel_started',threading.get_ident()))
            self.entered.set()
            if not self.release.wait(3):raise TimeoutError('test worker stalled')
            self.calls.append(('cancel_finished',threading.get_ident()))
        self.g=NS(cancel=native_cancel)
        calls=self.calls
        class Native:
            def __init__(self,g):
                self.generator=g;self.jobs={};self.error=None
            async def close(self):calls.append(('base_close',threading.get_ident()))
        native_async=ModuleType('exllamav3.generator.async_generator')
        native_async.__file__=str(base.ENGINE/'generator/async_generator.py')
        native_async._CANCELLED_SENTINEL=object()
        self.sentinel=native_async._CANCELLED_SENTINEL
        generator_module=ModuleType('exllamav3.generator');generator_module.__path__=[]
        generator_module.async_generator=native_async
        engine_module=ModuleType('exllamav3');engine_module.__path__=[]
        engine_module.generator=generator_module
        self.patches=[patch.dict(sys.modules,{engine_module.__name__:engine_module,
                       generator_module.__name__:generator_module,native_async.__name__:native_async}),
                      patch.object(glm_async,'responsive_generator_class',glm_async.responsive_generator_class),
                      patch.object(glm_dflash_sessions,'enable_session_cache',glm_dflash_sessions.enable_session_cache)]
        for p in self.patches:p.start()
        engine_adapter.install_combined_constructor_hook(None,policy=engine_adapter.TrialPolicy(enabled=True))
        self.asyncg=glm_async.responsive_generator_class(Native)(self.g)

    async def asyncTearDown(self):
        self.release.set()
        self.asyncg._gpu_worker.shutdown(wait=True,cancel_futures=True)
        for p in reversed(self.patches):p.stop()

    async def wait_entered(self):
        for _ in range(100):
            if self.entered.is_set():return
            await asyncio.sleep(.005)
        self.fail('worker did not start')

    async def test_cancel_drains_repeated_cancellation_then_bookkeeping(self):
        job=NS(job=object(),results=[])
        job.put_result=job.results.append
        self.asyncg.jobs[job.job]=job
        task=asyncio.create_task(self.asyncg.cancel(job));await self.wait_entered()
        self.assertIn(job.job,self.asyncg.jobs)
        task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done());self.assertEqual(job.results,[])
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertNotIn(job.job,self.asyncg.jobs);self.assertEqual(job.results,[self.sentinel])
        self.assertNotEqual(self.calls[0][1],self.main)
        self.assertEqual(self.calls[0][1],self.calls[1][1])

    async def test_close_returns_layout_on_worker_before_base_close(self):
        def restore(reason):
            self.calls.append(('restore',threading.get_ident()))
            self.entered.set()
            if not self.release.wait(3):raise TimeoutError('test worker stalled')
        coordinator=NS(ensure_gpu=restore,
            adapter=NS(close=lambda:self.calls.append(('adapter_close',threading.get_ident()))),
            close=lambda:self.calls.append(('coordinator_close',threading.get_ident())))
        self.g._glm_combined_memory_coordinator=coordinator
        task=asyncio.create_task(self.asyncg.close());await self.wait_entered()
        task.cancel();await asyncio.sleep(0);task.cancel();await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.release.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual([r[0] for r in self.calls],['restore','adapter_close','coordinator_close','base_close'])
        self.assertEqual(len({r[1] for r in self.calls[:3]}),1)
        self.assertNotEqual(self.calls[0][1],self.main);self.assertEqual(self.calls[3][1],self.main)

    async def test_cancel_worker_fault_is_published_to_all_jobs(self):
        def fail(job):raise RuntimeError('worker fault')
        self.g.cancel=fail
        jobs=[NS(job=object(),results=[]) for _ in range(2)]
        for job in jobs:job.put_result=job.results.append;self.asyncg.jobs[job.job]=job
        with self.assertRaisesRegex(RuntimeError,'worker fault'):await self.asyncg.cancel(jobs[0])
        self.assertEqual(self.asyncg.jobs,{})
        self.assertTrue(all(job.results==[self.asyncg.error] for job in jobs))
