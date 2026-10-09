"""Fault-inject the actual adapter transactions with independent CPU stand-ins."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine_adapter import ActiveHostTrialAdapter, TrialPolicy
import selected_host_q8


class Tensor:
    def __init__(self,shape,*,device='cpu',dtype=None):
        self.shape=(shape,) if isinstance(shape,int) else tuple(shape)
        self.device=SimpleNamespace(type=device,index=0)
        self.payload=None
    def copy_(self,other):self.payload=other.payload;return self
    def numel(self):
        result=1
        for n in self.shape:result*=n
        return result
    def __getitem__(self,index):
        end=index.stop if isinstance(index,slice) else 1
        return Tensor((end,),device=self.device.type)
    def view(self,*shape):
        self.shape=tuple(shape);return self


class Torch:
    int32='int32';int64='int64';float16='float16'
    def __init__(self):
        self.calls=0;self.fail_at=None
        self.cuda=SimpleNamespace(is_current_stream_capturing=lambda:False,
            current_stream=lambda *a:SimpleNamespace(synchronize=lambda:None),
            device=lambda *a:nullcontext(),memory_allocated=lambda *a:0)
    def empty(self,shape,*,dtype=None,device='cpu',pin_memory=False):
        self.calls+=1
        if self.calls==self.fail_at:raise MemoryError('injected allocation failure')
        kind=device if isinstance(device,str) else device.type
        return Tensor(shape,device=kind,dtype=dtype)
    def full(self,shape,fill,*,dtype=None,device=None):
        t=Tensor(shape,device='cuda',dtype=dtype);t.payload=fill;return t
    def arange(self,*args,**kwargs):return Tensor((8,),device='cuda')


class Stager:
    def __init__(self,extension,q,s,**kwargs):
        self.host_q,self.host_s=q,s
        self.q_alias,self.s_alias=Tensor(q.shape,device='cuda'),Tensor(s.shape,device='cuda')
        self.q_alias.payload,self.s_alias.payload=q.payload,s.payload
        self.device=SimpleNamespace(type='cuda',index=0)
        self._event=None;self.closed=False
    def close(self):self.closed=True


def adapter():
    a=ActiveHostTrialAdapter.__new__(ActiveHostTrialAdapter)
    a.policy=TrialPolicy(enabled=True,context_threshold=4096)
    a.generator=SimpleNamespace(active_jobs=[SimpleNamespace(
        sequences=[SimpleNamespace(prefill_complete=True,kv_position=8192)],
        is_requeued=False,orig_max_rq_tokens=None)],pending_jobs=[],
        cache=SimpleNamespace(max_num_tokens=393216),cpu_page_cache=None,
        pagetable=SimpleNamespace(cpu_tier=None))
    a.torch=Torch();a.extension=None;a.states={};a.active=False;a.epoch=0
    a._check_attention_safety=lambda:None
    import threading
    a.lock=threading.RLock();a.host_bytes=0;a.layers={}
    for key in range(11):
        q=Tensor((1,256,128),device='cuda');q.payload=('packed',key)
        s=Tensor((1,256,16),device='cuda');s.payload=('scale',key)
        a.layers[key]=SimpleNamespace(qk=q,sk=s,qshape=q.shape,sshape=s.shape)
    return a


class AdapterTransactionTests(unittest.TestCase):
    def test_disabled_default_precedes_torch_import(self):
        with self.assertRaises(RuntimeError):ActiveHostTrialAdapter(None,None)

    def test_copy_failure_does_not_publish_partial_host_layout(self):
        a=adapter();before=[(l.qk,l.sk) for l in a.layers.values()]
        a.torch.fail_at=7
        with patch.object(selected_host_q8,'SelectedHostQ8',Stager):
            with self.assertRaises(MemoryError):a.activate()
        self.assertFalse(a.active);self.assertFalse(a.states)
        self.assertEqual([(l.qk,l.sk) for l in a.layers.values()],before)

    def test_migration_and_restoration_keep_packed_and_scale_payloads(self):
        a=adapter()
        with patch.object(selected_host_q8,'SelectedHostQ8',Stager):a.activate()
        self.assertTrue(a.active)
        for key,l in a.layers.items():
            self.assertEqual(l.qk.payload,('packed',key));self.assertEqual(l.sk.payload,('scale',key))
        a.deactivate()
        self.assertFalse(a.active)
        for key,l in a.layers.items():
            self.assertEqual(l.qk.payload,('packed',key));self.assertEqual(l.sk.payload,('scale',key))

    def test_failed_GPU_restoration_keeps_all_host_aliases_published(self):
        a=adapter()
        with patch.object(selected_host_q8,'SelectedHostQ8',Stager):a.activate()
        before=[(l.qk,l.sk) for l in a.layers.values()]
        a.torch.fail_at=a.torch.calls+7
        with self.assertRaises(MemoryError):a.deactivate()
        self.assertTrue(a.active)
        self.assertEqual([(l.qk,l.sk) for l in a.layers.values()],before)

    def test_no_migration_below_explicit_threshold(self):
        a=adapter();a.generator.active_jobs[0].sequences[0].kv_position=2048
        a.activate()
        self.assertFalse(a.active);self.assertEqual(a.torch.calls,0)

    def test_migration_rejects_requeue_and_pending_second_request(self):
        for change in ('requeue','pending'):
            a=adapter()
            if change=='requeue':a.generator.active_jobs[0].is_requeued=True
            else:a.generator.pending_jobs=[object()]
            with self.assertRaises(RuntimeError):a.activate()
            self.assertFalse(a.active);self.assertEqual(a.torch.calls,0)

    def test_capture_rejected_before_transaction(self):
        a=adapter();a.torch.cuda.is_current_stream_capturing=lambda:True
        with self.assertRaises(RuntimeError):a.activate()
        self.assertFalse(a.active);self.assertEqual(a.torch.calls,0)


if __name__=='__main__':unittest.main(verbosity=2)
