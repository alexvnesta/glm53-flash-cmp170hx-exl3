"""Offline HTTP workflow tests. No sockets or model imports."""
import copy
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from qualify_session_cache import Client,content,qualify,signature


class ClientDouble:
    def __init__(self,early_stop=False):
        self.tag='initial';self.early_stop=early_stop;self.calls=[]
    def health(self,tag='health'):
        step=tag.replace('_health','')
        before=step in ('initial','cold_A')
        return {'status':'ok','busy':False,'cache_tokens':65536,'context_length':65280,
                'prompt_cache':{'mode':'multi_session','hits':0 if step=='initial' else 2,
                  'completed_checkpoints':0 if step=='initial' else 2,
                  'paired_host_bytes':400000000,'paired_budget_bytes':1073741824,
                  'target_cpu_tier':{'reservation_policy':'pressure_demand_pinned_slabs',
                     'pinned_bytes':0 if before else 1000000,'reserved_budget_bytes':1073741824,
                     'metrics':{'pushes':0 if before else 8,'restores':0 if step=='cold_B' else 3}}}}
    def request(self,method,path,payload,tag):
        self.calls.append(tag)
        warm=tag in ('resume_A','resume_B','recover_B')
        label=payload['label']
        return {'choices':[{'message':{'role':'assistant','content':'stable_'+label},
                            'finish_reason':'stop' if self.early_stop else 'length'}],
                'usage':{'prompt_tokens':40000 if label in ('A','B') else 4096,
                         'completion_tokens':3 if self.early_stop else 128,
                         'prompt_tokens_details':{'cached_tokens':38912 if warm else 0}}}
    def cancel_stream(self,payload):
        self.calls.append('cancel_D');return self.health('cancel_recovery_health')


class HTTPWorkflow(unittest.TestCase):
    def fixture(self): return {'requests':{x:{'label':x} for x in 'ABCD'}}
    def test_complete_workflow_and_early_stop(self):
        for early in (False,True):
            c=ClientDouble(early)
            self.assertTrue(qualify(c,self.fixture())['qualified'])
            self.assertEqual(c.calls,['cold_A','cold_B','resume_A','cold_C','resume_B','cancel_D','recover_B'])
    def test_no_actual_pressure_refuses(self):
        c=ClientDouble();original=c.health
        def health(tag):
            r=original(tag);r['prompt_cache']['target_cpu_tier']['metrics']['pushes']=0
            return r
        c.health=health
        with self.assertRaises(AssertionError):qualify(c,self.fixture())
    def test_output_mismatch_refuses(self):
        c=ClientDouble();original=c.request
        def request(*args,**kw):
            r=original(*args,**kw)
            if kw['tag']=='resume_A':r['choices'][0]['message']['content']='changed'
            return r
        c.request=request
        with self.assertRaisesRegex(AssertionError,'output mismatch'):qualify(c,self.fixture())
    def test_production_port_refused(self):
        with self.assertRaisesRegex(ValueError,'8012'):Client('http://127.0.0.1:8012',None)
    def test_distinct_session_tags_before_shared_ledger(self):
        a,b=content('A',2),content('B',2)
        self.assertNotEqual(a.splitlines()[0],b.splitlines()[0])
        self.assertEqual(a.splitlines()[1:-1],b.splitlines()[1:-1])


if __name__=='__main__':unittest.main()
