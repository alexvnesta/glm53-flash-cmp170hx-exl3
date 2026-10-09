"""Stdlib fault contracts. No HTTP, Torch, GPU or service operations."""
from contextlib import nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_active_host_full as probe


def fixture():
    return {'cases':[{'id':'ledger','request':{'model':'GLM-5.3-Flash',
        'messages':[{'role':'system','content':'Preserve identifiers'},
                    {'role':'user','content':'CONTROL_CASE_A ledger'}],
        'max_tokens':128,'temperature':0,'top_p':1,'seed':170,'stream':False,
        'ignore_eos':True,'reasoning_effort':'low'}}]}


def response(prompt=133153):
    return {'choices':[{'index':0,'finish_reason':'length','message':
        {'role':'assistant','content':'MICA-4827','reasoning_content':'Checking records'}}],
        'usage':{'prompt_tokens':prompt,'completion_tokens':128,'total_tokens':prompt+128,
                 'prompt_tokens_details':{'cached_tokens':0},'draft_tokens':512,'draft_tokens_accepted':128}}


def health():
    return {'status':'ok','engine':'ExLlamaV3','model':'GLM-5.3-Flash','cache_tokens':393216,
            'context_length':392960,'speculative_method':'dflash2','draft_num_tokens':7,
            'busy':False,'prompt_cache':None,'max_chunk_size':2048,
            'load_chunk_size':2048,'prefill_chunk_cap':2048,'cache_format':'Q8',
            'draft_cache_format':'Q8','q8_staging':True,'output_gpu':0,
            'workspace_margin_mb':128,'draft_ring_tokens':8192,'recurrent_checkpoint_interval':2048}


class ClientDouble:
    def __init__(self):self.calls=[];self.fail_idle=False;self.bad_health=False
    def health(self,tag):
        self.calls.append(tag);h=health()
        if self.bad_health:h['prompt_cache']={'mode':'multi_session'}
        return h
    def request(self,method,path,payload,tag):
        self.calls.append(tag)
        return {'request':payload,'request_sha256':probe.request_hash(payload),'response':response()}
    def cancel(self,payload):self.calls.append('cancel');return {'generated_text_observed':True}
    def idle(self,cache_tokens):
        self.calls.append('idle')
        if self.fail_idle:raise RuntimeError('Cancellation not drained')
        return health()


def reports():
    reqs=probe.prepare_requests(fixture());record={'http_passed':True,'fixture_sha256':'fixture',
        'requests':reqs,'expected_cache_tokens':393216,'threshold':131072,'expected_prompt_tokens':133153}
    for label,name in [('first','cold_first'),('recovery','fresh_recovery')]:
        record[label]={'request_sha256':probe.request_hash(reqs[name]),'response':response()}
    normal=deepcopy(record);normal['condition']='normal'
    active=deepcopy(record);active['condition']='active-host'
    host=393216*544*11;events=[]
    for epoch in (1,2,3):
        events.extend([{'event':'active_host_migrated','layout_epoch':epoch,'host_bytes':host,
            'GPU_latent_bytes_replaced':host,'actual_allocated_reduction_bytes':100,
            'allocated_before':{'0':1000,'1':1000},'allocated_after':{'0':950,'1':950},
            'GPU_indexer_unchanged':True,'capacity_tokens':393216},
            {'event':'active_host_returned_to_GPU','layout_epoch':epoch,'restored_bytes':host}])
    return normal,active,events


class HTTPContracts(unittest.TestCase):
    def test_private_endpoint_rejects_production_and_remote(self):
        for url in ('http://127.0.0.1:8012','http://example.org:8013','http://localhost',
                    'http://user@localhost:8013','http://localhost:8013/other'):
            with self.assertRaises(ValueError):probe.private_endpoint(url)
        self.assertEqual(probe.private_endpoint('http://127.0.0.1:8013'),('127.0.0.1',8013))

    def test_saved_request_remains_intact_and_mutations_are_explicit(self):
        source=fixture();before=deepcopy(source);reqs=probe.prepare_requests(source)
        self.assertEqual(source,before);self.assertEqual(reqs['cold_first'],source['cases'][0]['request'])
        self.assertEqual(reqs['cancel']['max_tokens'],8192);self.assertTrue(reqs['cancel']['stream'])
        self.assertNotEqual(reqs['fresh_recovery']['messages'],reqs['cold_first']['messages'])
        self.assertEqual(reqs['fresh_recovery']['max_tokens'],128)

    def test_lifecycle_orders_idle_before_fresh_prefill(self):
        client=ClientDouble();result=probe.qualify(client,probe.prepare_requests(fixture()),393216,131072)
        self.assertTrue(result['http_passed']);self.assertFalse(result['full_model_qualified'])
        self.assertEqual(client.calls,['initial_health','cold_first','first_complete_health',
                                       'cancel','idle','fresh_recovery','final_health'])

    def test_failed_cancel_does_not_start_recovery(self):
        client=ClientDouble();client.fail_idle=True
        with self.assertRaises(RuntimeError):probe.qualify(client,probe.prepare_requests(fixture()),393216,131072)
        self.assertNotIn('fresh_recovery',client.calls)

    def test_foreign_prefix_owner_refuses_before_completion(self):
        client=ClientDouble();client.bad_health=True
        with self.assertRaisesRegex(AssertionError,'manager'):
            probe.qualify(client,probe.prepare_requests(fixture()),393216,131072)
        self.assertEqual(client.calls,['initial_health'])

    def test_sse_requires_text_before_disconnect_and_rejects_done(self):
        role=b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
        text=b'data: {"choices":[{"delta":{"reasoning_content":"Checking"},"finish_reason":null}]}\n\n'
        self.assertEqual(probe.read_first_text(io.BytesIO(role+text)),role+text.rstrip(b'\n')+b'\n')
        with self.assertRaisesRegex(RuntimeError,'completed'):
            probe.read_first_text(io.BytesIO(role+b'data: [DONE]\n\n'))

    def test_cancel_closes_response_and_socket(self):
        calls=[]
        class Response(io.BytesIO):
            status=200
            def getheaders(self):return [('Content-Type','text/event-stream')]
            def close(self):calls.append('response_close');super().close()
        response_obj=Response(b'data: {"choices":[{"delta":{"content":"4"},"finish_reason":null}]}\n\n')
        class Connection:
            def request(self,*a,**k):calls.append('request')
            def getresponse(self):return response_obj
            def close(self):calls.append('socket_close')
        with tempfile.TemporaryDirectory() as temp:
            client=probe.Client('http://127.0.0.1:8013',Path(temp),180)
            with patch.object(probe.http.client,'HTTPConnection',return_value=Connection()),patch.object(probe,'deadline',return_value=nullcontext()):
                client.cancel(probe.prepare_requests(fixture())['cancel'])
        self.assertEqual(calls,['request','response_close','socket_close'])

    def test_imported_first_rejects_repeats_or_changed_payload(self):
        payload=probe.prepare_requests(fixture())['cold_first']
        row={'event':'result','completion_attempted':True,'success':True,'http_status':200,
             'request':payload,'request_sha256':probe.request_hash(payload),'response':response(),
             'started_utc':'start','finished_utc':'end'}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'first.jsonl';path.write_text(json.dumps(row)+'\n')
            self.assertEqual(probe.first_from_benchmark(path,payload)['response'],response())
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'exactly one'):probe.first_from_benchmark(path,payload)
            row['request_sha256']='changed';path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'differs'):probe.first_from_benchmark(path,payload)

    def test_parity_preserves_entire_choices_and_semantic_usage(self):
        normal,active,events=reports()
        active['first']['response']['usage']['draft_tokens']=999
        self.assertTrue(probe.compare_reports(normal,active,[],events)['qualified'])
        active['first']['response']['choices'][0]['message']['reasoning_content']='changed'
        with self.assertRaisesRegex(AssertionError,'mismatch'):probe.compare_reports(normal,active,[],events)

    def test_usage_corruption_and_cached_reuse_rejected(self):
        for change in ('total','cache'):
            r=response()
            if change=='total':r['usage']['total_tokens']+=1
            else:r['usage']['prompt_tokens_details']['cached_tokens']=256
            with self.assertRaises(AssertionError):probe.semantic_response(r)

    def test_journal_requires_three_complete_positive_release_cycles(self):
        normal,active,events=reports()
        with self.assertRaisesRegex(AssertionError,'migration and restoration'):
            probe.compare_reports(normal,active,[],events[:-1])
        events[0]['actual_allocated_reduction_bytes']=0
        with self.assertRaisesRegex(AssertionError,'memory release'):probe.compare_reports(normal,active,[],events)

    def test_saved_exact_fixture_is_pinned_and_output_bound128(self):
        raw=(Path(__file__).parent/'matched_fixture_full.json').read_bytes()
        requests=probe.validate_fixture(raw)
        self.assertEqual(requests['cold_first']['max_tokens'],128)
        self.assertEqual(requests['cold_first'],json.loads(raw)['cases'][0]['request'])
        with self.assertRaisesRegex(ValueError,'hash changed'):
            probe.validate_fixture(raw+b' ')

    def test_changed_token_count_fails_before_cancellation(self):
        client=ClientDouble()
        with self.assertRaisesRegex(AssertionError,'prompt count changed'):
            probe.qualify(client,probe.prepare_requests(fixture()),393216,131072,expected_prompt_tokens=133154)
        self.assertNotIn('cancel',client.calls)

    def test_oversized_reply_and_partial_arena_release_are_rejected(self):
        value=response();value['usage']['completion_tokens']=129
        value['usage']['total_tokens']=value['usage']['prompt_tokens']+129
        with self.assertRaisesRegex(AssertionError,'saved bound'):probe.semantic_response(value)
        normal,active,events=reports();events[0]['host_bytes']=65536*544*11
        with self.assertRaisesRegex(AssertionError,'memory release'):
            probe.compare_reports(normal,active,[],events)

    def test_reordered_or_misattributed_epochs_do_not_qualify(self):
        normal,active,events=reports()
        events[0],events[2]=events[2],events[0]
        with self.assertRaisesRegex(AssertionError,'order'):
            probe.compare_reports(normal,active,[],events)

    def test_nested_deadline_never_extends_total_remaining_time(self):
        calls=[]
        with patch.object(probe.signal,'getsignal',return_value='old'), \
             patch.object(probe.signal,'getitimer',return_value=(12,0)), \
             patch.object(probe.signal,'signal'), \
             patch.object(probe.signal,'setitimer',side_effect=lambda *x:calls.append(x)), \
             patch.object(probe.time,'monotonic',side_effect=[100,102]):
            with probe.deadline(300):pass
        self.assertEqual(calls,[(probe.signal.ITIMER_REAL,12),
                               (probe.signal.ITIMER_REAL,0),
                               (probe.signal.ITIMER_REAL,10,0)])

    def test_expired_outer_deadline_does_not_interrupt_receipt_cleanup(self):
        calls=[]
        with patch.object(probe.signal,'getsignal',return_value='old'), \
             patch.object(probe.signal,'getitimer',return_value=(2,0)), \
             patch.object(probe.signal,'signal'), \
             patch.object(probe.signal,'setitimer',side_effect=lambda *x:calls.append(x)), \
             patch.object(probe.time,'monotonic',side_effect=[100,102]):
            with probe.deadline(300):pass
        self.assertEqual(calls,[(probe.signal.ITIMER_REAL,2),(probe.signal.ITIMER_REAL,0)])

    def test_profile_drift_rejected_before_first_request(self):
        client=ClientDouble()
        original=client.health
        def changed(tag):
            h=original(tag);h['workspace_margin_mb']=96;return h
        client.health=changed
        with self.assertRaisesRegex(AssertionError,'Health profile mismatch'):
            probe.qualify(client,probe.prepare_requests(fixture()),393216,131072)
        self.assertEqual(client.calls,['initial_health'])


if __name__=='__main__':unittest.main(verbosity=2)
