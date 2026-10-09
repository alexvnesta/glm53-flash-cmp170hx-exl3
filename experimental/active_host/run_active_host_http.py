"""Root-owned private HTTP lifecycle probe, no GPU/service imports or actions."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import signal
import time
from urllib.parse import urlparse


def utc():return datetime.now(timezone.utc).isoformat()
def sha(data):return hashlib.sha256(data).hexdigest()
def request_hash(payload):return sha(json.dumps(payload,sort_keys=True).encode())
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


@contextmanager
def deadline(seconds):
    handler=signal.getsignal(signal.SIGALRM)
    previous=signal.getitimer(signal.ITIMER_REAL)
    started=time.monotonic()
    def expired(signum,frame):raise TimeoutError('HTTP total deadline exceeded')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
        if previous[0]:
            signal.setitimer(signal.ITIMER_REAL,max(.000001,previous[0]-(time.monotonic()-started)),previous[1])


def private_endpoint(base):
    url=urlparse(base)
    if (url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost')
            or url.port in (None,8012) or url.username or url.password
            or url.path not in ('','/') or url.query or url.fragment):
        raise ValueError('Use an explicit private loopback HTTP port; production8012 is refused')
    return url.hostname,url.port


def prepare_requests(fixture):
    cases=fixture.get('cases')
    if not isinstance(cases,list) or len(cases)!=1:raise ValueError('Exactly one saved first-request case required')
    first=deepcopy(cases[0]['request'])
    if (first.get('model')!='GLM-5.3-Flash' or first.get('stream') is not False
            or first.get('max_tokens')!=256 or first.get('temperature')!=0
            or first.get('top_p')!=1 or not isinstance(first.get('seed'),int)):
        raise ValueError('Expected canonical deterministic256-output saved fixture')
    cancellation={**deepcopy(first),'stream':True,'ignore_eos':True,'max_tokens':8192}
    recovery=deepcopy(first)
    user=next((m for m in recovery['messages'] if m.get('role')=='user'),None)
    if user is None or not isinstance(user.get('content'),str):raise ValueError('Text user request required')
    user['content']='ACTIVE_HOST_FRESH_PREFILL_CASE_B.\n'+user['content']
    return {'cold_first':first,'cancel':cancellation,'fresh_recovery':recovery}


def semantic_response(response):
    choices=response['choices'];usage=deepcopy(response['usage'])
    if not isinstance(choices,list) or len(choices)!=1:raise AssertionError('Expected one full choice')
    if choices[0].get('finish_reason') not in ('stop','length'):raise AssertionError('Invalid finish reason')
    if not isinstance(choices[0].get('message'),dict):raise AssertionError('Missing complete message')
    for key in ('prompt_tokens','completion_tokens','total_tokens'):
        if type(usage.get(key)) is not int or usage[key]<=0:raise AssertionError('Invalid semantic usage')
    if usage['total_tokens']!=usage['prompt_tokens']+usage['completion_tokens']:
        raise AssertionError('Usage sum mismatch')
    if usage['completion_tokens']>256:raise AssertionError('Output exceeded saved bound')
    details=usage.get('prompt_tokens_details')
    if not isinstance(details,dict) or details.get('cached_tokens')!=0:
        raise AssertionError('This no-prefix pilot requires cached_tokens0')
    # Preserve every other usage field, including completion-token details.
    # Only cache and draft performance counts are excluded from parity.
    details.pop('cached_tokens',None)
    if not details:usage.pop('prompt_tokens_details',None)
    for key in ('draft_tokens','draft_tokens_accepted','draft_stats','cached_tokens'):
        usage.pop(key,None)
    return {'choices':choices,'usage':usage}


def health_guard(health,cache_tokens):
    expected={'status':'ok','engine':'ExLlamaV3','model':'GLM-5.3-Flash',
        'cache_tokens':cache_tokens,'context_length':cache_tokens-256,
        'speculative_method':'dflash2','draft_num_tokens':7,'busy':False}
    mismatch={k:{'expected':v,'actual':health.get(k)} for k,v in expected.items()
              if health.get(k)!=v or (k=='busy' and health.get(k) is not False)}
    if mismatch:raise AssertionError('Health profile mismatch: '+json.dumps(mismatch))
    if health.get('prompt_cache') is not None:raise AssertionError('Prefix/session manager enabled')
    if health.get('target_cpu_cache_budget_bytes',0)!=0:raise AssertionError('General CPU page tier enabled')


def first_from_benchmark(path,payload):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    attempted=[r for r in rows if r.get('event')=='result' and r.get('completion_attempted')]
    if len(attempted)!=1:raise ValueError('Import requires exactly one attempted first benchmark request')
    record=attempted[0]
    if not record.get('success') or record.get('http_status')!=200:raise ValueError('First benchmark request failed')
    if record.get('request')!=payload or record.get('request_sha256')!=request_hash(payload):
        raise ValueError('Imported first request differs from fixture')
    semantic_response(record['response'])
    return {'request':payload,'request_sha256':request_hash(payload),'response':record['response'],
            'started_utc':record['started_utc'],'finished_utc':record['finished_utc'],
            'http_status':200,'wall_seconds':record.get('wall_seconds'),
            'imported_receipt_sha256':sha(path.read_bytes())}


def read_first_text(response,frames=None):
    frames=[] if frames is None else frames
    data_events=0
    for _ in range(512):
        line=response.readline()
        if not line:raise RuntimeError('Stream ended before deliberate cancellation')
        frames.append(line)
        if line.strip()==b'data: [DONE]':raise RuntimeError('Stream completed before deliberate cancellation')
        if not line.startswith(b'data: '):continue
        data_events+=1
        if data_events>64:raise RuntimeError('No generated text within64 SSE events')
        event=json.loads(line[6:])
        if event.get('error'):raise RuntimeError('SSE API error: '+str(event['error']))
        choices=event.get('choices',[])
        if choices and choices[0].get('finish_reason') is not None:
            raise RuntimeError('Generation finished before deliberate cancellation')
        delta=choices[0].get('delta',{}) if choices else {}
        if delta.get('content') or delta.get('reasoning_content'):return b''.join(frames)
    raise RuntimeError('SSE line bound exhausted')


class Client:
    def __init__(self,base,out,timeout):
        self.host,self.port=private_endpoint(base);self.out,self.timeout=out,timeout
        self.serial=0
    def request(self,method,path,payload=None,tag='health'):
        self.serial+=1;prefix=self.out/f'{self.serial:03d}_{tag}'
        started=time.monotonic();record={'started_utc':utc(),'request':payload,
            'request_sha256':request_hash(payload) if payload is not None else None}
        raw=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
        if raw is not None:prefix.with_suffix('.request.json').write_bytes(raw)
        connection=http.client.HTTPConnection(self.host,self.port,timeout=min(self.timeout,10) if payload is None else self.timeout)
        body=b''
        try:
            with deadline(min(self.timeout,10) if payload is None else self.timeout):
                connection.request(method,path,body=raw,headers={'Content-Type':'application/json'})
                response=connection.getresponse();record['http_status']=response.status
                record['headers']=response.getheaders();body=response.read()
            prefix.with_suffix('.response.json').write_bytes(body)
            record['response']=json.loads(body)
            if response.status!=200 or record['response'].get('error'):
                raise RuntimeError(f'{tag}: unsuccessful HTTP/API response')
            return record
        except BaseException as error:
            if isinstance(error,http.client.IncompleteRead):body=error.partial
            prefix.with_suffix('.response.json').write_bytes(body)
            record['error']=repr(error);raise
        finally:
            connection.close();record.update(finished_utc=utc(),wall_seconds=time.monotonic()-started)
            dump(prefix.with_suffix('.http.json'),record)
    def health(self,tag):return self.request('GET','/health',tag=tag)['response']
    def cancel(self,payload):
        prefix=self.out/'cancel';raw=json.dumps(payload,ensure_ascii=False).encode()
        prefix.with_suffix('.request.json').write_bytes(raw)
        connection=http.client.HTTPConnection(self.host,self.port,timeout=self.timeout)
        response=None;record={'started_utc':utc(),'request_sha256':request_hash(payload)}
        started=time.monotonic();lines=[]
        try:
            with deadline(self.timeout):
                connection.request('POST','/v1/chat/completions',body=raw,headers={'Content-Type':'application/json'})
                response=connection.getresponse();record['http_status']=response.status
                record['headers']=response.getheaders()
                if response.status!=200:
                    lines.append(response.read())
                    raise RuntimeError('Cancellation stream HTTP failure')
                read_first_text(response,lines)
                record['generated_text_observed']=True
        except BaseException as error:record['error']=repr(error);raise
        finally:
            if response is not None:response.close()
            connection.close()  # Both file and socket close BEFORE idle polling.
            prefix.with_suffix('.sse.txt').write_bytes(b''.join(lines))
            record.update(disconnected_utc=utc(),wall_seconds=time.monotonic()-started)
            dump(prefix.with_suffix('.http.json'),record)
        return record
    def idle(self,cache_tokens):
        end=time.monotonic()+30
        while time.monotonic()<end:
            health=self.health('cancel_idle_health')
            if health.get('busy') is False:
                health_guard(health,cache_tokens);return health
            time.sleep(1)
        raise RuntimeError('Cancelled request did not become idle within30 seconds')


def qualify(client,requests,cache_tokens,threshold,imported_first=None):
    initial=client.health('initial_health');health_guard(initial,cache_tokens)
    first=imported_first or client.request('POST','/v1/chat/completions',requests['cold_first'],tag='cold_first')
    semantic_response(first['response'])
    if first['response']['usage']['prompt_tokens']<threshold:raise AssertionError('Fixture below migration threshold')
    health_guard(client.health('first_complete_health'),cache_tokens)
    cancel=client.cancel(requests['cancel'])
    cancelled_health=client.idle(cache_tokens)
    recovery=client.request('POST','/v1/chat/completions',requests['fresh_recovery'],tag='fresh_recovery')
    semantic_response(recovery['response'])
    if recovery['response']['usage']['prompt_tokens']<threshold:raise AssertionError('Recovery below migration threshold')
    final=client.health('final_health');health_guard(final,cache_tokens)
    return {'http_passed':True,'full_model_qualified':False,
            'qualification_pending':'paired response comparison and root journal migration/restoration telemetry',
            'first':first,'cancel':cancel,'cancel_idle_health':cancelled_health,'recovery':recovery,'final_health':final}


def journal_events(path):
    decoder=json.JSONDecoder();events=[]
    for line in path.read_text().splitlines():
        if '{' not in line:continue
        try:value,_=decoder.raw_decode(line[line.index('{'):])
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and value.get('event') in ('active_host_migrated','active_host_returned_to_GPU'):
            events.append(value)
    return events


def compare_reports(normal,active,normal_events,active_events):
    if normal.get('condition')!='normal' or active.get('condition')!='active-host':
        raise AssertionError('Condition identities do not match the paired arms')
    if not normal.get('http_passed') or not active.get('http_passed'):raise AssertionError('Both HTTP arms must pass')
    for key in ('fixture_sha256','requests','expected_cache_tokens','threshold'):
        if normal.get(key)!=active.get(key):raise AssertionError('Unmatched arms: '+key)
    comparisons={}
    for label in ('first','recovery'):
        a,b=normal[label],active[label]
        if a['request_sha256']!=b['request_sha256']:raise AssertionError('Request hash mismatch: '+label)
        comparisons[label]=semantic_response(a['response'])==semantic_response(b['response'])
        if not comparisons[label]:raise AssertionError('Full choices/semantic usage mismatch: '+label)
    if normal_events:raise AssertionError('Normal arm journal contains active-host transitions')
    migrated=[e for e in active_events if e['event']=='active_host_migrated']
    restored=[e for e in active_events if e['event']=='active_host_returned_to_GPU']
    if len(migrated)!=3 or len(restored)!=3:raise AssertionError('Expected first/cancel/recovery migration and restoration')
    by_epoch={e['layout_epoch']:e for e in migrated}
    if len(by_epoch)!=3 or {e['layout_epoch'] for e in restored}!=set(by_epoch):
        raise AssertionError('Incomplete/duplicate migration lifecycle epochs')
    for event in migrated:
        expected_latent_bytes=active['expected_cache_tokens']*544*11
        if (event.get('actual_allocated_reduction_bytes',0)<=0 or
                event.get('GPU_indexer_unchanged') is not True or
                event.get('capacity_tokens')!=active['expected_cache_tokens'] or
                event.get('host_bytes')!=expected_latent_bytes or
                event.get('GPU_latent_bytes_replaced')!=expected_latent_bytes):
            raise AssertionError('Actual memory release/indexer/capacity telemetry did not qualify')
        before,after=event['allocated_before'],event['allocated_after']
        if (set(before)!={'0','1'} or set(after)!={'0','1'} or
                sum(before.values())-sum(after.values())!=event['actual_allocated_reduction_bytes']):
            raise AssertionError('Actual allocator delta does not match before/after receipts')
    for event in restored:
        if event.get('restored_bytes')!=by_epoch[event['layout_epoch']]['host_bytes']:
            raise AssertionError('Incomplete restored arena receipt')
    return {'qualified':True,'comparisons':comparisons,'migration_events':migrated,
            'restoration_events':restored,'scope':'64Ki initial GPU cache; fixed selected-latent decode host migration; exact saved synthetic request and lifecycle only; no larger capacity or concurrent sessions'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--run',action='store_true')
    run.add_argument('--fixture',type=Path,required=True);run.add_argument('--first-jsonl',type=Path)
    run.add_argument('--out',type=Path,required=True);run.add_argument('--condition',choices=['normal','active-host'],required=True)
    run.add_argument('--base-url',default='http://127.0.0.1:8013');run.add_argument('--timeout',type=float,default=180)
    run.add_argument('--expected-cache-tokens',type=int,default=65536);run.add_argument('--threshold',type=int,default=8192)
    comp=sub.add_parser('compare');comp.add_argument('--normal',type=Path,required=True)
    comp.add_argument('--active',type=Path,required=True);comp.add_argument('--normal-journal',type=Path,required=True)
    comp.add_argument('--active-journal',type=Path,required=True);comp.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='compare':
        if args.output.exists() or args.output.is_symlink():raise ValueError('Comparison output must be fresh')
        result=compare_reports(json.loads(args.normal.read_text()),json.loads(args.active.read_text()),
            journal_events(args.normal_journal),journal_events(args.active_journal))
        result.update(normal_journal_sha256=sha(args.normal_journal.read_bytes()),active_journal_sha256=sha(args.active_journal.read_bytes()))
        with args.output.open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'qualified':True,'output':str(args.output)}));return
    if not args.run:raise ValueError('Explicit --run is required for private HTTP actions')
    private_endpoint(args.base_url)
    if not math.isfinite(args.timeout) or not 0<args.timeout<=180:raise ValueError('Timeout must be0..180 seconds')
    if args.expected_cache_tokens!=65536 or args.threshold!=8192:raise ValueError('First pilot is fixed64Ki/threshold8192')
    raw=args.fixture.read_bytes();requests=prepare_requests(json.loads(raw))
    first=first_from_benchmark(args.first_jsonl,requests['cold_first']) if args.first_jsonl else None
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'fixture.json').write_bytes(raw);dump(args.out/'requests.json',requests)
    report={'condition':args.condition,'fixture_sha256':sha(raw),'requests':requests,
        'expected_cache_tokens':args.expected_cache_tokens,'threshold':args.threshold,'started_utc':utc(),
        'http_passed':False,'full_model_qualified':False}
    try:report.update(qualify(Client(args.base_url,args.out,args.timeout),requests,args.expected_cache_tokens,args.threshold,first))
    except BaseException as error:report['error']=repr(error);raise
    finally:report['finished_utc']=utc();dump(args.out/'report.json',report)
    print(json.dumps({'http_passed':True,'full_model_qualified':False,'report':str(args.out/'report.json')}))


if __name__=='__main__':main()
