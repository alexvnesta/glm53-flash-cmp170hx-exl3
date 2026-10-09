"""Bounded private-port HTTP qualification; no service/GPU management.

The operator runs this explicitly after an isolated candidate is ready.
Requests are calibrated via /tokenize and saved before inference. All response
and health bytes are retained. Default cache geometry is the 64Ki pilot.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import time
import urllib.parse


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def signature(response):
    choice=response['choices'][0]
    usage=response['usage']
    count=usage['completion_tokens']
    if not 0 < count <= 128 or choice['finish_reason'] not in ('stop','length'):
        raise AssertionError('Invalid bounded completion')
    return choice['message'],count,choice['finish_reason']


class Client:
    def __init__(self,base,out):
        u=urllib.parse.urlparse(base)
        if u.scheme!='http' or u.hostname not in ('127.0.0.1','localhost') or u.port in (None,8012):
            raise ValueError('Use an explicit private loopback port; production port 8012 is refused')
        self.host,self.port=u.hostname,u.port
        self.out=out
        self.serial=0

    def request(self,method,path,payload=None,tag='request',timeout=180):
        self.serial+=1
        prefix=f'{self.serial:03d}_{tag}'
        raw=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
        if raw is not None: (self.out/(prefix+'_request.json')).write_bytes(raw)
        c=http.client.HTTPConnection(self.host,self.port,timeout=timeout)
        try:
            c.request(method,path,body=raw,headers={'Content-Type':'application/json'})
            r=c.getresponse(); body=r.read()
            (self.out/(prefix+'_response.json')).write_bytes(body)
            dump(self.out/(prefix+'_http.json'),{'status':r.status,'headers':r.getheaders(),'path':path})
            if r.status!=200: raise RuntimeError(f'{tag}: HTTP {r.status}')
            return json.loads(body)
        finally: c.close()

    def health(self,tag='health'):
        return self.request('GET','/health',tag=tag,timeout=10)

    def cancel_stream(self,payload):
        payload={**payload,'stream':True,'ignore_eos':True,'max_tokens':8192}
        raw=json.dumps(payload,ensure_ascii=False).encode()
        (self.out/'cancel_request.json').write_bytes(raw)
        c=http.client.HTTPConnection(self.host,self.port,timeout=180)
        frames=[]
        try:
            c.request('POST','/v1/chat/completions',body=raw,headers={'Content-Type':'application/json'})
            r=c.getresponse()
            dump(self.out/'cancel_http.json',{'status':r.status,'headers':r.getheaders(),
                                            'path':'/v1/chat/completions'})
            if r.status!=200: raise RuntimeError(f'Cancel stream HTTP {r.status}')
            started=time.monotonic()
            while len(frames)<64 and time.monotonic()-started<180:
                line=r.readline()
                if not line: raise RuntimeError('Stream ended before cancellation')
                frames.append(line.decode())
                if line.strip()==b'data: [DONE]': raise RuntimeError('Stream completed before cancellation')
                if line.startswith(b'data: '):
                    event=json.loads(line[6:])
                    if event.get('error'): raise RuntimeError(event['error'])
                    choices=event.get('choices',[])
                    delta=choices[0].get('delta',{}) if choices else {}
                    if delta.get('content') or delta.get('reasoning_content'):
                        break
            else: raise RuntimeError('No generated text before bounded cancel')
        finally:
            c.close()
            (self.out/'cancel_sse.txt').write_text(''.join(frames))
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            health=self.health('cancel_recovery_health')
            if health['status']=='ok' and not health['busy']:
                return health
            time.sleep(1)
        raise RuntimeError('Cancellation did not return the candidate to idle within 30 seconds')


def content(label,rows):
    # Different leading session tags make entire target hash chains distinct.
    lines=[f'Session {label}. Preserve this independent project ledger.']
    lines.extend(f'Record {i:05d}: component cedar has revision {i%19:02d}; '
                 f'owner is team violet; status is verified; next review is cycle {i%31:02d}.'
                 for i in range(rows))
    lines.append(f'Respond only with SESSION_{label}_READY. Do not summarize the ledger.')
    return '\n'.join(lines)


def build_fixture(client):
    requests={}
    measured={}
    for label,target in (('A',40000),('B',40000),('C',4096),('D',2048)):
        lo,hi=1,2000
        for attempt in range(14):
            rows=(lo+hi)//2
            text=content(label,rows)
            n=client.request('POST','/tokenize',{'model':'GLM-5.3-Flash','content':text},
                             tag=f'tokenize_{label}_{attempt}')['token_length']
            if abs(n-target)<=40: break
            if n<target: lo=rows+1
            else: hi=rows-1
            if lo>hi: raise RuntimeError('Ledger token calibration failed')
        else: raise RuntimeError('Ledger calibration exceeded 14 attempts')
        measured[label]=n
        requests[label]={'model':'GLM-5.3-Flash','messages':[{'role':'user','content':text}],
                         'max_tokens':128,'temperature':0,'top_p':1,'seed':12345,
                         'reasoning_effort':'low','stream':False}
    fixture={'version':1,'tokenized_content_lengths':measured,'requests':requests,
             'sequence':['cold_A','cold_B','resume_A','cold_C','resume_B','cancel_D','recover_B']}
    dump(client.out/'fixture.json',fixture)
    (client.out/'fixture.sha256').write_text(hashlib.sha256((client.out/'fixture.json').read_bytes()).hexdigest()+'\n')
    return fixture


def qualify(client,fixture):
    h=client.health('initial_health')
    pc=h.get('prompt_cache') or {}
    tier=pc.get('target_cpu_tier') or {}
    assert h['cache_tokens']==65536 and h['context_length']==65280, 'Pilot requires a 64Ki target cache'
    assert pc.get('mode')=='multi_session' and not h['busy'], 'Fresh idle session candidate required'
    assert pc['completed_checkpoints']==0 and pc['hits']==0, 'Cold pilot cannot reuse an earlier campaign'
    assert tier.get('reservation_policy')=='pressure_demand_pinned_slabs', 'Demand-only host allocation required'
    assert tier['pinned_bytes']==0 and tier['metrics']['pushes']==0, 'No allocation/spill before pressure'
    responses={}; rows=[]
    sequence=(('cold_A','A',False),('cold_B','B',False),('resume_A','A',True),
              ('cold_C','C',False),('resume_B','B',True))
    for tag,label,warm in sequence:
        r=client.request('POST','/v1/chat/completions',fixture['requests'][label],tag=tag)
        signature(r)
        cached=r['usage']['prompt_tokens_details']['cached_tokens']
        assert (cached>0)==warm, f'{tag}: unexpected cached prefix'
        if label in ('A','B'): assert 39000<r['usage']['prompt_tokens']<42000
        responses[tag]=r
        h=client.health(tag+'_health');pc=h['prompt_cache'];tier=pc['target_cpu_tier']
        assert not h['busy'] and h['status']=='ok'
        assert pc['paired_host_bytes']<=pc['paired_budget_bytes']
        assert tier['pinned_bytes']<=tier['reserved_budget_bytes']
        rows.append({'tag':tag,'cached_tokens':cached,'prompt_tokens':r['usage']['prompt_tokens'],
                     'completion_tokens':r['usage']['completion_tokens'],'health':h})
        if tag=='cold_A': assert tier['metrics']['pushes']==0 and tier['pinned_bytes']==0
        if tag=='cold_B': assert tier['metrics']['pushes']>0 and tier['pinned_bytes']>0
        if tag=='resume_A': assert tier['metrics']['restores']>0
    assert signature(responses['cold_A'])==signature(responses['resume_A']), 'A cold/restored output mismatch'
    assert signature(responses['cold_B'])==signature(responses['resume_B']), 'B cold/restored output mismatch'
    client.cancel_stream(fixture['requests']['D'])
    recovered=client.request('POST','/v1/chat/completions',fixture['requests']['B'],tag='recover_B')
    assert recovered['usage']['prompt_tokens_details']['cached_tokens']>0, 'B lost across unrelated cancellation'
    assert signature(recovered)==signature(responses['cold_B']), 'B output mismatch after cancellation'
    final=client.health('final_health')
    assert final['status']=='ok' and not final['busy']
    return {'qualified':True,'scope':'64Ki LS target; inactive target-only CPU spill; exact saved requests',
            'steps':rows,'cancel_recovery_cached_tokens':recovered['usage']['prompt_tokens_details']['cached_tokens'],
            'final_health':final}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',default='http://127.0.0.1:8013')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fixture',type=Path)
    p.add_argument('--run',action='store_true',help='Explicitly execute private HTTP tokenization/inference')
    args=p.parse_args()
    if not args.run: p.error('Pass --run only after an isolated candidate owns the private test port')
    args.out.mkdir(parents=True,exist_ok=False)
    client=Client(args.base_url,args.out)
    try:
        fixture=json.loads(args.fixture.read_text()) if args.fixture else build_fixture(client)
        if args.fixture:
            dump(args.out/'fixture.json',fixture)
            (args.out/'fixture.sha256').write_text(
                hashlib.sha256((args.out/'fixture.json').read_bytes()).hexdigest()+'\n')
        report=qualify(client,fixture)
    except BaseException as exc:
        dump(args.out/'report.json',{'qualified':False,'error':repr(exc)})
        raise
    dump(args.out/'report.json',report)
    print(json.dumps({'qualified':report['qualified'],'report':str(args.out/'report.json')}))


if __name__=='__main__': main()
