"""Root-only bounded native/DSA probe, intentionally not run by its author."""
import argparse
import hashlib
import importlib.util
import json
import csv
import os
from pathlib import Path
import subprocess
import gc


def validate_native_receipt(receipt):
    if receipt['identity']['name'] != 'glm53_hostq8_v3_ext':
        raise RuntimeError('v3 probe requires its distinct newly-built native extension')
    root=Path(__file__).resolve().parent
    expected={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root/'native').glob('selected_host_q8.*')}
    actual={Path(p).name:sha for p,sha in receipt['identity']['sources'].items()}
    if actual!=expected:
        raise RuntimeError('native build receipt source hashes disagree with this v3 bundle')


def probe_pinned_alias_lifecycle(torch, extension, devices, records):
    """Synthetic mapped-host checks; each cross-device access is fully drained."""
    if len(devices)!=2 or len(set(devices))!=2:
        raise RuntimeError('v3 alias lifecycle probe requires exactly two distinct GPUs')
    # Initialize both contexts before asking the shared pinned allocator for a slab.
    # Current-device allocation is not assumed to determine its registration device.
    for d in devices:
        with torch.cuda.device(d):
            scratch=torch.empty(1,device=torch.device('cuda',d))
            torch.cuda.current_stream(d).synchronize()
            del scratch
    seen={};reuses=0;cross_reuses=0;allocation_device_mismatches=0
    for iteration in range(12):
        allocating_device=devices[iteration%2]
        with torch.cuda.device(allocating_device):
            host=torch.empty(4096,dtype=torch.int32,pin_memory=True)
        ptr=host.data_ptr();prior=seen.get(ptr)
        reused=prior is not None
        cross_reused=reused and prior!=allocating_device
        reuses+=int(reused);cross_reuses+=int(cross_reused)
        seen[ptr]=allocating_device
        host.fill_(100+iteration)
        for requested in (allocating_device,devices[1-iteration%2]):
            info=dict(extension.pinned_alias_info(host,requested))
            alias=extension.pinned_alias(host,requested)
            if alias.device!=torch.device('cuda',requested):
                raise AssertionError('mapped host alias has wrong requested device')
            if info['current_device']!=requested or not info['host_type_verified']:
                raise AssertionError('host mapping capability check failed')
            allocation_device_mismatches+=int(info['allocation_device']!=requested)
            with torch.cuda.device(requested):
                # clone enqueues a real GPU read of the mapped host slab.
                clone=alias.clone()
                torch.cuda.current_stream(requested).synchronize()
                torch.testing.assert_close(clone.cpu(),host,rtol=0,atol=0)
                marker=2000+iteration*10+requested
                alias.fill_(marker)
                torch.cuda.current_stream(requested).synchronize()
                if not bool(torch.all(host==marker)):
                    raise AssertionError('GPU mapped-host write was not coherent after completion')
            records.append(dict(test='pinned_alias_cross_device_read_write',passed=True,
                iteration=iteration,allocating_device=allocating_device,
                reused_pointer=reused,reused_from_other_allocating_device=cross_reused,
                **info))
            del clone,alias
        # Neither alias nor CPU owner escapes this iteration: allocator can reuse.
        del host
        gc.collect()
    if allocation_device_mismatches==0:
        raise AssertionError('probe never exercised a differing host registration/request device')
    records.append(dict(test='pinned_allocator_reuse_observation',passed=True,
        allocation_iterations=12,reused_allocations=reuses,
        reused_across_allocating_device=cross_reuses,
        registration_device_mismatches=allocation_device_mismatches,
        reuse_observed=reuses>0,cross_allocating_device_reuse_observed=cross_reuses>0))
    # CPU Tensor destruction must not recycle its slab while the native alias owns it.
    host=torch.full((4096,),314159,dtype=torch.int32,pin_memory=True)
    alias=extension.pinned_alias(host,devices[0]);owner_ptr=host.data_ptr()
    del host;gc.collect()
    pressure=[]
    for _ in range(4):
        other=torch.empty(4096,dtype=torch.int32,pin_memory=True)
        if other.data_ptr()==owner_ptr:
            raise AssertionError('pinned owner slab recycled while alias still exists')
        other.fill_(-1);pressure.append(other)
    with torch.cuda.device(devices[0]):
        clone=alias.clone();torch.cuda.current_stream(devices[0]).synchronize()
    if not bool(torch.all(clone.cpu()==314159)):
        raise AssertionError('retained owner contents changed')
    records.append(dict(test='pinned_alias_retains_cpu_storage_owner',passed=True))
    del clone,alias,pressure
    # Actual device and managed pointers are never accepted through this API.
    # Managed memory is not created here; the mandatory Host-only native gate is
    # source-tested separately. Exercise real device and unpinned CPU rejection.
    for name,tensor in [('actual_cuda_allocation',torch.empty(16,dtype=torch.int32,
                         device=torch.device('cuda',devices[1]))),
                        ('unpinned_cpu_allocation',torch.empty(16,dtype=torch.int32))]:
        try:extension.pinned_alias(tensor,devices[0])
        except RuntimeError as error:
            if 'contiguous pinned CPU storage' not in str(error):raise
        else:raise AssertionError(name+' was relabeled as mapped host memory')
        records.append(dict(test='pinned_alias_rejects_'+name,passed=True))


def preimport_guards(args):
    """Fail with stdlib checks before importing Torch/native or creating CUDA."""
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError('refusing a pre-existing probe output')
    device_list=[int(x) for x in args.devices.split(',')]
    devices=set(device_list)
    if len(device_list)!=2 or len(devices)!=2:
        raise RuntimeError('v3 probe requires exactly two distinct requested GPUs')
    query=lambda fields:subprocess.check_output(
        ['nvidia-smi','--query-'+fields,'--format=csv,noheader,nounits'],
        text=True,timeout=10)
    uuid_by_device={int(r[0].strip()):r[1].strip() for r in
        csv.reader(query('gpu=index,uuid').splitlines()) if r}
    if not devices or not devices.issubset(uuid_by_device):
        raise RuntimeError('requested GPU inventory does not match nvidia-smi')
    uuids={uuid_by_device[i] for i in devices}
    for row in csv.reader(query('compute-apps=pid,gpu_uuid').splitlines()):
        if not row:continue
        if row[1].strip() in uuids and int(row[0].strip())!=os.getpid():
            raise RuntimeError('foreign GPU owner present before probe: '+row[0].strip())
    inputs=json.loads(args.input_manifest.read_text())
    if set(inputs)!= {'dsa_triton','mla_triton','engine_extension'}:
        raise RuntimeError('input manifest must pin exactly the deployed three engine inputs')
    for entry in inputs.values():
        path=Path(entry['path']).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise RuntimeError('engine input hash mismatch: '+str(path))
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-gpu-probe', action='store_true')
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--devices', default='0,1')
    args = parser.parse_args()
    if not args.run_gpu_probe:
        raise RuntimeError('Explicit guarded GPU-probe flag is required')
    inputs=preimport_guards(args)
    receipt = json.loads(args.build_receipt.read_text())
    validate_native_receipt(receipt)
    path = Path(receipt['extension_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['extension_sha256']:
        raise RuntimeError('Native extension hash changed after compilation')
    if receipt['cuda_initialized']:
        raise RuntimeError('Compilation receipt claims a CUDA context')
    import torch
    from selected_host_q8 import SelectedHostQ8
    from exllamav3.modules.attention_fn import dsa_triton,mla_triton
    from exllamav3.ext import exllamav3_ext as engine_extension
    for name,module in [('dsa_triton',dsa_triton),('mla_triton',mla_triton),
                        ('engine_extension',engine_extension)]:
        if Path(module.__file__).resolve()!=Path(inputs[name]['path']).resolve():
            raise RuntimeError('import origin disagrees with pinned engine input: '+name)
    dsa_attn,mla_kv_quant_append=dsa_triton.dsa_attn,mla_triton.mla_kv_quant_append
    spec = importlib.util.spec_from_file_location(receipt['identity']['name'], path)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    records = []
    device_indices=[int(x) for x in args.devices.split(',')]
    probe_pinned_alias_lifecycle(torch,extension,device_indices,records)
    for device_index in device_indices:
        device = torch.device('cuda',device_index)
        with torch.cuda.device(device):
            physical_pages, logical_pages = 80,64
            rng = torch.Generator(device='cpu').manual_seed(20261009)
            q_cpu = torch.empty((physical_pages,256,128),dtype=torch.int32,pin_memory=True)
            q_cpu.copy_(torch.randint(-(2**31),2**31-1,q_cpu.shape,generator=rng,dtype=torch.int32))
            s_cpu = torch.full((physical_pages,256,16),.003,dtype=torch.float16,pin_memory=True)
            resident_q, resident_s = q_cpu.to(device),s_cpu.to(device)
            table_cpu = torch.randperm(physical_pages,generator=rng)[:logical_pages].to(torch.int32).unsqueeze(0)
            table = table_cpu.to(device)
            epochs = torch.full((physical_pages,),37,dtype=torch.int64,device=device)
            expected = torch.full((logical_pages,),37,dtype=torch.int64,device=device)
            stager = SelectedHostQ8(extension,q_cpu,s_cpu,device_index=device_index,
                max_host_bytes=16*2**20,max_staging_bytes=16*2**20,allow_experimental=True)
            rope=torch.empty((physical_pages,256,1,0),dtype=torch.float16,device=device)
            # Exercise the ACTUAL deployed quantize/scatter path against mapped
            # host aliases, crossing a page boundary and then overwriting rejected
            # speculative rows from position514 after accepting only3 of8.
            for start in (511,514):
                new=(torch.randn((1,8,512),generator=rng)*.05).half().to(device)
                new_rope=torch.empty((1,8,0),dtype=torch.float16,device=device)
                lengths=torch.tensor([start],dtype=torch.int32,device=device)
                mla_kv_quant_append(new,new_rope,resident_q,resident_s,rope,table,lengths,8)
                mla_kv_quant_append(new,new_rope,stager.q_alias,stager.s_alias,rope,table,lengths,8)
                torch.cuda.current_stream(device).synchronize()
                torch.testing.assert_close(resident_q.cpu(),q_cpu,rtol=0,atol=0)
                torch.testing.assert_close(resident_s.cpu(),s_cpu,rtol=0,atol=0)
                records.append(dict(device=device_index,start_position=start,
                    test='deployed_quant_append_host_alias_crosspage_and_rejected_tail_overwrite',passed=True))
            for query_rows in (1,8):
                visible = [16377+r for r in range(query_rows)]
                selected=[]
                for limit in visible:
                    pools=torch.randperm(limit//4,generator=rng)[:512]
                    raw=(pools[:,None]*4+torch.arange(4)).reshape(-1).tolist()
                    tail=list(range(limit-limit%4,limit))
                    row=raw+tail+[-1]*(2080-len(raw)-len(tail))
                    row[7]=row[1]  # genuine duplicate; preserve it in attention
                    selected.append(row)
                indices=torch.tensor(selected,dtype=torch.int32,device=device)
                limits=torch.tensor(visible,dtype=torch.int32,device=device)
                metadata=dict(source_table=table,slot_generations=epochs,
                    expected_generations=expected,row_visible_limits=limits)
                remapped=stager.begin_eager(indices,**metadata)
                torch.cuda.current_stream(device).synchronize()
                if stager.error.item(): raise AssertionError('unexpected guarded stage error')
                remap_cpu=remapped.cpu(); valid=indices.cpu()>=0
                logical=indices.cpu()[valid].long()
                physical=table_cpu[0,logical//256].long()*256+logical%256
                slots=remap_cpu[valid].long()
                torch.testing.assert_close(stager.hot_q.flatten(0,1).cpu()[slots],
                    q_cpu.flatten(0,1)[physical],rtol=0,atol=0)
                torch.testing.assert_close(stager.hot_s.flatten(0,1).cpu()[slots],
                    s_cpu.flatten(0,1)[physical],rtol=0,atol=0)
                query=(torch.randn((64,query_rows,512),generator=rng)*.05).half().to(device)
                query_pe=torch.empty((query_rows,64,0),dtype=torch.float16,device=device)
                kwargs=dict(k_len=2080,scale=256**-.5,page_size=256,
                            q_pe=query_pe,out_latent=True)
                control=dsa_attn(query,resident_q,rope,table,indices=indices,
                    qc=(resident_s,8),pool_len=max(visible),**kwargs)
                candidate=dsa_attn(query,stager.hot_q,stager.rope_dummy,stager.block_table,
                    indices=remapped,qc=(stager.hot_s,8),pool_len=stager.pool_pages*256,**kwargs)
                stager.record_completion()
                torch.cuda.current_stream(device).synchronize()
                stats=stager.finish_eager()
                torch.testing.assert_close(candidate,control,rtol=0,atol=0)
                if stats['unique_rows'] != len(set(logical.tolist())):
                    raise AssertionError('Q8 union was not deduplicated exactly')
                records.append(dict(device=device_index,query_rows=query_rows,
                    test='packed_raw_bytes_and_unchanged_DSA_exact',passed=True,**stats))
                # Same fixed buffers and metadata pointers, no CPU decisions in capture.
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): stager.launch_into(indices,**metadata)
                graph.replay()
                torch.cuda.current_stream(device).synchronize()
                if stager.error.item(): raise AssertionError('captured stage error')
                logical=indices.cpu()[valid].long()
                slots=stager.output_indices[:indices.numel()].view(indices.shape).cpu()[valid].long()
                torch.testing.assert_close(stager.hot_q.flatten(0,1).cpu()[slots],
                    q_cpu.flatten(0,1)[physical],rtol=0,atol=0)
                records.append(dict(device=device_index,query_rows=query_rows,
                    test='fixed_buffer_capture_replay_raw_bytes',passed=True))
                del graph
                stager.begin_eager(indices,**metadata)
                stager.record_completion();stager.cancel()
                torch.cuda.current_stream(device).synchronize()
                try: stager.finish_eager()
                except RuntimeError as error:
                    if 'cancelled=True' not in str(error): raise
                else: raise AssertionError('cancelled step published')
                records.append(dict(device=device_index,query_rows=query_rows,
                    test='cancel_after_enqueue_drains_before_reuse',passed=True))
                # Generation guard fails before downstream attention can see bad rows.
                epochs[table_cpu[0,0].item()]=38
                bad=indices.clone();bad[0,0]=1
                stager.begin_eager(bad,**metadata);stager.record_completion()
                torch.cuda.current_stream(device).synchronize()
                try: stager.finish_eager()
                except RuntimeError as error:
                    if 'error_mask=4' not in str(error): raise
                else: raise AssertionError('stale generation was published')
                epochs.fill_(37)
                records.append(dict(device=device_index,query_rows=query_rows,
                    test='stale_page_generation_rejected',passed=True))
            stager.close()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'passed':True,'records':records,
        'extension_sha256':receipt['extension_sha256'],
        'engine_inputs':inputs,
        'scope':'synthetic cross-device pinned alias reads/writes, allocator reuse observation, owner retention, rejected actual CUDA/unpinned pointers; plus small native staging, unchanged DSA and deployed quant-append crossing pages/rejected-tail overwrite; not full GLM, indexer selection, KDA rollback, session or model lifecycle qualification'},indent=2)+'\n')
    print(json.dumps({'passed':True,'checks':len(records),'output':str(args.output)}))


if __name__=='__main__':main()
