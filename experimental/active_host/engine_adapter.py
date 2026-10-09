"""Disabled-by-default, decode-only GLM Q8 LS active-host trial adapter.

Actual engine hooks are present, but uncompiled/unrun here. This first adapter
keeps ordinary GPU prefill and capacity. It migrates whole latent arenas only
at a completed-prefill generation boundary, restores them before subsequent
prefill/idle/defrag/cancel, and leaves GPU indexers/KDA/drafter unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import threading
import types


@dataclass(frozen=True)
class TrialPolicy:
    enabled: bool = False
    context_threshold: int = 131072
    host_budget_bytes: int = 3 * 2**30
    per_layer_staging_bytes: int = 16 * 2**20

    def validate(self):
        if self.context_threshold <= 2048 or self.context_threshold % 256:
            raise ValueError("explicit sparse context threshold must be >2048 and256 aligned")
        if self.host_budget_bytes <= 0 or self.per_layer_staging_bytes <= 0:
            raise ValueError("positive bounded host and staging budgets required")


@dataclass
class LayerState:
    layer: object
    stager: object
    slot_epochs: object
    expected_epochs: object
    row_limits: object
    row_offsets: object


class ActiveHostTrialAdapter:
    def __init__(self, generator, extension, *, policy=TrialPolicy()):
        policy.validate()
        if not policy.enabled:
            raise RuntimeError("active host trial is disabled by default")
        # Validate topology/lifecycle BEFORE importing Torch or allocating memory.
        if (generator.max_batch_size != 1 or generator.model.loaded_tp or
                generator.cpu_page_cache is not None or
                getattr(generator.pagetable,'cpu_tier',None) is not None or
                getattr(generator,"_glm_dflash_prefix_cache",None) is not None):
            raise ValueError("first trial requires LS batch1, no CPUPageCache or session/prefix manager")
        if os.environ.get('EXL3_BC_ATTN') != '0':
            raise ValueError("first engine adapter requires explicitly disabled BC attention")
        import torch
        from exllamav3.modules import mla_attn
        from exllamav3.modules.attention_fn import bc_attn,bc_mla
        from attention_safety import target_mla_modules,require_dispatch_attention
        MLAttention=mla_attn.MLAttention
        self.attention_engine_modules={'mla_attn':mla_attn,'bc_attn':bc_attn,'bc_mla':bc_mla}
        self.target_attention_modules=target_mla_modules(generator.model,MLAttention)
        require_dispatch_attention(self.attention_engine_modules,self.target_attention_modules)
        from exllamav3.cache import CacheLayer_MLA_quant
        layers=generator.cache.layers
        if len(layers)!=11 or generator.cache.max_history!=7:
            raise ValueError("expected GLM eleven MLA layers and DFlashK7 rollback capacity")
        if (not generator.dflash_draft or generator.num_draft_tokens!=7 or
                getattr(generator.draft_cache,'dflash_ring_tokens',0)!=8192):
            raise ValueError("this adapter only qualifies the current DFlashK7/8192 ring")
        for layer in layers.values():
            if (not isinstance(layer,CacheLayer_MLA_quant) or layer.bits!=8 or
                    layer.kv_lora_rank!=512 or layer.qk_rope_head_dim!=0 or
                    layer.k_idx is None or layer.k_pool is None or
                    tuple(layer.qk.shape[1:])!=(256,128) or
                    tuple(layer.sk.shape[1:])!=(256,16)):
                raise ValueError("only current Q8 NoPE full-indexer geometry is supported")
        required=sum(l.qk.numel()*4+l.sk.numel()*2 for l in layers.values())
        if required>policy.host_budget_bytes:
            raise ValueError("whole-arena migration exceeds aggregate host budget")
        self.torch,self.attention_class=torch,MLAttention
        self.generator,self.extension,self.policy=generator,extension,policy
        self.host_bytes,self.layers=required,dict(layers)
        self.states={}
        self.active=False
        self.epoch=0
        self.lock=threading.RLock()
        self.installed=False
        self.originals={}

    def _outside_capture(self):
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError("whole-engine active-host capture is not supported by this first trial")

    def _check_attention_safety(self):
        from attention_safety import require_dispatch_attention
        require_dispatch_attention(self.attention_engine_modules,self.target_attention_modules)

    def _drain(self):
        self._outside_capture()
        records=[]
        errors=[]
        from selected_host_q8 import UnpublishedStepError
        for state in self.states.values():
            event=state.stager._event
            if event is not None:
                event.synchronize()
                try:records.append(state.stager.finish_eager())
                except UnpublishedStepError as error:errors.append(error)
        if errors:raise errors[0]
        return records

    def activate(self):
        """Normal GPU prefill already finished. Commit all aliases atomically.

        No page defrag/eviction or session switch may occur while aliases are live.
        Current target page references and physical mapping remain unchanged.
        """
        with self.lock:
            self._outside_capture()
            if self.active:return
            self._check_attention_safety()
            g=self.generator
            if (getattr(g,'_glm_dflash_prefix_cache',None) is not None or
                    g.cpu_page_cache is not None or
                    getattr(g.pagetable,'cpu_tier',None) is not None):
                raise RuntimeError("ownership manager was enabled after adapter installation")
            if len(g.active_jobs)!=1 or g.pending_jobs:
                raise RuntimeError("migration requires one active request and no pending job")
            job=g.active_jobs[0]
            if (len(job.sequences)!=1 or not job.sequences[0].prefill_complete or
                    job.is_requeued or job.orig_max_rq_tokens is not None):
                raise RuntimeError("migration requires completed prefill and no requeue")
            if job.sequences[0].kv_position<self.policy.context_threshold:
                return
            from selected_host_q8 import SelectedHostQ8
            prepared={}
            self.epoch+=1
            before={str(i):self.torch.cuda.memory_allocated(i) for i in (0,1)}
            # Arena allocation and alias construction are outside CUDA graphs.
            # Source tensors stay GPU resident until all eleven copies succeed.
            try:
                for key,layer in self.layers.items():
                    device=layer.qk.device
                    with self.torch.cuda.device(device):
                        self.torch.cuda.current_stream(device).synchronize()
                        host_q=self.torch.empty(layer.qk.shape,dtype=self.torch.int32,
                                                device='cpu',pin_memory=True)
                        host_s=self.torch.empty(layer.sk.shape,dtype=self.torch.float16,
                                                device='cpu',pin_memory=True)
                        host_q.copy_(layer.qk)
                        host_s.copy_(layer.sk)
                        stager=SelectedHostQ8(self.extension,host_q,host_s,
                            device_index=device.index,
                            max_host_bytes=host_q.numel()*4+host_s.numel()*2,
                            max_staging_bytes=self.policy.per_layer_staging_bytes,
                            allow_experimental=True)
                        pages=layer.qk.shape[0]
                        epochs=self.torch.full((pages,),self.epoch,dtype=self.torch.int64,device=device)
                        expected=self.torch.full((pages,),self.epoch,dtype=self.torch.int64,device=device)
                        limits=self.torch.empty(8,dtype=self.torch.int32,device=device)
                        offsets=self.torch.arange(1,9,dtype=self.torch.int32,device=device)
                        prepared[key]=LayerState(layer,stager,epochs,expected,limits,offsets)
            except BaseException:
                # No layer/cache field was changed; current GPU prefill data survive.
                prepared.clear()
                raise
            # Plain field swaps only; all mapped host storage already exists.
            for state in prepared.values():
                state.layer.qk,state.layer.sk=state.stager.q_alias,state.stager.s_alias
            self.states=prepared
            self.active=True
            after={str(i):self.torch.cuda.memory_allocated(i) for i in (0,1)}
            print(json.dumps({'event':'active_host_migrated','layout_epoch':self.epoch,
                'host_bytes':self.host_bytes,'GPU_latent_bytes_replaced':self.host_bytes,
                'allocated_before':before,'allocated_after':after,
                'actual_allocated_reduction_bytes':sum(before.values())-sum(after.values()),
                'GPU_indexer_unchanged':True,'capacity_tokens':g.cache.max_num_tokens,
                'scope':'decode-only trial; GPU initial allocation still bounds capacity'},sort_keys=True),flush=True)

    def deactivate(self):
        """Restore ordinary GPU tensors before PP/page ownership changes.

        On allocation/copy failure, do not partially publish GPU replacements.
        The trial fails and its root controller must restore the production
        process; it cannot safely continue a new prefill on half-restored state.
        """
        with self.lock:
            if not self.active:return
            self._outside_capture()
            drain_error=None
            from selected_host_q8 import UnpublishedStepError
            try:self._drain()
            except UnpublishedStepError as error:drain_error=error
            replacements={}
            for key,state in self.states.items():
                device=state.stager.device
                with self.torch.cuda.device(device):
                    q=self.torch.empty(state.layer.qshape,dtype=self.torch.int32,device=device)
                    s=self.torch.empty(state.layer.sshape,dtype=self.torch.float16,device=device)
                    # CPU pinned source copy, not cudaMemcpy on an alias pretending
                    # to be device memory. All host writes have been drained above.
                    q.copy_(state.stager.host_q)
                    s.copy_(state.stager.host_s)
                    replacements[key]=(q,s)
            for key,(q,s) in replacements.items():
                self.states[key].layer.qk,self.states[key].layer.sk=q,s
            for state in self.states.values():state.stager.close()
            self.states.clear()
            self.active=False
            print(json.dumps({'event':'active_host_returned_to_GPU',
                'layout_epoch':self.epoch,'restored_bytes':self.host_bytes},sort_keys=True),flush=True)
            if drain_error is not None:raise drain_error

    def sparse(self, original, module, q_lat, q_pe, bsz, seqlen, params,
               ckv_cache,kpe_cache,block_table,indices,qc,pool_len=0):
        if not self.active or params.get('cache') is not self.generator.cache:
            return original(module,q_lat,q_pe,bsz,seqlen,params,ckv_cache,kpe_cache,
                            block_table,indices,qc,pool_len)
        self._outside_capture()
        if bsz!=1 or not 1<=seqlen<=8 or qc is None or qc[1]!=8:
            raise RuntimeError("active host adapter received a prefill/unsupported shape")
        key=(module.layer_idx,params.get('layer_instance') or 0)
        state=self.states[key]
        host_lengths=params.get('_mla_host_seqlens')
        if not host_lengths or len(host_lengths)!=1 or host_lengths[0]+seqlen!=pool_len:
            raise RuntimeError("explicit host causal bounds unavailable")
        limits=state.row_limits[:seqlen]
        limits.copy_(state.row_offsets[:seqlen]);limits.add_(host_lengths[0])
        remapped=state.stager.begin_eager(indices,
            source_table=block_table,slot_generations=state.slot_epochs,
            expected_generations=state.expected_epochs[:block_table.shape[1]],
            row_visible_limits=limits)
        # First integrated trial is deliberately eager and checks before DSA.
        # A flagged empty selection must not propagate NaNs into the MoE router.
        self.torch.cuda.current_stream(q_lat.device).synchronize()
        error_mask=int(state.stager.error.item())
        if error_mask:
            state.stager.record_completion()
            raise RuntimeError(f"active host gather rejected source: {error_mask}")
        try:
            result=original(module,q_lat,q_pe,bsz,seqlen,params,
                state.stager.hot_q,state.stager.rope_dummy,state.stager.block_table,
                remapped,(state.stager.hot_s,8),state.stager.pool_pages*256)
        finally:
            state.stager.record_completion()
        return result

    def install(self):
        with self.lock:
            if self.installed:raise RuntimeError("adapter already installed")
            if getattr(self.attention_class,'_glm_active_host_owner',None) is not None:
                raise RuntimeError("one active-host owner per process is required")
            g=self.generator
            originals={name:getattr(g,name) for name in
                ('iterate_gen','enqueue','cancel','on_queue_drained')}
            attention_original=self.attention_class._attend_sparse
            adapter=self

            def attention_hook(module,*args,**kwargs):
                return adapter.sparse(attention_original,module,*args,**kwargs)

            def generation_hook(instance,*args,**kwargs):
                with adapter.lock:
                    if (not adapter.active and len(instance.active_jobs)==1 and
                            instance.active_jobs[0].sequences[0].prefill_complete and
                            instance.active_jobs[0].sequences[0].kv_position>=adapter.policy.context_threshold):
                        adapter.activate()
                    try:
                        result=originals['iterate_gen'](*args,**kwargs)
                        adapter._drain()
                        return result
                    except BaseException:
                        adapter.deactivate()
                        raise

            def enqueue_hook(instance,*args,**kwargs):
                with adapter.lock:
                    if instance.num_remaining_jobs():
                        raise RuntimeError("active host trial requires serialized request admission")
                    if args and isinstance(args[0],list) and len(args[0])!=1:
                        raise RuntimeError("multiple-job enqueue is unsupported")
                    adapter.deactivate()
                    return originals['enqueue'](*args,**kwargs)

            def cancel_hook(instance,*args,**kwargs):
                with adapter.lock:
                    adapter.deactivate()
                    return originals['cancel'](*args,**kwargs)

            def idle_hook(instance,*args,**kwargs):
                with adapter.lock:
                    adapter.deactivate()
                    return originals['on_queue_drained'](*args,**kwargs)

            self.attention_class._attend_sparse=attention_hook
            self.attention_class._glm_active_host_owner=self
            for name,fn in [('iterate_gen',generation_hook),('enqueue',enqueue_hook),
                            ('cancel',cancel_hook),('on_queue_drained',idle_hook)]:
                setattr(g,name,types.MethodType(fn,g))
            self.originals=originals|{'attention':attention_original}
            self.installed=True
            g._glm_active_host_trial=self
            return self

    def close(self):
        with self.lock:
            self.deactivate()
            if self.installed:
                if getattr(self.attention_class,'_glm_active_host_owner',None) is not self:
                    raise RuntimeError("attention hook ownership changed")
                for name in ('iterate_gen','enqueue','cancel','on_queue_drained'):
                    setattr(self.generator,name,self.originals[name])
                self.attention_class._attend_sparse=self.originals['attention']
                self.attention_class._glm_active_host_owner=None
                self.installed=False


def install_constructor_hook(extension, *, policy):
    """Trial launcher calls this before runpy API, after setting BC0/prefix-off.

    The API constructs its generator only after normal model/cache GPU load.
    No installed API or engine file is changed.
    """
    if not policy.enabled:raise RuntimeError("explicit trial opt-in required")
    import glm_async
    original=glm_async.responsive_generator_class
    def factory(base):
        original_cls=original(base)
        class TrialGenerator(original_cls):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self._active_host_adapter=ActiveHostTrialAdapter(
                    self.generator,extension,policy=policy).install()
            async def close(self):
                import asyncio
                async with self._step_lock:
                    await asyncio.get_running_loop().run_in_executor(
                        self._gpu_worker,self._active_host_adapter.close)
                await super().close()
        return TrialGenerator
    glm_async.responsive_generator_class=factory
