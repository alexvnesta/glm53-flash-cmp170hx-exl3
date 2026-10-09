"""Explicit LS session/active-layout custody. No Torch or engine import at import.

Page/hash identities and recurrent checkpoint cookies survive a latent layout
epoch. Raw CPUPageCache copies do not: they require real restored GPU tensors
and a fresh, authorized segment binding on the original worker/streams.
"""
import threading
import types


class CombinedMemoryCoordinator:
    def __init__(self, generator, manager, tier):
        from glm_dflash_sessions import MultiSessionCache
        from glm_target_cpu_cache import LazyTargetCPUPageCache
        if (not isinstance(manager, MultiSessionCache)
                or not isinstance(tier, LazyTargetCPUPageCache)
                or generator._glm_dflash_prefix_cache is not manager
                or manager.generator is not generator
                or generator.cpu_page_cache is not tier
                or generator.pagetable.cpu_tier is not tier
                or tier._generator_owner is not generator
                or not tier._prefer_referenced_duplicates
                or generator.enable_defrag or generator.model.loaded_tp
                or generator.max_batch_size != 1 or generator.num_remaining_jobs()
                or getattr(generator, '_glm_combined_memory_coordinator', None) is not None):
            raise ValueError('Fresh owned LS session manager and duplicate tier required')
        self.generator, self.manager, self.tier = generator, manager, tier
        self.cache, self.model, self.pagetable = generator.cache, generator.model, generator.pagetable
        self.owner_serial = getattr(self.cache, 'owner_serial', None)
        self.pages = self.pagetable.all_pages
        self.page_ids = tuple(id(page) for page in self.pages)
        self.max_pages = self.pagetable.max_pages
        self.adapter = None
        self.phase = 'GPU'
        self.worker_thread = None
        self.forward_depth = 0
        self.epoch = 0
        self.installed = False
        self.hooks = []
        self._expected_gpu_ids = None
        self._before_gpu_ids = None
        self._nonlatent = None
        self._status = {'mode': 'combined_ls', 'layout': 'GPU', 'layout_epoch': 0,
                        'GPU_indexer_unchanged': True}
        tier.bind_layout_coordinator(self)
        generator._glm_combined_memory_coordinator = self

    def stats(self):
        # Replace immutable snapshots on the worker. Health must not acquire a
        # coordinator lock while holding the session-manager lock.
        return dict(self._status)

    def _publish(self, **extra):
        self._status = {**self._status, 'layout': self.phase, 'layout_epoch': self.epoch, **extra}

    def _owner(self):
        g = self.generator
        if (g.cache is not self.cache or g.model is not self.model or g.pagetable is not self.pagetable
                or g._glm_dflash_prefix_cache is not self.manager
                or g.cpu_page_cache is not self.tier or self.pagetable.cpu_tier is not self.tier
                or self.pagetable.generator is not g or self.pagetable.cache is not self.cache
                or getattr(self.cache, 'owner_serial', None) != self.owner_serial
                or g.model.loaded_tp or g.max_batch_size != 1 or g.enable_defrag
                or self.pagetable.all_pages is not self.pages
                or self.pagetable.max_pages != self.max_pages
                or tuple(id(page) for page in self.pages) != self.page_ids
                or getattr(g, '_glm_combined_memory_coordinator', None) is not self):
            raise RuntimeError('Combined cache/page/session ownership changed')
        if self.phase == 'FAILED':
            raise RuntimeError('Combined memory layout is failed; process restart required')

    def _worker(self):
        current = threading.get_ident()
        if self.worker_thread is None:
            self.worker_thread = current
        elif self.worker_thread != current:
            raise RuntimeError('Combined GPU lifecycle left its persistent worker thread')

    def validate_for_adapter(self, generator):
        self._owner()
        if generator is not self.generator or self.adapter is not None or self.phase != 'GPU':
            raise ValueError('One fresh combined active adapter required')

    def bind_adapter(self, adapter):
        self.validate_for_adapter(adapter.generator)
        self.adapter = adapter
        policy = getattr(adapter, 'policy', None)
        if policy is not None:
            self._publish(activation_mode=policy.activation_mode,
                          minimum_context=policy.context_threshold,
                          pressure_free_bytes=policy.pressure_free_bytes,
                          active_host_pin_request_budget_bytes=policy.host_budget_bytes)

    def ensure_gpu(self, reason):
        self._owner()
        if self.phase == 'GPU':
            return
        self._worker()
        if self.forward_depth:
            raise RuntimeError('Cannot restore latent layout during target forward: ' + reason)
        if self.phase != 'HOST_ACTIVE' or self.adapter is None:
            raise RuntimeError('Incomplete layout transaction before ' + reason)
        try:
            self.adapter.deactivate()
        except BaseException:
            if self.phase != 'GPU':
                self.phase = 'FAILED'
                self._publish(failed_reason=reason)
            raise
        self._owner()
        if self.phase != 'GPU' or self.tier._layout_suspended:
            raise RuntimeError('Return did not restore transferable GPU segments')

    def prepare_host_commit(self):
        self._owner(); self._worker()
        if self.phase != 'GPU' or self.forward_depth:
            raise RuntimeError('Migration must follow prefill outside target forward')
        self._before_gpu_ids = tuple(id(t) for t in self.cache.get_all_tensors() if t is not None)
        latent = {id(t) for layer in self.adapter.layers.values() for t in (layer.qk, layer.sk)}
        self._nonlatent = tuple((i, id(t), t.data_ptr()) for i, t in enumerate(
            t for t in self.cache.get_all_tensors() if t is not None) if id(t) not in latent)
        self.tier.suspend_gpu_segments(self)
        self.phase = 'HOST_PREPARED'
        self._publish()

    def host_committed(self):
        self._owner(); self._worker()
        if self.phase != 'HOST_PREPARED' or not self.tier._layout_suspended:
            raise RuntimeError('Host commit lacks suspended tier custody')
        for state in self.adapter.states.values():
            if state.layer.qk is not state.stager.q_alias or state.layer.sk is not state.stager.s_alias:
                raise RuntimeError('Host alias commit incomplete')
        self.epoch += 1
        self.phase = 'HOST_ACTIVE'
        self._publish(host_payload_bytes=self.adapter.host_bytes,
                      host_pin_request_bytes=self.adapter.host_pin_request_bytes,
                      target_transfers_suspended=True)

    def abort_host_commit(self):
        self._expected_gpu_ids = self._before_gpu_ids
        self.phase = 'GPU_REBIND'
        self.tier.resume_gpu_segments(self)
        self._expected_gpu_ids = self._before_gpu_ids = None
        self.phase = 'GPU'
        self._publish(target_transfers_suspended=False)

    def prepare_gpu_commit(self, replacements):
        self._owner(); self._worker()
        if self.phase != 'HOST_ACTIVE' or self.forward_depth or not self.tier._layout_suspended:
            raise RuntimeError('GPU return lacks exclusive host layout custody')
        mapping = {}
        for key, (q, s) in replacements.items():
            state = self.adapter.states[key]
            for old, new in ((state.stager.q_alias, q), (state.stager.s_alias, s)):
                if (new.device != state.stager.device or tuple(new.shape) != tuple(old.shape)
                        or new.dtype != old.dtype):
                    raise RuntimeError('Restoration tensor geometry/device changed')
                mapping[id(old)] = id(new)
        tensors = tuple(t for t in self.cache.get_all_tensors() if t is not None)
        if len(mapping) != 2 * len(self.adapter.layers):
            raise RuntimeError('All eleven latent/scales replacements required')
        for index, identity, pointer in self._nonlatent:
            if id(tensors[index]) != identity or tensors[index].data_ptr() != pointer:
                raise RuntimeError('Indexer/non-latent storage changed during migration')
        self._expected_gpu_ids = tuple(mapping.get(id(t), id(t)) for t in tensors)
        self.phase = 'GPU_REBIND'
        self._publish()

    def validate_gpu_rebind(self, tensors):
        self._owner(); self._worker()
        if self.phase != 'GPU_REBIND' or tuple(id(t) for t in tensors) != self._expected_gpu_ids:
            raise RuntimeError('GPU segment rebind did not use authorized restored tensors')

    def gpu_committed(self):
        self.tier.resume_gpu_segments(self)
        self._expected_gpu_ids = self._before_gpu_ids = None
        self.phase = 'GPU'
        self._publish(target_transfers_suspended=False)

    @staticmethod
    def checkpoint_needed(job, interval=None):
        seq = job.sequences[0]
        return bool(seq.kv_position and seq.kv_position % 256 == 0
                    and job.last_recurrent_checkpoint_pos != seq.kv_position
                    and job.is_checkpoint_boundary(interval))

    def install_lifecycle(self):
        if self.installed or self.adapter is None:
            raise RuntimeError('Bind one active adapter before combined lifecycle installation')
        try:
            self._install_lifecycle()
        except BaseException:
            self._remove_hooks()
            raise

    def _install_lifecycle(self):
        from exllamav3.generator.job import Job
        coordinator = self
        for name in ('allocate_pages', 'deallocate_pages', 'prefill', 'maybe_stash_recurrent'):
            original = getattr(Job, name)
            def hook(job, *args, _name=name, _original=original, **kwargs):
                if job.generator is coordinator.generator:
                    needed = _name != 'prefill' or not job.sequences[0].prefill_complete
                    if _name == 'maybe_stash_recurrent':
                        interval = kwargs.get('interval', args[1] if len(args) > 1 else None)
                        needed = coordinator.checkpoint_needed(job, interval)
                    if needed:
                        coordinator.ensure_gpu('job_' + _name)
                return _original(job, *args, **kwargs)
            setattr(Job, name, hook)
            self.hooks.append((Job, name, original, hook))
        # Raw tier operations may only use a stable GPU layout. This extra gate
        # catches a new native call site even if its Job wrapper was bypassed.
        for name in ('store', 'fetch', '_evict_one'):
            original = getattr(self.tier, name)
            def tier_hook(instance, *args, _original=original, _name=name, **kwargs):
                coordinator.ensure_gpu('tier_' + _name)
                return _original(*args, **kwargs)
            installed = types.MethodType(tier_hook, self.tier)
            setattr(self.tier, name, installed)
            self.hooks.append((self.tier, name, original, installed))
        forward = self.model.forward
        def forward_hook(instance, *args, **kwargs):
            coordinator._owner(); coordinator._worker()
            if coordinator.phase not in ('GPU', 'HOST_ACTIVE'):
                raise RuntimeError('Target forward during a layout transition')
            coordinator.forward_depth += 1
            try:
                return forward(*args, **kwargs)
            finally:
                coordinator.forward_depth -= 1
        installed = types.MethodType(forward_hook, self.model)
        self.model.forward = installed
        self.hooks.append((self.model, 'forward', forward, installed))
        # clear_queue deallocates pages before its idle callback. Restore first.
        original = self.generator.clear_queue
        def clear_hook(instance, *args, **kwargs):
            coordinator.ensure_gpu('clear_queue')
            return original(*args, **kwargs)
        installed = types.MethodType(clear_hook, self.generator)
        self.generator.clear_queue = installed
        self.hooks.append((self.generator, 'clear_queue', original, installed))
        self.installed = True

    def _remove_hooks(self):
        for owner, name, original, installed in reversed(self.hooks):
            if getattr(owner, name) is not installed:
                raise RuntimeError('Combined lifecycle hook ownership changed')
            setattr(owner, name, original)
        self.hooks.clear()
        self.installed = False

    def close(self):
        self.ensure_gpu('coordinator_close')
        self._remove_hooks()
        if self.tier._layout_coordinator is not self:
            raise RuntimeError('Combined tier owner changed')
        self.tier._layout_coordinator = None
        if getattr(self.generator, '_glm_combined_memory_coordinator', None) is self:
            del self.generator._glm_combined_memory_coordinator
        self.phase = 'CLOSED'
        self._publish()
