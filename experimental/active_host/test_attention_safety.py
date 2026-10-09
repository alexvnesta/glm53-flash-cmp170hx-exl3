"""Verify real pinned flag/decline semantics with stdlib stand-ins only."""
import ast
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace as NS
import unittest
from unittest.mock import patch

from attention_safety import require_dispatch_attention,target_mla_modules

from engine_inputs import source_paths
ENGINE_PATHS = source_paths(Path(os.environ['EXLLAMAV3_ENGINE_ROOT']))


class MLA:
    def __init__(self):
        self.modules=[];self.dispatch_cache={('projection',1):object()}
        self.q_proj=NS(inner=NS(bc=object()))
        self.idx_wq_b=NS(inner=NS(bc=object()))


def engine_modules():
    return {'mla_attn':NS(__file__=str(ENGINE_PATHS['mla_attn']),_bc_mla_enable=True),
            'bc_attn':NS(__file__=str(ENGINE_PATHS['bc_attn']),bc_attn_enable=False),
            'bc_mla':NS(__file__=str(ENGINE_PATHS['bc_mla']),bc_attn_enable=False)}


class AttentionSafety(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'EXL3_BC_ATTN':'0'});self.env.start();self.addCleanup(self.env.stop)
        self.modules=tuple(MLA() for _ in range(11));self.engine=engine_modules()

    def test_bc_mla_default_true_does_not_enable_full_attention_with_bc0(self):
        require_dispatch_attention(self.engine,self.modules)

    def test_independent_mla_disable_also_safe(self):
        self.engine['mla_attn']._bc_mla_enable=False
        require_dispatch_attention(self.engine,self.modules)

    def test_stale_loaded_full_attention_snapshot_refused(self):
        for name in ('bc_attn','bc_mla'):
            current=engine_modules();current[name].bc_attn_enable=True
            with self.assertRaisesRegex(ValueError,'snapshots'):require_dispatch_attention(current,self.modules)

    def test_missing_loaded_snapshot_not_assumed_false(self):
        self.engine['bc_mla'].bc_attn_enable=None
        with self.assertRaisesRegex(ValueError,'snapshots'):require_dispatch_attention(self.engine,self.modules)

    def test_cached_full_owner_refused_even_when_loaded_flags_false(self):
        self.modules[0].dispatch_cache[('bcm',123)]=object()
        with self.assertRaisesRegex(ValueError,'cached full'):require_dispatch_attention(self.engine,self.modules)

    def test_declined_full_owners_and_projection_handles_allowed(self):
        self.modules[0].dispatch_cache[('bcm',123)]=False
        self.modules[1].dispatch_cache[('bcm',456)]=None
        require_dispatch_attention(self.engine,self.modules)

    def test_source_drift_and_environment_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'wrong.py';path.write_text('changed')
            self.engine['mla_attn'].__file__=str(path)
            with self.assertRaisesRegex(ValueError,'pinned stock'):require_dispatch_attention(self.engine,self.modules)
        os.environ['EXL3_BC_ATTN']='1'
        with self.assertRaisesRegex(ValueError,'explicitly disable'):require_dispatch_attention(engine_modules(),self.modules)

    def test_tree_deduplicates_target_and_never_follows_drafter(self):
        model=NS(modules=[NS(modules=list(self.modules)),self.modules[0]],
                 draft_model=NS(modules=[MLA()]))
        self.assertEqual({id(m) for m in target_mla_modules(model,MLA)},set(map(id,self.modules)))
        with self.assertRaisesRegex(ValueError,'eleven'):target_mla_modules(NS(modules=list(self.modules[:10])),MLA)

    def test_exact_stock_builder_declines_before_bc_owner_constructor(self):
        # Execute the ACTUAL build_bc_mla function. Relative import stand-ins and
        # torch.device label parsing do not import Torch or allocate anything.
        tree=ast.parse((ENGINE_PATHS['bc_mla']).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_bc_mla')
        traces=[]
        def never_construct(*a):raise AssertionError('Full graph owner constructed under BC0')
        ns={'__package__':'exllamav3.modules.attention_fn','torch':NS(device=lambda d:d),
            'bc_attn_enable':False,'BCMLA':never_construct,
            '_trace_build':lambda m,result,kind:traces.append((result,kind))}
        cache=ModuleType('exllamav3.cache.mla');cache.CacheLayer_MLA_fp16=type('FP16',(),{});cache.CacheLayer_MLA_quant=type('Q8',(),{})
        rope=ModuleType('exllamav3.util.rope');rope.RopeStyle=NS(NONE=0)
        code=ast.Module(body=[node],type_ignores=[])
        with patch.dict(sys.modules,{cache.__name__:cache,rope.__name__:rope}):
            exec(compile(ast.fix_missing_locations(code),'pinned_bc_mla.py','exec'),ns)
            module=NS(num_q_heads=64,kv_lora_rank=512,qk_rope_head_dim=0,v_head_dim=128,device='cuda:0')
            self.assertIsNone(ns['build_bc_mla'](module,object()))
        self.assertEqual(traces,[(None,'mla')])


if __name__=='__main__':unittest.main(verbosity=2)
