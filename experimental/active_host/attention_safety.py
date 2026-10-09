"""Stdlib-only full-attention safety gate for the exact stock16a MLA route."""
import hashlib
import os
from pathlib import Path

SOURCE_PINS={
    'mla_attn':'90eed44b790f4810ed1eeddb70cb4e4bdd38a82b54d43fcf7208148d4d250d10',
    'bc_attn':'561ed1de5d76cff176c84afa7a26fa35697a174ae0d0d1b30ae2c6863cc7ee13',
    'bc_mla':'eb2f7121bcac2cfd6cdda2f4a3eda8977f955201434e92065088fc45f5e22350',
}


def target_mla_modules(model,attention_class):
    """Walk the stock Module.modules tree once; never follow the drafter model."""
    pending=list(model.modules);seen=set();result=[]
    while pending:
        module=pending.pop()
        if id(module) in seen:continue
        seen.add(id(module))
        if isinstance(module,attention_class):result.append(module)
        pending.extend(getattr(module,'modules',()))
    if len(result)!=11:
        raise ValueError('Expected exactly eleven unique target MLA modules')
    return tuple(result)


def require_dispatch_attention(engine_modules,target_modules):
    """Reject full MLA captured cache owners, permit native Linear projections.

    _bc_mla_enable is EXL3_BC_MLA's independent switch, defaultTrue. Actual
    builder eligibility additionally requires bc_mla.bc_attn_enable. The exact
    pinned builder returnsNone with that flagFalse before constructing BCMLA.
    """
    if os.environ.get('EXL3_BC_ATTN')!='0':
        raise ValueError('EXL3_BC_ATTN=0 must explicitly disable full attention')
    if set(engine_modules)!=set(SOURCE_PINS):raise ValueError('All three stock attention modules must be pinned')
    for name,module in engine_modules.items():
        path=getattr(module,'__file__',None)
        if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=SOURCE_PINS[name]:
            raise ValueError('Active-host requires pinned stock attention source: '+name)
    if (getattr(engine_modules['bc_attn'],'bc_attn_enable',None) is not False or
            getattr(engine_modules['bc_mla'],'bc_attn_enable',None) is not False):
        raise ValueError('Loaded full-attention enable snapshots must both be False')
    if type(getattr(engine_modules['mla_attn'],'_bc_mla_enable',None)) is not bool:
        raise ValueError('Independent MLA enable switch must retain its stock bool contract')
    if len(target_modules)!=11:raise ValueError('Exactly eleven target MLA owners required')
    for module in target_modules:
        for key,value in module.dispatch_cache.items():
            if isinstance(key,tuple) and key and key[0]=='bcm' and value is not None and value is not False:
                raise ValueError('Already-cached full BCMLA owner must not retain/replay latent cache')
