"""Root-owned process launcher. Normal mode has no host-KV imports or hooks."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    argv=sys.argv[1:]
    if '--' not in argv:raise RuntimeError('separate wrapper and unchanged API args with --')
    boundary=argv.index('--')
    parser=argparse.ArgumentParser()
    parser.add_argument('--api',type=Path,required=True)
    parser.add_argument('--mode',choices=['normal','active-host','combined-memory'],default='normal')
    parser.add_argument('--build-receipt',type=Path)
    parser.add_argument('--threshold',type=int,default=131072)
    parser.add_argument('--host-budget-bytes',type=int,default=3*2**30)
    parser.add_argument('--activation-mode',choices=['context_threshold','context_and_pressure'],default='context_threshold')
    parser.add_argument('--pressure-free-bytes',type=int,default=0)
    parser.add_argument('--no-prefix-cache',action='store_true')
    parser.add_argument('--no-session-cache',action='store_true')
    args=parser.parse_args(argv[:boundary])
    forwarded=argv[boundary+1:]
    api=args.api.resolve(strict=True)
    sys.path.insert(0,str(api.parent))
    if args.mode in ('active-host','combined-memory'):
        combined=args.mode=='combined-memory'
        if not args.no_prefix_cache or (not combined and not args.no_session_cache):
            raise RuntimeError('active-host requires explicit --no-prefix-cache --no-session-cache')
        flags={arg.split('=',1)[0] for arg in forwarded}
        if '--dflash-prefix-cache' in flags or (not combined and '--dflash-session-cache' in flags):
            raise RuntimeError('prefix/session caching cannot coexist with this active-host trial')
        if combined and (args.no_session_cache or '--dflash-session-cache' not in flags
                         or '--target-cpu-cache-gib' not in flags):
            raise RuntimeError('combined-memory requires explicit session and positive target tier CLI')
        target=None
        for i,arg in enumerate(forwarded):
            if arg.split('=',1)[0]=='--target-cpu-cache-gib':
                if target is not None:raise ValueError('duplicate target CPU tier argument')
                if '=' not in arg and i+1>=len(forwarded):raise ValueError('target CPU tier value missing')
                value=arg.split('=',1)[1] if '=' in arg else forwarded[i+1]
                target=int(value)
                if not combined and target!=0:
                    raise RuntimeError('general target CPU tier cannot coexist with this trial')
        if combined and (target is None or target<=0):
            raise RuntimeError('combined-memory target tier budget must be positive')
        if combined and os.environ.get('GLM53_CPU_DUPLICATE_RECYCLE')!='1':
            raise RuntimeError('combined-memory requires GLM53_CPU_DUPLICATE_RECYCLE=1')
        if os.environ.get('EXL3_BC_ATTN')!='0':
            raise RuntimeError('active-host requires EXL3_BC_ATTN=0 before imports')
        if args.build_receipt is None:raise RuntimeError('active-host requires a pinned native receipt')
        receipt=json.loads(args.build_receipt.read_text())
        from validate_gpu import validate_native_receipt
        validate_native_receipt(receipt)
        path=Path(receipt['extension_path']).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['extension_sha256']:
            raise RuntimeError('native extension hash changed')
        import torch
        if torch.cuda.is_initialized():raise RuntimeError('trial launcher must precede model initialization')
        spec=importlib.util.spec_from_file_location(receipt['identity']['name'],path)
        extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)
        from engine_adapter import TrialPolicy,install_constructor_hook,install_combined_constructor_hook
        policy=TrialPolicy(enabled=True,context_threshold=args.threshold,
                           host_budget_bytes=args.host_budget_bytes,
                           activation_mode=args.activation_mode,
                           pressure_free_bytes=args.pressure_free_bytes)
        policy.validate()
        hook=install_combined_constructor_hook if combined else install_constructor_hook
        hook(extension,policy=policy)
    sys.argv=[str(api),*forwarded]
    runpy.run_path(str(api),run_name='__main__')


if __name__=='__main__':main()
