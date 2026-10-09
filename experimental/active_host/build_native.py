"""Root-owned CPU compilation harness. Does not import ExLlama or initialize CUDA."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-directory', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if os.environ.get('TORCH_CUDA_ARCH_LIST') != '8.0':
        raise RuntimeError('Set TORCH_CUDA_ARCH_LIST=8.0 explicitly; no GPU architecture discovery')
    os.environ.setdefault('MAX_JOBS','4')
    import torch
    from torch.utils.cpp_extension import load, CUDA_HOME
    if torch.cuda.is_initialized():
        raise RuntimeError('Compilation process already has a CUDA context')
    root = Path(__file__).resolve().parent
    sources = [root/'native/selected_host_q8.cpp',root/'native/selected_host_q8.cu']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    tensor_maker_header=Path(torch.__file__).resolve().parent/'include/ATen/ops/from_blob.h'
    header_source=tensor_maker_header.read_text()
    if 'TensorMaker& target_device(' not in header_source or 'TensorMaker for_blob(' not in header_source:
        raise RuntimeError('Deployed Torch lacks the required official TensorMaker API')
    build = args.build_directory.resolve()
    build.mkdir(parents=True,exist_ok=True)
    manifest_path = build/'build_manifest.json'
    identity = {'name':'glm53_hostq8_v3_ext','torch':torch.__version__,
        'torch_cuda':torch.version.cuda,'CUDA_HOME':str(CUDA_HOME),
        'arch':os.environ['TORCH_CUDA_ARCH_LIST'],'flags':['-O2'],
        'sources':{str(p):sha(p) for p in sources},
        'tensor_maker_header':{'path':str(tensor_maker_header),'sha256':sha(tensor_maker_header)},
        'builder_sha256':sha(Path(__file__).resolve())}
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text())['identity'] != identity:
            raise RuntimeError('Resume requires the same recorded source/toolchain/flags')
    elif any(build.iterdir()):
        raise RuntimeError('Fresh isolated build directory required, or --resume')
    manifest_path.write_text(json.dumps({'identity':identity,
        'started_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    extension = load(name=identity['name'],sources=[str(p) for p in sources],
        build_directory=str(build),extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],
        with_cuda=True,verbose=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('Build unexpectedly initialized CUDA')
    extension_file = Path(extension.__file__).resolve()
    receipt = {'identity':identity,'extension_path':str(extension_file),
        'extension_sha256':sha(extension_file),'cuda_initialized':False,
        'finished_utc':datetime.now(timezone.utc).isoformat()}
    (build/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
