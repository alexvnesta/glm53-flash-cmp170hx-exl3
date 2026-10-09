#!/usr/bin/env python3
"""Portable CPU-only contract runner; does not import engine or native module."""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import unittest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if os.environ.get("GLM53_ACTUAL_KDA_CONFIG"):
        parser.error("CPU validation refuses an active diagnostic configuration")
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ["EXLLAMAV3_ENGINE_ROOT"]=str(args.engine_root.resolve())
    import torch
    before=torch.cuda.is_initialized()
    if before:parser.error("CUDA was already initialized")
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent),pattern="test_kda*.py")
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    after=torch.cuda.is_initialized()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"cpu_validation.txt").write_text(stream.getvalue())
    document={"status":"passed" if result.wasSuccessful() and not after else "failed",
        "tests_run":result.testsRun,"torch_version":torch.__version__,
        "cuda_initialized_before":before,"cuda_initialized_after":after,
        "scope":"CPU tensors and native/HTTP doubles plus exact-source applicability; no native/GPU inference",
        "engine_source_base":"16a49792a3c93d8432d72e6c4bce800841566577"}
    (args.output/"cpu_validation.json").write_text(json.dumps(document,indent=2)+"\n")
    print(json.dumps(document))
    return 0 if document["status"]=="passed" else 1


if __name__=="__main__":sys.exit(main())
