#!/usr/bin/env python3
"""Pinned private API launcher with a bounded actual-layer Q1/Q8 probe.

Root owns GPU/service execution. This launcher contains no service-control or
HTTP client action. All file pins are checked before Torch/engine import.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from kda_actual_hook import check_config, install_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine-root", required=True, type=Path)
    p.add_argument("--api", required=True, type=Path)
    p.add_argument("--api-sha256", required=True)
    p.add_argument("--native", required=True, type=Path)
    p.add_argument("--native-sha256", required=True)
    p.add_argument("--trace-directory", required=True, type=Path)
    p.add_argument("--max-layers", type=int, default=34)
    p.add_argument("--max-clone-bytes", type=int, default=80 * 1024**2)
    p.add_argument("--probe-output-projection", action="store_true")
    p.add_argument("--max-projection-layers", type=int, default=1)
    p.add_argument("--max-projection-bytes", type=int, default=1024**2)
    p.add_argument("api_arguments", nargs=argparse.REMAINDER)
    args = p.parse_args()
    trace = args.trace_directory.resolve()
    if trace.exists():
        p.error("trace directory must be fresh")
    if os.environ.get("GLM53_ACTUAL_KDA_CONFIG"):
        p.error("inherited KDA diagnostic configuration is not allowed")
    trace.mkdir(mode=0o700, parents=False)
    config = {"engine_root": str(args.engine_root.resolve()),
        "api_path": str(args.api.resolve()), "api_sha256": args.api_sha256,
        "native_path": str(args.native.resolve()), "native_sha256": args.native_sha256,
        "trace_directory": str(trace), "max_layers": args.max_layers,
        "max_clone_bytes": args.max_clone_bytes,
        "probe_output_projection": args.probe_output_projection,
        "max_projection_layers": args.max_projection_layers,
        "max_projection_bytes": args.max_projection_bytes}
    check_config(config)
    config_path = trace / "config.json"
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    here = Path(__file__).resolve().parent
    # runpy.run_path(file) does not add the script's parent as a normal script
    # invocation would. API sibling helpers also need this path in spawned workers.
    paths = [str(here), str(args.native.resolve().parent), str(args.engine_root.resolve()),
             str(args.api.resolve().parent)]
    for path in reversed(paths):
        sys.path.insert(0, path)
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    os.environ["GLM53_ACTUAL_KDA_CONFIG"] = str(config_path)
    install_config(config)
    argv = args.api_arguments[1:] if args.api_arguments[:1] == ["--"] else args.api_arguments
    sys.argv = [str(args.api.resolve()), *argv]
    runpy.run_path(str(args.api.resolve()), run_name="__main__")


if __name__ == "__main__":
    main()
