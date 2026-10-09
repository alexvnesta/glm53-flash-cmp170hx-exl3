"""Resolve and pin explicit local engine inputs without importing the engine."""
import argparse
import hashlib
import json
from pathlib import Path

from attention_safety import SOURCE_PINS

SOURCE_LAYOUT = {
    "mla_attn": "modules/mla_attn.py",
    "bc_attn": "modules/attention_fn/bc_attn.py",
    "bc_mla": "modules/attention_fn/bc_mla.py",
    "dsa_triton": "modules/attention_fn/dsa_triton.py",
    "mla_triton": "modules/attention_fn/mla_triton.py",
}
PINS = {**SOURCE_PINS,
        "dsa_triton": "4bbe06bff677ed252efa639f14aea5f40262bfc088c92f441fc8f6f19fd6e101",
        "mla_triton": "2d4b7aeb63c1a34e9a1ebf10e17c653ae69579acd709688a310ba80a9a6a3713"}


def source_paths(engine_root):
    root = Path(engine_root).resolve(strict=True)
    package = root / "exllamav3" if (root / "exllamav3").is_dir() else root
    result = {}
    for name, relative in SOURCE_LAYOUT.items():
        path = (package / relative).resolve(strict=True)
        if not path.is_relative_to(package.resolve()) or not path.is_file():
            raise ValueError("Engine input escapes the supplied package: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
            raise ValueError("Unsupported stock engine source: " + name)
        result[name] = path
    return result


def probe_manifest(engine_root, extension, expected_extension_sha256):
    sources = source_paths(engine_root)
    if (len(expected_extension_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_extension_sha256)):
        raise ValueError("Supply the independently attested engine-extension SHA256")
    extension = Path(extension).resolve(strict=True)
    if not extension.is_file() or hashlib.sha256(extension.read_bytes()).hexdigest() != expected_extension_sha256:
        raise ValueError("Engine extension differs from its supplied attestation")
    return {**{name: {"path": str(sources[name]), "sha256": PINS[name]}
               for name in ("dsa_triton", "mla_triton")},
            "engine_extension": {"path": str(extension), "sha256": expected_extension_sha256}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--engine-extension", type=Path, required=True)
    parser.add_argument("--engine-extension-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = probe_manifest(args.engine_root, args.engine_extension, args.engine_extension_sha256)
    # Exclusive create preserves a previously attested manifest.
    with args.output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
