"""Portable local-input and default-off launcher contracts, no CUDA."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import engine_inputs
import run_trial


class EngineInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = engine_inputs.source_paths(Path(os.environ["EXLLAMAV3_ENGINE_ROOT"]))
        self.extension = self.root / "engine.so"
        self.extension.write_bytes(b"test-only independently attested binary")
        self.extension_sha = hashlib.sha256(self.extension.read_bytes()).hexdigest()

    def test_known_source_paths_derive_from_external_checkout(self):
        self.assertEqual(set(self.paths), set(engine_inputs.PINS))
        self.assertTrue(all(path.is_file() for path in self.paths.values()))

    def test_generated_manifest_matches_probe_exact_three_input_contract(self):
        result = engine_inputs.probe_manifest(Path(os.environ["EXLLAMAV3_ENGINE_ROOT"]),
                                             self.extension, self.extension_sha)
        self.assertEqual(set(result), {"dsa_triton", "mla_triton", "engine_extension"})
        self.assertEqual(result["engine_extension"]["sha256"], self.extension_sha)

    def test_changed_engine_source_refuses_before_manifest(self):
        package = self.root / "engine/exllamav3"
        for name, relative in engine_inputs.SOURCE_LAYOUT.items():
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.paths[name].read_bytes())
        (package / engine_inputs.SOURCE_LAYOUT["dsa_triton"]).write_bytes(b"changed source")
        with self.assertRaisesRegex(ValueError, "Unsupported stock.*dsa_triton"):
            engine_inputs.probe_manifest(package, self.extension, self.extension_sha)

    def test_source_symlink_outside_supplied_package_refuses(self):
        package = self.root / "engine/exllamav3"
        path = package / engine_inputs.SOURCE_LAYOUT["mla_attn"]
        path.parent.mkdir(parents=True)
        path.symlink_to(self.paths["mla_attn"])
        with self.assertRaisesRegex(ValueError, "escapes.*mla_attn"):
            engine_inputs.source_paths(package)

    def test_extension_drift_and_malformed_attestation_refuse(self):
        for digest in ("A" * 64, "0" * 63, "0" * 64):
            with self.assertRaises(ValueError):
                engine_inputs.probe_manifest(Path(os.environ["EXLLAMAV3_ENGINE_ROOT"]),
                                             self.extension, digest)

    def test_cli_exclusive_output_preserves_previous_bytes(self):
        output = self.root / "inputs.json"
        output.write_text("previous attestation")
        argv = ["engine_inputs.py", "--engine-root", os.environ["EXLLAMAV3_ENGINE_ROOT"],
                "--engine-extension", str(self.extension),
                "--engine-extension-sha256", self.extension_sha, "--output", str(output)]
        with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
            engine_inputs.main()
        self.assertEqual(output.read_text(), "previous attestation")

    def test_normal_launcher_keeps_host_hooks_unimported(self):
        api = self.root / "glm_api.py"
        api.write_text("# normal API placeholder")
        before = {name: name in sys.modules for name in ("torch", "engine_adapter", "selected_host_q8")}
        argv = ["run_trial.py", "--api", str(api), "--", "--max-batch-size", "1"]
        with patch.object(sys, "argv", argv), patch.object(run_trial.runpy, "run_path") as run:
            run_trial.main()
            run.assert_called_once_with(str(api.resolve()), run_name="__main__")
        self.assertEqual(before, {name: name in sys.modules for name in before})


if __name__ == "__main__":
    unittest.main(verbosity=2)
