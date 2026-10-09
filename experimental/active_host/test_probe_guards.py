"""Stdlib-only probes of guard ordering. These tests do not run nvidia-smi."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import validate_gpu


class ProbeGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.inputs={}
        for name in ('dsa_triton','mla_triton','engine_extension'):
            source=root/name;source.write_bytes(name.encode())
            self.inputs[name]={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        manifest=root/'inputs.json';manifest.write_text(json.dumps(self.inputs))
        self.args=SimpleNamespace(output=root/'output.json',devices='0,1',input_manifest=manifest)

    def query(self,cmd,**kwargs):
        if '--query-gpu=index,uuid' in cmd:return '0, GPU-A\n1, GPU-B\n'
        if '--query-compute-apps=pid,gpu_uuid' in cmd:return ''
        raise AssertionError(cmd)

    def test_existing_output_rejected_before_inventory(self):
        self.args.output.write_text('owned by earlier run')
        with patch.object(validate_gpu.subprocess,'check_output') as query:
            with self.assertRaisesRegex(RuntimeError,'pre-existing'):
                validate_gpu.preimport_guards(self.args)
            query.assert_not_called()

    def test_foreign_gpu_owner_rejected(self):
        def query(cmd,**kwargs):
            if '--query-compute-apps=pid,gpu_uuid' in cmd:return f'{os.getpid()+100}, GPU-A\n'
            return self.query(cmd,**kwargs)
        with patch.object(validate_gpu.subprocess,'check_output',side_effect=query):
            with self.assertRaisesRegex(RuntimeError,'foreign GPU owner'):
                validate_gpu.preimport_guards(self.args)

    def test_changed_engine_input_rejected(self):
        Path(self.inputs['dsa_triton']['path']).write_bytes(b'changed')
        with patch.object(validate_gpu.subprocess,'check_output',side_effect=self.query):
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                validate_gpu.preimport_guards(self.args)

    def test_complete_pinned_inputs_accepted(self):
        with patch.object(validate_gpu.subprocess,'check_output',side_effect=self.query):
            self.assertEqual(validate_gpu.preimport_guards(self.args),self.inputs)


if __name__=='__main__':unittest.main(verbosity=2)
