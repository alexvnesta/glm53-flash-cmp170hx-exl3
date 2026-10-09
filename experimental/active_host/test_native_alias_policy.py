"""Stdlib source/custody checks. These cannot prove CUDA pointer semantics."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import validate_gpu

ROOT=Path(__file__).resolve().parent


class AliasSourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.source=(ROOT/'native/selected_host_q8.cpp').read_text()
        self.check=self.source.split('static HostMapping checked_host_mapping',1)[1].split('at::Tensor pinned_alias',1)[0]
        self.alias=self.source.split('at::Tensor pinned_alias',1)[1].split('pinned_alias_info',1)[0]

    def test_host_capability_checks_precede_explicit_tensor_target(self):
        checks=['cpu.device().is_cpu()', 'cpu.is_pinned()', 'c10::cuda::CUDAGuard guard',
                'cudaStreamIsCapturing', 'cudaPointerGetAttributes',
                'attr.type == cudaMemoryTypeHost', 'cudaHostGetFlags',
                'cudaHostGetDevicePointer', 'current_device == device_index']
        positions=[self.check.index(x) for x in checks]
        self.assertEqual(positions,sorted(positions))
        self.assertLess(self.alias.index('checked_host_mapping'),self.alias.index('at::for_blob'))
        self.assertIn('.target_device(target)',self.alias)
        self.assertIn('.options(cpu.options().device(target)',self.alias)
        self.assertNotIn('at::from_blob(',self.source)

    def test_device_and_managed_memory_have_no_relabel_path(self):
        self.assertIn('TORCH_CHECK(attr.type == cudaMemoryTypeHost',self.check)
        self.assertNotIn('cudaMemoryTypeDevice',self.check)
        self.assertNotIn('cudaMemoryTypeManaged',self.check)
        self.assertNotIn('attr.device == device_index',self.check)
        self.assertIn('std::numeric_limits<c10::DeviceIndex>::max()',self.check)

    def test_cpu_owner_retained_and_zero_host_flags_allowed(self):
        self.assertIn('.deleter([keep = cpu]',self.alias)
        self.assertIn('keep = at::Tensor()',self.alias)
        self.assertNotIn('flags & cudaHostAllocMapped',self.check)
        self.assertIn('cudaHostGetDevicePointer(&device_pointer, cpu.data_ptr(), 0)',self.check)

    def test_builder_has_distinct_extension_and_checks_deployed_header(self):
        source=(ROOT/'build_native.py').read_text()
        self.assertIn("'name':'glm53_hostq8_v3_ext'",source)
        self.assertIn('TensorMaker& target_device(',source)
        self.assertIn("'tensor_maker_header'",source)


class NativeReceiptTests(unittest.TestCase):
    def receipt(self):
        return {'identity':{'name':'glm53_hostq8_v3_ext','sources':{
            '/ABS/native/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'native').glob('selected_host_q8.*')}}}

    def test_current_v3_sources_accepted_independent_of_staged_parent_path(self):
        validate_gpu.validate_native_receipt(self.receipt())

    def test_earlier_native_binary_name_refused(self):
        r=self.receipt();r['identity']['name']='glm53_hostq8_ext'
        with self.assertRaisesRegex(RuntimeError,'distinct newly-built'):
            validate_gpu.validate_native_receipt(r)

    def test_changed_native_source_receipt_refused(self):
        r=self.receipt();key=next(iter(r['identity']['sources']))
        r['identity']['sources'][key]='0'*64
        with self.assertRaisesRegex(RuntimeError,'source hashes'):
            validate_gpu.validate_native_receipt(r)

    def test_two_distinct_device_guard_precedes_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            for devices in ('0','0,0','0,1,2'):
                args=SimpleNamespace(output=Path(temp)/'new.json',devices=devices)
                with patch.object(validate_gpu.subprocess,'check_output') as query:
                    with self.assertRaisesRegex(RuntimeError,'two distinct'):
                        validate_gpu.preimport_guards(args)
                    query.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
