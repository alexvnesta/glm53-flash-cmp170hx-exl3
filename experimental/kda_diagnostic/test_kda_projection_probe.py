from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import kda_actual_hook as hook
import kda_projection_probe as projection
from test_kda_actual_hook import Fixture


class Native:
    def __init__(self):
        self.calls, self.checks = [], []
        self.tags = [2, 3, 2, 2]
        self.mutate = None
        self.compatible = lambda tag, m: True
        self.num_shapes = 7
        self.nonfinite = False

    def g_get_num_sms(self, device): return 108
    def g_get_cc(self, device): return 80
    def exl3_gemm_num_kernel_shapes(self): return self.num_shapes
    def exl3_gemm_shape_compat(self, tag, m, k, n, bits, half_k):
        self.checks.append((tag, m, k, n, bits, half_k))
        return self.compatible(tag, m)
    def exl3_gemm(self, a, b, c, suh, scratch, svh, tag, mcg, mul1, sms):
        number = len(self.calls)
        self.calls.append((a, b, c, suh, scratch, svh, tag, mcg, mul1, sms))
        c.copy_(a[:, :c.shape[-1]].to(c.dtype))
        if number == 1: c.add_(0.125)  # Controlled automatic profile difference.
        if self.nonfinite: c.fill_(float("inf"))
        scratch.fill_(9)
        if self.mutate: self.mutate()
        return self.tags[number]


class ProjectionFixture:
    def __init__(self, *, dtype=torch.half, bias=False, mul1=False, half_k=False):
        self.inner = SimpleNamespace(quant_type="exl3", trellis=torch.zeros((8, 8, 40 if half_k else 32), dtype=torch.int16),
            suh=torch.ones(128,dtype=torch.half), svh=torch.ones(128,dtype=torch.half),
            bias=torch.ones(128,dtype=torch.half) if bias else None,
            mcg=False,mul1=mul1,mcg_tensor=None,
            mul1_tensor=torch.tensor(1,dtype=torch.int32) if mul1 else None,
            in_features=128,out_features=128)
        self.layer=SimpleNamespace(o_proj=SimpleNamespace(quant_type="exl3",inner=self.inner),
            layer_idx=0,hidden_size=128,out_dtype=dtype)
        self.first=torch.arange(128,dtype=torch.half).reshape(1,128)
        self.protected={"input":torch.ones((1,8,128),dtype=torch.half),
                        "state":torch.ones((2,8,1,3,5)),
                        "scratch":torch.ones((1,8,128),dtype=torch.half),
                        "origin":self.first}
        self.native=Native()
        self.config={"max_projection_bytes":1024**2}
    def run(self, **kwargs):
        with patch.dict(os.environ,{"EXL3_GEMV":"0","EXL3_INT8_GEMV":"0"}):
            return projection.compare_projection(torch,self.native,self.layer,self.first,
                lambda:self.protected,self.config,allow_cpu=True,**kwargs)


class ProjectionTests(unittest.TestCase):
    def test_actual_origin_repeated_and_automatic_vs_common_difference(self):
        f=ProjectionFixture(); row=f.run()
        self.assertTrue(row["success"])
        self.assertEqual([call[-4] for call in f.native.calls],[-1,-1,2,2])
        self.assertEqual([call[-1] for call in f.native.calls],[0,0,1,1])
        self.assertTrue(torch.equal(f.native.calls[1][0],f.first.repeat(8,1)))
        self.assertEqual([c["actual_tag"] for c in row["calls"]],[2,3,2,2])
        self.assertFalse(row["automatic_first"]["bitwise_exact"])
        self.assertEqual(row["automatic_first"]["max_abs_finite"],0.125)
        self.assertTrue(row["common_first"]["bitwise_exact"])
        self.assertEqual(row["process_environment"]["EXL3_GEMV"],"0")
        self.assertEqual(row["protected_before"],row["protected_after"])
        self.assertTrue(row["owned_inputs_unchanged"])

    def test_native_receives_only_owned_input_output_scratch(self):
        f=ProjectionFixture(); before=f.protected["scratch"].clone(); f.run()
        borrowed={v.data_ptr() for v in f.protected.values()}
        for a,b,c,suh,scratch,svh,*_ in f.native.calls:
            self.assertIs(b,f.inner.trellis)
            for value in (a,c,scratch): self.assertNotIn(value.data_ptr(),borrowed)
            self.assertNotEqual(a.data_ptr(),scratch.data_ptr())
        self.assertTrue(torch.equal(f.protected["scratch"],before))

    def test_native_shape_checks_cover_both_row_counts(self):
        f=ProjectionFixture(); f.run()
        self.assertEqual(f.native.checks,[(2,1,128,128,2,False),(2,8,128,128,2,False),
                                         (3,1,128,128,2,False),(3,8,128,128,2,False)])

    def test_q8_shape_selected_only_if_common_for_both(self):
        f=ProjectionFixture(); f.native.compatible=lambda tag,m:tag==3
        f.native.tags=[2,3,3,3]
        self.assertEqual(f.run()["common_shape"],3)

    def test_no_common_shape_stops_before_forced_calls(self):
        f=ProjectionFixture(); f.native.compatible=lambda tag,m:m==8
        with self.assertRaisesRegex(projection.ProjectionRefusal,"no automatic tag") as caught: f.run()
        self.assertEqual(len(f.native.calls),2)
        self.assertFalse(caught.exception.diagnostic["success"])

    def test_gemv_or_zero_tags_never_forced(self):
        for tag in (0,90,91,-1,8):
            f=ProjectionFixture(); f.native.tags=[tag,tag,2,2]
            with self.subTest(tag=tag), self.assertRaisesRegex(projection.ProjectionRefusal,"cooperative"):
                f.run()
            self.assertEqual(len(f.native.calls),2)

    def test_wrong_forced_tag_fails_with_actual_tags_retained(self):
        f=ProjectionFixture(); f.native.tags=[2,3,3,2]
        with self.assertRaisesRegex(projection.ProjectionRefusal,"different forced") as caught: f.run()
        self.assertEqual(caught.exception.diagnostic["calls"][2]["actual_tag"],3)

    def test_invalid_tag_type_or_shape_count_refused(self):
        for kind in ("boolean_tag","shape_count"):
            f=ProjectionFixture()
            if kind=="boolean_tag":f.native.tags[0]=True
            else:f.native.num_shapes=8
            with self.subTest(kind=kind),self.assertRaises(projection.ProjectionRefusal):f.run()
            self.assertLessEqual(len(f.native.calls),2)

    def test_nonfinite_actual_input_refused_before_native_call(self):
        f=ProjectionFixture();f.first[0,0]=float("nan")
        with self.assertRaisesRegex(projection.ProjectionRefusal,"input contains nonfinite"):f.run()
        self.assertEqual(f.native.calls,[])

    def test_strided_origin_or_trellis_refused_before_native_call(self):
        for kind in ("origin","trellis"):
            f=ProjectionFixture()
            if kind=="origin":f.first=torch.ones((1,256),dtype=torch.half)[:,::2]
            else:f.inner.trellis=f.inner.trellis.transpose(0,1)
            with self.subTest(kind=kind),self.assertRaisesRegex(projection.ProjectionRefusal,"contigu"):f.run()
            self.assertEqual(f.native.calls,[])

    def test_native_error_retained_and_no_original_call(self):
        f=ProjectionFixture()
        def fail(*args): raise RuntimeError("native launch fault")
        f.native.exl3_gemm=fail
        with self.assertRaisesRegex(projection.ProjectionRefusal,"launch fault") as caught:f.run()
        self.assertEqual(caught.exception.diagnostic["error_type"],"RuntimeError")

    def test_nonfinite_output_fails_before_forcing(self):
        f=ProjectionFixture(); f.native.nonfinite=True
        with self.assertRaisesRegex(projection.ProjectionRefusal,"nonfinite") as caught:f.run()
        self.assertFalse(caught.exception.diagnostic["calls"][0]["output_finite"])
        self.assertEqual(len(f.native.calls),1)

    def test_environment_refusal_precedes_native_calls(self):
        for name,value in (("EXL3_GEMV","1"),("EXL3_INT8_GEMV","2"),("EXL3_GEMV",None)):
            f=ProjectionFixture()
            env={"EXL3_GEMV":"0","EXL3_INT8_GEMV":"0"}
            if value is None:env.pop(name)
            else:env[name]=value
            with patch.dict(os.environ,env,clear=True),self.assertRaisesRegex(projection.ProjectionRefusal,"actual EXL3"):
                projection.compare_projection(torch,f.native,f.layer,f.first,lambda:f.protected,f.config,allow_cpu=True)
            self.assertEqual(f.native.calls,[])

    def test_cpu_device_is_refused_without_internal_test_allowance(self):
        f=ProjectionFixture()
        with patch.dict(os.environ,{"EXL3_GEMV":"0","EXL3_INT8_GEMV":"0"}),self.assertRaisesRegex(projection.ProjectionRefusal,"actual CUDA"):
            projection.compare_projection(torch,f.native,f.layer,f.first,lambda:f.protected,f.config)
        self.assertEqual(f.native.calls,[])

    def test_missing_each_abi_function_refuses_before_call(self):
        for name in projection.REQUIRED_ABI:
            f=ProjectionFixture(); setattr(f.native,name,None)
            with self.subTest(name=name),self.assertRaisesRegex(projection.ProjectionRefusal,"ABI missing"):f.run()
            self.assertEqual(f.native.calls,[])

    def test_budget_refusal_precedes_native_calls(self):
        f=ProjectionFixture(); f.config["max_projection_bytes"]=1
        with self.assertRaisesRegex(projection.ProjectionRefusal,"budget"):f.run()
        self.assertEqual(f.native.calls,[])

    def test_float_output_dtype_and_exact_allocation_estimate(self):
        f=ProjectionFixture(dtype=torch.float); row=f.run()
        self.assertEqual(row["owned_projection_bytes_estimate"],9*(128*2+128*2+128*4))
        self.assertEqual(f.native.calls[0][2].dtype,torch.float)

    def test_bias_is_hashed_and_excluded_from_native_projection(self):
        f=ProjectionFixture(bias=True); row=f.run()
        self.assertTrue(row["bias_present"])
        self.assertIn("weight:bias",row["protected_before"])
        self.assertIn("excluded",row["bias_scope"])

    def test_cpu_marker_and_half_integer_bitrate_supported(self):
        f=ProjectionFixture(mul1=True,half_k=True);row=f.run()
        self.assertTrue(row["half_integer_bits"])
        self.assertEqual(row["bits"],2)
        self.assertIn("weight:mul1_tensor",row["protected_before"])
        self.assertTrue(all(item[-1] for item in f.native.checks))

    def test_invalid_geometry_types_codebook_and_widths_refuse(self):
        mutations=[lambda f:setattr(f.inner,"trellis",f.inner.trellis.float()),
            lambda f:setattr(f.inner,"suh",f.inner.suh[:127]),
            lambda f:setattr(f.inner,"svh",None),lambda f:setattr(f.inner,"in_features",256),
            lambda f:setattr(f.layer,"hidden_size",256),lambda f:setattr(f.inner,"mcg",1),
            lambda f:setattr(f.inner,"mul1",True),lambda f:setattr(f.inner,"quant_type","fp16"),
            lambda f:setattr(f.inner,"trellis",torch.zeros((8,8,39),dtype=torch.int16)),
            lambda f:setattr(f.inner,"trellis",torch.zeros((8,8,136),dtype=torch.int16)),
            lambda f:setattr(f.layer,"out_dtype",torch.bfloat16)]
        for number,mutate in enumerate(mutations):
            f=ProjectionFixture();mutate(f)
            with self.subTest(number=number),self.assertRaises(projection.ProjectionRefusal):f.run()
            self.assertEqual(f.native.calls,[])

    def test_state_static_or_weight_mutation_detected(self):
        for kind in ("state","scratch","trellis","suh","marker"):
            f=ProjectionFixture(mul1=True)
            tensor=(f.protected[kind] if kind in f.protected else
                    f.inner.mul1_tensor if kind=="marker" else getattr(f.inner,kind))
            f.native.mutate=lambda:tensor.add_(1)
            with self.subTest(kind=kind),self.assertRaisesRegex(projection.ProjectionRefusal,"changed protected"):f.run()

    def test_equal_value_weight_or_state_storage_replacement_detected(self):
        for kind in ("state","trellis"):
            f=ProjectionFixture()
            def mutate():
                if kind=="state":f.protected[kind]=f.protected[kind].clone()
                else:f.inner.trellis=f.inner.trellis.clone()
            f.native.mutate=mutate
            with self.subTest(kind=kind),self.assertRaisesRegex(projection.ProjectionRefusal,"changed protected"):f.run()

    def test_projection_object_or_scalar_codebook_mutation_refused(self):
        for kind in ("object","codebook"):
            f=ProjectionFixture()
            def mutate():
                if kind=="object":f.layer.o_proj.inner=SimpleNamespace(**vars(f.inner))
                else:f.inner.mcg=True
            f.native.mutate=mutate
            with self.subTest(kind=kind),self.assertRaisesRegex(projection.ProjectionRefusal,"object/codebook changed"):f.run()

    def test_owned_input_mutation_refused(self):
        f=ProjectionFixture()
        f.native.mutate=lambda:f.native.calls[-1][0].add_(1)
        with self.assertRaisesRegex(projection.ProjectionRefusal,"owned inputs"):f.run()

    def test_source_abi_pins_and_exposed_shape_compat_signature(self):
        root=Path(os.environ.get("EXLLAMAV3_ENGINE_ROOT",Path(__file__).resolve().parents[1]/"upstream_publication_implementation/deployed_native"))
        for name,pin in hook.SOURCE_PINS.items():self.assertEqual(hook.sha(root/name),pin,name)
        source=(root/"exllamav3/exllamav3_ext/bindings.cpp").read_text()
        for name in projection.REQUIRED_ABI:self.assertIn('m.def("'+name+'"',source)
        self.assertIn('py::arg("half_k") = false',source)

    def test_projection_config_bounds_and_env_checked_before_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"api.py").write_text("api");(root/"native.so").write_text("native")
            config={"engine_root":str(root),"api_path":str(root/"api.py"),"api_sha256":hook.sha(root/"api.py"),
                "native_path":str(root/"native.so"),"native_sha256":hook.sha(root/"native.so"),
                "max_layers":34,"max_clone_bytes":80*1024**2,"trace_directory":str(root),
                "probe_output_projection":True,"max_projection_layers":1,"max_projection_bytes":1024**2}
            for change in ({"max_projection_layers":4},{"max_projection_layers":True},
                           {"max_projection_bytes":4*1024**2+1},{"probe_output_projection":"yes"}):
                called=[]
                with patch.object(hook,"SOURCE_PINS",{}),patch.dict(os.environ,{"EXL3_GEMV":"0","EXL3_INT8_GEMV":"0"}),self.assertRaises(hook.Refusal):
                    hook.install_config({**config,**change},loader=lambda:called.append("torch/native"))
                self.assertEqual(called,[])
            called=[]
            with patch.object(hook,"SOURCE_PINS",{}),patch.dict(os.environ,{},clear=True),self.assertRaisesRegex(hook.Refusal,"before native import"):
                hook.install_config(config,loader=lambda:called.append("torch/native"))
            self.assertEqual(called,[])

    def test_projection_pin_failure_precedes_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);name="exllamav3/exllamav3_ext/bindings.cpp"
            path=root/name;path.parent.mkdir(parents=True);path.write_text("bad ABI")
            called=[]
            with patch.object(hook,"SOURCE_PINS",{name:hook.SOURCE_PINS[name]}),self.assertRaisesRegex(hook.Refusal,"bindings.cpp"):
                hook.install_config({"engine_root":str(root)},loader=lambda:called.append("torch"))
            self.assertEqual(called,[])

    def test_integrated_hook_default_off_never_calls_direct_binding(self):
        f=Fixture();f.probe.native=Native();f.forward(f.layer,f.x,f.params)
        self.assertNotIn("output_projection",f.rows[0]);self.assertEqual(f.probe.native.calls,[])

    def test_integrated_hook_limit_and_same_origin_protected_before_original(self):
        f=Fixture();f.probe.config.update(probe_output_projection=True,max_projection_layers=1,max_projection_bytes=1024**2)
        seen=[]
        def compare(torch_,native,layer,first,protected,config):
            self.assertEqual(len(f.bc.calls),2)
            self.assertTrue(torch.equal(first,f.cache.cache[f.cache.make_key(f.layer.device,(1,8,6),torch.half,"s_caof")][1][:,:1].reshape(1,6)))
            self.assertEqual(len(protected()),32)
            seen.append(layer.layer_idx)
            return {"event":"kda_output_projection_compare","success":True}
        with patch.object(projection,"compare_projection",compare):
            with torch.inference_mode():
                f.forward(f.layer,f.x,f.params)
                second=SimpleNamespace(**vars(f.layer));second.layer_idx=4
                f.forward(second,f.x,f.params)
        self.assertEqual(seen,[3]);self.assertEqual(len(f.probe.projection_seen),1)
        self.assertTrue(f.rows[0]["original_forward_completed"])
        self.assertIn("request_context",f.rows[0]["output_projection"])
        self.assertNotIn("output_projection",f.rows[1])

    def test_integrated_projection_error_stops_original_and_clears_scope(self):
        f=Fixture();f.probe.config.update(probe_output_projection=True,max_projection_layers=1,max_projection_bytes=1024**2)
        error=projection.ProjectionRefusal("owned diagnostic fault",{"actual_tag":2})
        with patch.object(projection,"compare_projection",side_effect=error),self.assertRaisesRegex(projection.ProjectionRefusal,"diagnostic fault"):
            f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.original_calls,0)
        self.assertIsNone(f.probe.request_context())
        self.assertEqual(f.rows[-1]["projection_diagnostic"],{"actual_tag":2})

    def test_original_forward_failure_retains_completed_projection_diagnostic(self):
        f=Fixture();f.probe.config.update(probe_output_projection=True,max_projection_layers=1,max_projection_bytes=1024**2)
        def original(*args):raise RuntimeError("original forward fault")
        f.raw_forward=f.probe.wrap(original)
        with patch.object(projection,"compare_projection",return_value={"success":True,"calls":[{"actual_tag":2}]}),self.assertRaisesRegex(RuntimeError,"original forward"):
            f.forward(f.layer,f.x,f.params)
        self.assertTrue(f.rows[-1]["projection_diagnostic"]["success"])
        self.assertEqual(f.rows[-1]["projection_diagnostic"]["calls"][0]["actual_tag"],2)
        self.assertIsNone(f.probe.request_context())

    def test_no_cuda_initialization(self):self.assertFalse(torch.cuda.is_initialized())


if __name__=="__main__":unittest.main()
