import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import os
import subprocess
import sys
from unittest.mock import patch

import torch
import kda_actual_hook as hook


class Cache:
    def __init__(self): self.cache = {}
    def make_key(self, device, shape, dtype, tag): return str((device, shape, dtype, tag))


class BC:
    def __init__(self, layer, cache):
        self.layer, self.cache = layer, cache
        self.configured, self.calls = set(), []
        self.mutation = None
        self.divergent = None

    def needs_configure(self, batch, sequence, history):
        return (batch, sequence, history) not in self.configured

    def configure(self, batch, sequence, history):
        self.configured.add((batch, sequence, history))
        for tag, (shape, dtype, axis) in hook.buffer_specs(self.layer, torch, sequence).items():
            key = self.cache.make_key(self.layer.device, shape, dtype, tag)
            self.cache.cache[key] = (1, torch.zeros(shape, dtype=dtype))

    def run_bszN(self, x, y, conv, recurrent, slots, history):
        self.calls.append((x, y, conv, recurrent, slots, history))
        index = int(slots.item())
        y.copy_(x)
        initial = recurrent[index, 0].clone()
        if history:
            recurrent[index, 1].copy_(initial + x[:, 0].sum())
            recurrent[index, 0].add_(x.sum())
        else:
            recurrent[index, 0].add_(x.sum())
        conv[index].add_(1)
        for number, (tag, (shape, dtype, axis)) in enumerate(hook.buffer_specs(self.layer, torch, x.shape[1]).items()):
            tensor = self.cache.cache[self.cache.make_key(self.layer.device, shape, dtype, tag)][1]
            tensor.fill_(number + (1 if self.divergent == tag and history else 0))
        if self.mutation:
            self.mutation()


class Fixture:
    def __init__(self, *, exported=False, slot=1, budget=80*1024**2):
        self.cache = Cache()
        self.layer = SimpleNamespace(device=torch.device("cpu"), kda=True, bc_split=True,
            ba_weight_filled=True, num_k_heads=2, num_v_heads=2, k_head_dim=2, v_head_dim=3,
            fdim_qkv=14, hidden_size=6, layer_idx=3, out_dtype=torch.half)
        self.conv = torch.zeros((3,14,11),dtype=torch.bfloat16)
        self.recurrent = torch.arange(3*8*2*2*3,dtype=torch.float).reshape(3,8,2,2,3)
        self.rsl = SimpleNamespace(get_state_tensors=lambda:(self.conv,self.recurrent))
        self.lookups = []
        def lookup(instance):
            self.lookups.append(instance)
            return self.rsl
        self.rsg = SimpleNamespace(exported=exported, cache=17 if exported else SimpleNamespace(get_recurrent_layer=lookup))
        self.layer.tp_recurrent_lookup = {17:self.rsl}
        self.bc = self.layer.bc = BC(self.layer,self.cache)
        self.layer._bc_configure_slot_kda = self.bc.configure
        self.params = {"recurrent_history":True,"recurrent_states":[self.rsg],
            "recurrent_slots":torch.tensor([slot],dtype=torch.int),"layer_instance":2}
        self.x = torch.arange(48,dtype=torch.half).reshape(1,8,6)
        self.rows, self.events = [], []
        def writer(row):
            self.events.append(row)
            if row["event"] != "kda_request_phase_started":
                self.rows.append(row)
        self.probe = hook.ActualKDAProbe({"max_layers":34,"max_clone_bytes":budget},torch,self.cache,
            lambda params,key,device:params[key],writer)
        self.original_calls = 0
        def original(layer,x,params,out_dtype):
            self.original_calls += 1
            if self.bc.needs_configure(1,x.shape[1],True): self.bc.configure(1,x.shape[1],True)
            y = torch.empty_like(x)
            self.bc.run_bszN(x,y,self.conv,self.recurrent,params["recurrent_slots"],True)
            return y
        self.raw_forward = self.probe.wrap(original)
        self.generator = SimpleNamespace(model=SimpleNamespace(loaded_tp=False), draft_model=None,
                                         num_remaining_jobs=lambda:1)
        scoped = self.probe.wrap_iterate(lambda generator,layer,x,params,out_dtype=None:
                                        self.raw_forward(layer,x,params,out_dtype))
        self.forward = lambda layer,x,params,out_dtype=None:scoped(self.generator,layer,x,params,out_dtype)


class HookTests(unittest.TestCase):
    def test_loader_calls_do_not_consume_probe_before_actual_request(self):
        f = Fixture()
        f.raw_forward(f.layer,f.x,f.params)
        self.assertEqual(f.probe.seen,set())
        self.assertEqual(f.rows,[])
        self.assertEqual(len(f.bc.calls),1)
        f.forward(f.layer,f.x,f.params)
        self.assertEqual(len(f.rows),1)
        self.assertEqual([call[-1] for call in f.bc.calls],[True,False,True,True])
        self.assertEqual(f.rows[0]["request_context"]["phase"],"nonempty_generator_iterate")
        self.assertFalse(f.rows[0]["request_context"]["target_tp"])
        self.assertIsNone(f.probe.request_context())

    def test_empty_iteration_does_not_enable_probe(self):
        f = Fixture(); f.generator.num_remaining_jobs=lambda:0
        f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.probe.seen,set())
        self.assertEqual(f.rows,[])
        self.assertEqual(f.probe.iteration_number,0)

    def test_request_phase_is_thread_local(self):
        import threading
        f = Fixture(); other=[]
        def step(generator):
            self.assertIsNotNone(f.probe.request_context())
            worker=threading.Thread(target=lambda:other.append(
                (f.probe.request_context(),f.probe.eligible(f.layer,f.x,f.params))))
            worker.start(); worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            return "done"
        self.assertEqual(f.probe.wrap_iterate(step)(f.generator),"done")
        self.assertEqual(other,[(None,False)])
        self.assertIsNone(f.probe.request_context())

    def test_exception_always_clears_request_phase_and_loader_stays_inactive(self):
        f = Fixture()
        def fail(generator):
            self.assertIsNotNone(f.probe.request_context())
            raise ValueError("actual iterator fault")
        with self.assertRaisesRegex(ValueError,"actual iterator"):
            f.probe.wrap_iterate(fail)(f.generator)
        self.assertIsNone(f.probe.request_context())
        f.raw_forward(f.layer,f.x,f.params)
        self.assertEqual(f.probe.seen,set())

    def test_tp_target_or_draft_refused_before_iteration(self):
        for side in ("target","draft"):
            with self.subTest(side=side):
                f=Fixture(); called=[]
                if side=="target": f.generator.model.loaded_tp=True
                else: f.generator.draft_model=SimpleNamespace(loaded_tp=True)
                with self.assertRaisesRegex(hook.Refusal,"layer-split-only"):
                    f.probe.wrap_iterate(lambda generator:called.append("inference"))(f.generator)
                self.assertEqual(called,[])
                self.assertIsNone(f.probe.request_context())

    def test_nested_iteration_refused_and_outer_scope_cleared(self):
        f=Fixture()
        inner=f.probe.wrap_iterate(lambda generator:"unexpected")
        outer=f.probe.wrap_iterate(lambda generator:inner(generator))
        with self.assertRaisesRegex(hook.Refusal,"nested"):
            outer(f.generator)
        self.assertIsNone(f.probe.request_context())

    def test_nested_empty_iteration_cannot_inherit_outer_request_phase(self):
        f=Fixture(); inner_calls=[]
        empty=SimpleNamespace(model=SimpleNamespace(loaded_tp=False),draft_model=None,
                              num_remaining_jobs=lambda:0)
        inner=f.probe.wrap_iterate(lambda generator:inner_calls.append("unexpected"))
        outer=f.probe.wrap_iterate(lambda generator:inner(empty))
        with self.assertRaisesRegex(hook.Refusal,"nested"):
            outer(f.generator)
        self.assertEqual(inner_calls,[])
        self.assertIsNone(f.probe.request_context())

    def test_invalid_job_count_refused_before_iteration(self):
        for count in (-1,True,None,1.0):
            f=Fixture(); called=[]; f.generator.num_remaining_jobs=lambda:count
            with self.assertRaisesRegex(hook.Refusal,"job count"):
                f.probe.wrap_iterate(lambda generator:called.append("inference"))(f.generator)
            self.assertEqual(called,[])

    def test_actual_pinned_iterate_activates_only_request_model_work(self):
        import ast
        source_root=Path(os.environ.get("EXLLAMAV3_ENGINE_ROOT", Path(__file__).resolve().parents[1]/"upstream_publication_implementation/deployed_native"))
        path=source_root/"exllamav3/generator/generator.py"
        self.assertEqual(hook.sha(path),hook.SOURCE_PINS["exllamav3/generator/generator.py"])
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Generator")
        constructor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="iterate"
                             for n in ast.walk(constructor)))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="iterate")
        namespace={"torch":torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),"exec"),namespace)
        f=Fixture(); g=f.generator
        g.cache=SimpleNamespace(initialized=True);g.draft_cache=SimpleNamespace(initialized=True)
        g.recurrent_cache=None;g.active_jobs=[];g.pending_jobs=[object()]
        g.draft_model=SimpleNamespace(loaded_tp=False);g.dflash_draft=True;g.visualizer=None
        g.iterate_start_jobs=lambda results:None
        g.iterate_draftmodel_dflash_gen=lambda results:"draft-tokens"
        def generate(results,draft):
            self.assertEqual(draft,"draft-tokens")
            results.append(f.raw_forward(f.layer,f.x,f.params))
        g.iterate_gen=generate
        # Exact method's constructor-independent body runs on CPU doubles.
        result=f.probe.wrap_iterate(namespace["iterate"])(g)
        self.assertEqual(len(result),1)
        self.assertEqual(len(f.rows),1)
        self.assertEqual(f.rows[0]["request_context"]["jobs_at_entry"],1)
        self.assertIsNone(f.probe.request_context())

    def test_cases_fixture_preserves_original_request_bodies(self):
        import json,hashlib
        root=Path(__file__).resolve().parent
        original=json.loads((root/"target_equivalence_fixture.json").read_text())
        adapted=json.loads((root/"kda_request_fixture.json").read_text())
        cases=adapted["cases"]
        self.assertEqual([c["id"] for c in cases],list(original["requests"]))
        self.assertEqual({c["id"]:c["request"] for c in cases},original["requests"])
        self.assertEqual(hashlib.sha256((root/"target_equivalence_fixture.json").read_bytes()).hexdigest(),
                         "4f8e06dcf5354491f9a7bd29fe1a55c8ed23909fe3e68f54049b39076ce09f0d")
        self.assertTrue(all(c["request"]["model"]=="glm" for c in cases))
        import kda_trial_benchmark as runner
        _,loaded_cases,digest=runner.load_fixture(root/"kda_request_fixture.json")
        self.assertEqual(loaded_cases,cases)
        self.assertEqual(digest,hashlib.sha256((root/"kda_request_fixture.json").read_bytes()).hexdigest())

    def test_scoped_runner_accepts_only_alias_or_canonical_model_identity(self):
        import kda_trial_benchmark as runner
        health={"status":"ok","engine":"ExLlamaV3","cache_tokens":393216,
                "context_length":392960,"speculative_method":"dflash2","draft_num_tokens":7,
                "busy":False,"model":"GLM-5.3-Flash"}
        self.assertEqual(runner.health_error(health,{"glm"}),{})
        self.assertEqual(runner.health_error(health,{"GLM-5.3-Flash"}),{})
        for models in (set(),{"other"},{"glm","GLM-5.3-Flash"},{"glm","other"}):
            self.assertIn("model",runner.health_error(health,models))
        self.assertIn("model",runner.health_error({**health,"model":"glm"},{"glm"}))
        self.assertIn("busy",runner.health_error({**health,"busy":True},{"glm"}))

    def test_scoped_runner_never_rewrites_request_payload_or_hash(self):
        import json,io,contextlib
        import kda_trial_benchmark as runner
        fixture=Path(__file__).resolve().parent/"kda_request_fixture.json"
        original=json.loads(fixture.read_text()); submitted=[]
        health={"status":"ok","engine":"ExLlamaV3","cache_tokens":393216,
                "context_length":392960,"speculative_method":"dflash2","draft_num_tokens":7,
                "busy":False,"model":"GLM-5.3-Flash"}
        def fake_http(base,path,timeout,payload=None):
            if path=="/health": return {"http_status":200,"body":dict(health),"wall_seconds":0.1}
            submitted.append(payload)
            return {"http_status":200,"wall_seconds":1.0,"body":{
                "usage":{"completion_tokens":128,"draft_tokens":0,"draft_tokens_accepted":0},
                "timings":{"predicted_n":128,"predicted_ms":1000.0,"draft_n":0,"draft_n_accepted":0},
                "choices":[{"message":{"content":"synthetic HTTP double"},"finish_reason":"length"}]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=runner.parse_args(["--fixture",str(fixture),"--output",str(root/"raw.jsonl"),
                "--summary",str(root/"summary.json"),"--condition","CPU-double","--repeats","1"])
            with patch.object(runner,"http_request",fake_http),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run_trial(args),0)
            records=[json.loads(line) for line in (root/"raw.jsonl").read_text().splitlines()]
            results=[record for record in records if record["event"]=="result"]
            self.assertEqual(submitted,[case["request"] for case in original["cases"]])
            self.assertEqual(len(results),4)
            for record,case in zip(results,original["cases"]):
                self.assertTrue(record["success"])
                self.assertEqual(record["request"],case["request"])
                self.assertEqual(record["request_sha256"],runner.request_hash(case["request"]))
                self.assertEqual(record["request"]["model"],"glm")

    def test_scoped_runner_unknown_or_mixed_models_refuse_before_post(self):
        import json,io,contextlib
        import kda_trial_benchmark as runner
        health={"status":"ok","engine":"ExLlamaV3","cache_tokens":393216,
                "context_length":392960,"speculative_method":"dflash2","draft_num_tokens":7,
                "busy":False,"model":"GLM-5.3-Flash"}
        for models in (["other"],["glm","GLM-5.3-Flash"]):
            with self.subTest(models=models),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);fixture=root/"fixture.json"
                fixture.write_text(json.dumps({"cases":[{"id":str(i),"request":{"model":model,
                    "stream":False,"max_tokens":128}} for i,model in enumerate(models)]}))
                calls=[]
                def fake_http(base,path,timeout,payload=None):
                    calls.append(path)
                    return {"http_status":200,"body":dict(health),"wall_seconds":0.1}
                args=runner.parse_args(["--fixture",str(fixture),"--output",str(root/"raw.jsonl"),
                    "--summary",str(root/"summary.json"),"--condition","CPU-double","--repeats","1"])
                with patch.object(runner,"http_request",fake_http),contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.run_trial(args),1)
                self.assertEqual(calls,["/health"])
                summary=json.loads((root/"summary.json").read_text())
                self.assertEqual(summary["attempted_requests"],0)
                self.assertEqual(summary["failure"]["phase"],"health_before")
                self.assertIn("model",summary["failure"]["mismatches"])

    def test_launcher_imports_api_sibling_from_unrelated_directory(self):
        import json
        import launch_actual_kda as launcher
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            api_dir, other = root / "api", root / "other"
            api_dir.mkdir()
            other.mkdir()
            sibling_name = "kda_fake_sibling_9384"
            (api_dir / (sibling_name + ".py")).write_text("marker = 'sibling-loaded'\n")
            api = api_dir / "fake_api.py"
            api.write_text("import json, os, sys\nfrom pathlib import Path\n"
                f"from {sibling_name} import marker\n"
                "Path(__file__).with_suffix('.json').write_text(json.dumps({'marker': marker, "
                "'path': sys.path, 'pythonpath': os.environ['PYTHONPATH']}))\n")
            engine, native_dir = root / "engine", root / "native"
            engine.mkdir()
            native_dir.mkdir()
            native = native_dir / "extension.so"
            native.write_bytes(b"fake")
            argv = ["launch_actual_kda.py", "--engine-root", str(engine), "--api", str(api),
                    "--api-sha256", "fake", "--native", str(native), "--native-sha256", "fake",
                    "--trace-directory", str(root / "trace"), "--"]
            old_directory = Path.cwd()
            try:
                os.chdir(other)
                with patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)), \
                     patch.dict(os.environ, {"GLM53_ACTUAL_KDA_CONFIG": "", "PYTHONPATH": "existing-tail"}), \
                     patch.object(launcher, "check_config"), patch.object(launcher, "install_config"):
                    launcher.main()
                    data = json.loads(api.with_suffix(".json").read_text())
                    self.assertEqual(data["marker"], "sibling-loaded")
                    expected = [str(Path(launcher.__file__).resolve().parent), str(native_dir.resolve()),
                                str(engine.resolve()), str(api_dir.resolve())]
                    self.assertEqual(data["path"][:4], expected)
                    self.assertEqual(data["pythonpath"].split(os.pathsep), expected + ["existing-tail"])
            finally:
                os.chdir(old_directory)
                sys.modules.pop(sibling_name, None)

    def test_cloned_calls_restore_original_pointers_before_forward(self):
        f = Fixture()
        out = f.forward(f.layer,f.x,f.params)
        self.assertTrue(torch.equal(out,f.x))
        self.assertEqual([call[-1] for call in f.bc.calls],[False,True,True])
        self.assertIs(f.bc.calls[-1][2],f.conv)
        self.assertIs(f.bc.calls[-1][3],f.recurrent)
        for call in f.bc.calls[:2]:
            self.assertNotEqual(call[2].data_ptr(),f.conv.data_ptr())
            self.assertNotEqual(call[3].data_ptr(),f.recurrent.data_ptr())
        self.assertTrue(f.rows[0]["original_storage_and_contents_unchanged"])
        self.assertTrue(f.rows[0]["original_forward_completed"])

    def test_history_index_uses_selected_slot_and_first_intermediate(self):
        for slot in (0,1,2):
            with self.subTest(slot=slot):
                f = Fixture(slot=slot)
                f.forward(f.layer,f.x,f.params)
                self.assertEqual(f.rows[0]["slot"],slot)
                self.assertTrue(f.rows[0]["q1_canonical_vs_q8_first_intermediate"]["exact"])

    def test_layer_instance_lookup_matches_deployed_forward(self):
        f = Fixture()
        f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.lookups,[(3,2)])

    def test_exported_worker_state_is_refused_in_ls_scope(self):
        f = Fixture(exported=True)
        with self.assertRaisesRegex(hook.Refusal,"exported TP"):
            f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.lookups,[])
        self.assertEqual(f.original_calls,0)

    def test_once_per_layer(self):
        f = Fixture()
        with torch.inference_mode():
            f.forward(f.layer,f.x,f.params)
            f.forward(f.layer,f.x,f.params)
        self.assertEqual(len(f.rows),1)
        self.assertEqual(f.original_calls,2)
        self.assertEqual(len(f.bc.calls),4)

    def test_all_buffer_shapes_and_axis(self):
        f = Fixture()
        f.forward(f.layer,f.x,f.params)
        row = f.rows[0]
        self.assertEqual(set(row["buffers"]),set(hook.BUFFER_ORDER))
        for tag,data in row["buffers"].items():
            self.assertTrue(data["bitwise_exact"],tag)
            axis = 2 if tag=="s_mqkv" else 1
            self.assertEqual(data["q8_metadata"]["sequence_axis"],axis)
            self.assertEqual(data["q8_metadata"]["shape"][axis],8)
            self.assertEqual(data["q8_metadata"]["first_row_shape"][axis],1)

    def test_projection_divergence_is_reported_without_quality_claim(self):
        f = Fixture();f.bc.divergent="s_qkv"
        f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.rows[0]["first_non_scratch_difference"],"s_qkv")
        self.assertEqual(f.rows[0]["buffers"]["s_qkv"]["max_abs_finite"],1.0)

    def test_unused_scratch_difference_is_not_a_stage_cause(self):
        f = Fixture();f.bc.divergent="s_o_xh"
        f.forward(f.layer,f.x,f.params)
        self.assertIsNone(f.rows[0]["first_non_scratch_difference"])
        self.assertTrue(f.rows[0]["buffers"]["s_o_xh"]["scratch_activity_unproven"])

    def test_original_state_mutation_refuses_before_original_forward(self):
        f = Fixture();f.bc.mutation=lambda:f.conv.add_(1)
        with self.assertRaisesRegex(hook.Refusal,"changed original"):
            f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.original_calls,0)
        self.assertEqual(f.rows[-1]["event"],"kda_probe_error")

    def test_original_storage_replacement_refuses_even_equal_values(self):
        f = Fixture();f.bc.mutation=lambda:setattr(f,"conv",f.conv.clone())
        with self.assertRaisesRegex(hook.Refusal,"changed original"):
            f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.original_calls,0)

    def test_budget_failure_precedes_cloned_native_calls(self):
        f = Fixture(budget=1)
        with self.assertRaisesRegex(hook.Refusal,"budget"):
            f.forward(f.layer,f.x,f.params)
        self.assertEqual(f.bc.calls,[])

    def test_invalid_slot_history_or_dtype_refuses(self):
        for kind in ("slot","dtype","history"):
            with self.subTest(kind=kind):
                f = Fixture()
                if kind=="slot":f.params["recurrent_slots"][0]=3
                if kind=="dtype":f.params["recurrent_slots"]=f.params["recurrent_slots"].long()
                if kind=="history":f.recurrent=f.recurrent[:,:7].clone()
                with self.assertRaises(hook.Refusal): f.forward(f.layer,f.x,f.params)
                self.assertEqual(f.bc.calls,[])

    def test_prefill_no_history_no_prior_state_and_unfilled_are_skipped(self):
        for condition in ("prefill","history","state","filled","length","bc","kda"):
            with self.subTest(condition=condition):
                f = Fixture()
                if condition=="prefill":f.params["prefill"]=True
                if condition=="history":f.params["recurrent_history"]=False
                if condition=="state":f.params["recurrent_states"]=[]
                if condition=="filled":f.layer.ba_weight_filled=False
                if condition=="length":f.x=torch.zeros((1,9,6),dtype=torch.half)
                if condition=="bc":f.layer.bc_split=False
                if condition=="kda":f.layer.kda=False
                f.probe.request_phase.context={"phase":"test_request"}
                self.assertFalse(f.probe.eligible(f.layer,f.x,f.params))
                # An inert original demonstrates that the wrapper does not probe.
                result=f.probe.wrap(lambda *args:"ordinary")(f.layer,f.x,f.params)
                self.assertEqual(result,"ordinary")
                self.assertEqual(f.rows,[])

    def test_generator_pin_failure_precedes_loader_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for relative in hook.SOURCE_PINS:
                path=root/relative; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text("same")
            pins={name:hook.sha(root/name) for name in hook.SOURCE_PINS}
            pins["exllamav3/generator/generator.py"]="bad"
            called=[]
            with patch.object(hook,"SOURCE_PINS",pins):
                with self.assertRaisesRegex(hook.Refusal,"generator/generator.py"):
                    hook.install_config({"engine_root":str(root)},loader=lambda:called.append("torch"))
            self.assertEqual(called,[])

    def test_install_wires_both_classes_and_loader_stays_inactive(self):
        import json
        f=Fixture()
        class GDN:
            forward=staticmethod(lambda *args:"loader-forward")
        class Generator:
            def iterate(self): return GDN.forward(None,None,{},None)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            api,native=root/"api.py",root/"native.so"
            api.write_text("api");native.write_text("native")
            config={"engine_root":str(root),"api_path":str(api),"api_sha256":hook.sha(api),
                    "native_path":str(native),"native_sha256":hook.sha(native),
                    "max_layers":34,"max_clone_bytes":80*1024**2,"trace_directory":str(root)}
            with patch.object(hook,"SOURCE_PINS",{}),patch.object(hook,"_installed",None):
                probe=hook.install_config(config,loader=lambda:(torch,None,GDN,f.cache,
                    lambda params,key,device:params[key],Generator))
                self.assertIs(GDN._glm_actual_kda_probe,probe)
                self.assertIs(Generator._glm_actual_kda_probe,probe)
                # A direct loader call has no context; avoid dereferencing dummy layer/input.
                self.assertEqual(GDN.forward(None,None,{},None),"loader-forward")
                self.assertEqual(probe.seen,set())
            header=json.loads(next(root.glob("kda-*.jsonl")).read_text().splitlines()[0])
            self.assertEqual(header["schema"],3)
            self.assertEqual(header["target_mode"],"layer_split_only")
            import io
            for cell in probe.writer.__closure__:
                if isinstance(cell.cell_contents,io.TextIOBase): cell.cell_contents.close()

    def test_byte_mutation_check_accepts_unchanged_nan_bits(self):
        x=torch.tensor([float("nan")])
        self.assertTrue(hook.byte_equal(torch,x,x.clone()))
        result=hook.metrics(torch,x,x.clone())
        self.assertTrue(result["bitwise_exact"])
        self.assertEqual(result["nonfinite_pairs"],1)

    def test_pin_failure_precedes_loader_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'exllamav3/modules').mkdir(parents=True)
            (root/'exllamav3/modules/gated_delta_net.py').write_text("bad")
            called=[]
            config={"engine_root":str(root)}
            with self.assertRaisesRegex(hook.Refusal,"source pin"):
                hook.install_config(config,loader=lambda:called.append("torch"))
            self.assertEqual(called,[])

    def test_native_pin_failure_precedes_loader_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            from unittest.mock import patch
            with patch.object(hook,"SOURCE_PINS",{}):
                (root/'api.py').write_text("api");(root/'native.so').write_text("native")
                config={"engine_root":str(root),"api_path":str(root/'api.py'),
                    "api_sha256":hook.sha(root/'api.py'),"native_path":str(root/'native.so'),"native_sha256":"bad"}
                called=[]
                with self.assertRaisesRegex(hook.Refusal,"native pin"):
                    hook.install_config(config,loader=lambda:called.append("torch"))
                self.assertEqual(called,[])

    def test_no_cuda_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_sitecustomize_is_inert_without_explicit_config(self):
        env={k:v for k,v in os.environ.items() if k!="GLM53_ACTUAL_KDA_CONFIG"}
        env["PYTHONPATH"]=str(Path(__file__).resolve().parent)
        result=subprocess.run([sys.executable,"-c","import sys; assert 'torch' not in sys.modules"],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_worker_pin_failure_is_fatal_before_torch_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'exllamav3/modules').mkdir(parents=True)
            (root/'exllamav3/modules/gated_delta_net.py').write_text("bad")
            import json
            config=root/'config.json';config.write_text(json.dumps({"engine_root":str(root)}))
            env={**os.environ,"GLM53_ACTUAL_KDA_CONFIG":str(config),"PYTHONPATH":str(Path(__file__).resolve().parent)}
            result=subprocess.run([sys.executable,"-c","raise SystemExit(99)"],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,78,result.stderr)
            self.assertIn("source pin differs",result.stderr)

    def test_source_buffer_shapes_match_actual_pinned_method(self):
        source_root=Path(os.environ.get("EXLLAMAV3_ENGINE_ROOT", Path(__file__).resolve().parents[1]/"upstream_publication_implementation/deployed_native"))
        path=source_root/"exllamav3/modules/gated_delta_net.py"
        self.assertEqual(hook.sha(path),hook.SOURCE_PINS["exllamav3/modules/gated_delta_net.py"])
        import ast
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="GatedDeltaNet")
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_bc_configure_slot_kda")
        namespace={"g_tensor_cache":Cache(),"torch":torch}
        # Execute the actual method with a buffer cache exposing get(), then compare descriptors.
        def get(device,shape,dtype,tag):
            key=namespace["g_tensor_cache"].make_key(device,shape,dtype,tag)
            tensor=torch.empty(shape,dtype=dtype)
            namespace["g_tensor_cache"].cache[key]=(1,tensor)
            return tensor
        namespace["g_tensor_cache"].get=get
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),"exec"),namespace)
        f=Fixture();f.layer.bc.configure_slot_kda=lambda *args:None
        for length in (1,8):
            namespace["_bc_configure_slot_kda"](f.layer,1,length,length==8)
            for tag,(shape,dtype,axis) in hook.buffer_specs(f.layer,torch,length).items():
                key=namespace["g_tensor_cache"].make_key(f.layer.device,shape,dtype,tag)
                self.assertIn(key,namespace["g_tensor_cache"].cache)


if __name__ == "__main__":
    unittest.main()
