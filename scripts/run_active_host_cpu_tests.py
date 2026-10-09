"""Run default-off active-host stdlib contracts without engine/Torch imports."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, required=True)
    args = parser.parse_args()
    adapter = Path(__file__).resolve().parent.parent / "experimental/active_host"
    env = os.environ.copy()
    env["EXLLAMAV3_ENGINE_ROOT"] = str(args.engine_root.resolve(strict=True))
    env["CUDA_VISIBLE_DEVICES"] = ""
    program = """import unittest,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd().parents[1]/'glm'))
before='torch' in sys.modules
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('.'))
after='torch' in sys.modules
print(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),
                  'torch_imported_before':before,'torch_imported_after':after}))
raise SystemExit(not result.wasSuccessful() or before or after)
"""
    return subprocess.run([sys.executable, "-c", program], cwd=adapter, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
