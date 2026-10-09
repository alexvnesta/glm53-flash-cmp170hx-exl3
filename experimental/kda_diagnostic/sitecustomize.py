"""Explicit opt-in hook propagation to this trial's spawned Python workers."""
import json
import os
import sys

config_path = os.environ.get("GLM53_ACTUAL_KDA_CONFIG", "")
if config_path:
    try:
        from kda_actual_hook import install_config
        with open(config_path) as stream:
            install_config(json.load(stream))
    except BaseException as error:
        # Python normally ignores sitecustomize failures. An owned diagnostic
        # worker must fail closed rather than silently run without its pin fence.
        print(f"Actual KDA diagnostic startup refused: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        os._exit(78)
