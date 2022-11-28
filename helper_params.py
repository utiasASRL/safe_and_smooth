def load_parameters(params_dir, out_dir, default_file="default.py"):
    from importlib import import_module
    import os
    import sys

    params_file = "params.py"
    if os.path.exists(os.path.join(out_dir, default_file)):
        sys.path.append(out_dir)
        default_mod = import_module(default_file.split(".")[0])
        params = default_mod.params
    else:
        params = {}

    sys.path.append(os.path.join(out_dir, params_dir))
    params_mod = import_module(params_file.split(".")[0])
    params.update(params_mod.params)
    return params


def parse_log_argument(description=""):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-l", "--logging", help="turn logging on", action="store_true", default=False
    )
    args = parser.parse_args()
    return args.logging
