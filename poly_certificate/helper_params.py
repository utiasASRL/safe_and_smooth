def read_json(filename):
    import json

    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_parameters(params_dir, out_dir, default_file="default.json"):
    import os

    default_file = os.path.join(out_dir, default_file)
    if os.path.exists(default_file):
        params = read_json(default_file)
    else:
        params = {}

    # sys.path.insert(0, os.path.join(out_dir, params_dir))
    params_file = os.path.join(out_dir, params_dir, "params.json")
    params.update(read_json(params_file))

    return params


def parse_log_argument(description=""):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-l", "--logging", help="turn logging on", action="store_true", default=False
    )
    args = parser.parse_args()
    return args.logging
