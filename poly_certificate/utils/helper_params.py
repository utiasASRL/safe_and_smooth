import argparse
import json
import os
import sys
from datetime import datetime


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def load_parameters(params_dir, out_dir, default_file="default.json"):
    default_file = os.path.join(out_dir, default_file)
    if os.path.exists(default_file):
        params = read_json(default_file)
    else:
        params = {}

    # sys.path.insert(0, os.path.join(out_dir, params_dir))
    params_file = os.path.join(out_dir, params_dir, "params.json")
    params.update(read_json(params_file))

    return params


class Logger(object):
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        pass


class logs_to_file(object):
    def __init__(self, file_name=None):
        self.file_name = file_name

    def __enter__(self):
        self.logfile = open(self.file_name, "w+")
        print(f"Writing logs to {self.file_name}.")

        self.print_header()
        self.old_stdout = sys.stdout
        logger = Logger(sys.stdout, self.logfile)
        sys.stdout = logger

    def __exit__(self, *args):
        sys.stdout = self.old_stdout
        self.print_footer()
        self.logfile.close()

    def print_header(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.logfile.write(f"Logging data from {current_time}\n")

    def print_footer(self):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self.logfile.write(f"Finished logging at {current_time}\n\n")


def parse_arguments(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-t",
        "--test",
        help="turn testing on (use fewer parameters)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--resultdir",
        help="directory of results",
        default="_results_test",
    )
    parser.add_argument(
        "-p",
        "--plotdir",
        help="directory for plots",
        default="_plots_test",
    )
    args = parser.parse_args()
    return args
