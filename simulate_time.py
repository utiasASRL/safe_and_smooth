import os
import sys
import time

import numpy as np
import pandas as pd

from certificate import get_certificate
from certificate import get_rho_and_lambdas
from helper_params import parse_log_argument
from gauss_newton import gauss_newton
from problem import Problem

regularization = "constant-velocity"

if __name__ == "__main__":

    results_name = "simulation_time.pkl"

    out_dir = "_results"
    logging = parse_log_argument(description="Run timing experiments.")
    if logging:
        old_stdout = sys.stdout
        logfile = os.path.join(out_dir, "simulate_time.log")
        f = open(logfile, "w")
        sys.stdout = f

    fname = os.path.join(out_dir, results_name)
    max_N = 6
    Ns = np.logspace(1, max_N, max_N * 2 - 1).astype(int)

    print("generating Ns:", Ns)
    results = pd.DataFrame(columns=["N", "time", "method", "status"])

    np.random.seed(0)
    for i, N in enumerate(Ns):
        print(f"solving with {N}...")
        times_dict = {}

        prob = Problem(N, d=2, K=7)
        prob.generate_random()
        theta_0 = np.random.rand(N, 2 * prob.d)

        # converge only to tol1e-5 to save time
        theta_hat, stats = gauss_newton(theta_0, prob, regularization=regularization, tol=1e-5)

        # makes sure to only save time for actual solve,
        # and not time to generate other data used for
        # plotting / debugging.
        times_dict["solve"] = stats["time"]  # time.time() - t1

        if theta_hat is None:
            continue

        t1 = time.time()
        rho, lamdas = get_rho_and_lambdas(
            theta_hat,
            prob,
            regularization=regularization,
        )
        times_dict["duals"] = time.time() - t1

        t1 = time.time()
        cert_value = get_certificate(
            prob,
            rho,
            lamdas,
            regularization=regularization,
        )
        times_dict["certificate"] = time.time() - t1

        for method, t in times_dict.items():
            results.loc[len(results), :] = {
                "N": N,
                "time": t,
                "method": method,
                "status": stats["status"],
            }

        results.to_pickle(fname)
        print("saved as", fname)
    print("done")

    if logging:
        sys.stdout = old_stdout
        f.close()
