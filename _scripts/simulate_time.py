import os
import time

import numpy as np
import pandas as pd

from poly_certificate.certificate import get_certificate
from poly_certificate.certificate import get_rho_and_lambdas
from poly_certificate.gauss_newton import gauss_newton
from poly_certificate.problem import Problem


def generate_results(fname, max_N):
    regularization = "constant-velocity"

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
        theta_hat, stats = gauss_newton(
            theta_0, prob, regularization=regularization, tol=1e-8
        )

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


if __name__ == "__main__":
    from utils.helper_params import logs_to_file, parse_arguments

    max_N = 6
    out_dir = "_results_final"

    args = parse_arguments("Perform timing study.")
    if args.test:
        max_N = 3
        out_dir = "_results"

    results_name = "simulation_time"

    logfile = os.path.join(out_dir, results_name)
    with logs_to_file(logfile + ".log"):
        fname = os.path.join(out_dir, results_name + ".pkl")
        generate_results(fname, max_N=max_N)
