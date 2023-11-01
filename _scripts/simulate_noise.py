import itertools
import time
import os

import numpy as np
import pandas as pd
import progressbar

from utils.helper_params import load_parameters

from poly_certificate.certificate import get_certificate, get_rho_and_lambdas
from poly_certificate.gauss_newton import gauss_newton
from poly_certificate.problem import Problem

# 0: none
# 1: only progress output
# 2: debugging mode
VERBOSE = 2

# if the relative cost difference between a solution's cost and the minimum cost for the same setup
# is bigger than TOL_GLOBAL, we consider the solution to be local
TOL_GLOBAL = 1e-2

# use exact Hessian in Gauss-Newton (effectively turning it into a Newton solver)
EXACT_HESS = False

REG = 1e-10

SAVE_RESULTS = True


def generate_results(params_dir, out_dir, save_results=SAVE_RESULTS):

    params = load_parameters(params_dir, out_dir)
    fname = os.path.join(out_dir, params_dir, "results.pkl")

    n_inits = params["init_seed"]
    n_setups = params["setup_seed"]
    init_seeds = np.arange(n_inits)
    setup_seeds = np.arange(n_setups)

    results_columns = [
        "setup seed",
        "init seed",
        "noise",
        "error",
        "regularization",
        "sigma acc est",
        "sigma dist real",
        "cert value",
        "cost",
        "rho",
        "time",
    ]

    # n_results = len(noises) * len(n_its) * n_starts
    results = pd.DataFrame(columns=results_columns)
    for regularization in params["regularization"]:
        if VERBOSE > 0:
            print(f"----- regularization: {regularization} ------")

        for sigma_acc_est, sigma_dist_real in itertools.product(
            params["sigma_acc_est"], params["sigma_dist_real"]
        ):
            if VERBOSE > 0:
                print(
                    f"---- sigma dist:{sigma_dist_real:.1e} sigma acc:{sigma_acc_est:.1e} ----"
                )
                p = progressbar.ProgressBar(max_value=n_setups)

            for i, setup_seed in enumerate(setup_seeds):
                sigma_dist_est = (
                    params["sigma_dist_est"]
                    if params["sigma_dist_est"] is not None
                    else sigma_dist_real
                )
                prob = Problem(
                    params["N"],
                    params["d"],
                    params["K"],
                    sigma_acc_est=sigma_acc_est,
                    sigma_dist_est=sigma_dist_est,
                )
                np.random.seed(setup_seed)
                prob.generate_random(
                    sigma_dist_real=sigma_dist_real,
                    sigma_acc_real=params["sigma_acc_real"],
                )
                noise_realization = prob.calculate_noise_level()

                results_here = pd.DataFrame(
                    index=range(n_inits), columns=results_columns
                )
                for init_seed in init_seeds:

                    np.random.seed(init_seed)
                    theta_0 = prob.random_traj_init(regularization)

                    t1 = time.time()
                    theta_est, stats = gauss_newton(
                        theta_0,
                        prob,
                        regularization=regularization,
                        exact_hess=EXACT_HESS,
                    )
                    result_dict = {
                        "setup seed": setup_seed,
                        "init seed": init_seed,
                        "noise": noise_realization,
                        "regularization": regularization,
                        "sigma acc est": sigma_acc_est,
                        "sigma dist real": sigma_dist_real,
                        "error": None,
                        "cert value": None,
                        "cost": None,
                        "rho": None,
                        "time": None,
                    }
                    if stats["success"]:

                        trajectory_est = theta_est[:, : prob.d]
                        cost = stats["cost"]

                        rmse = prob.get_rmse(trajectory_est)

                        # old certificate computation
                        rho, lamdas = get_rho_and_lambdas(
                            theta_est,
                            prob,
                            regularization=regularization,
                        )

                        cert_value = get_certificate(
                            prob, rho, lamdas, regularization=regularization, reg=REG
                        )

                        if (VERBOSE > 1) and (cert_value < 0):
                            print("\n", stats["status"])
                            print(
                                f"cert at init {init_seed} setup {setup_seed}"
                                + f"rmse: {rmse:.1e}, cert: {cert_value:.2e}, rho:{rho:.1e}"
                            )
                        ttot = time.time() - t1
                        result_dict.update(
                            {
                                "error": rmse,
                                "cert value": cert_value,
                                "cost": cost,
                                "rho": rho,
                                "time": ttot,
                            }
                        )
                    elif VERBOSE:
                        print(
                            f"\n GN did not converge at noise level: {sigma_dist_real}"
                        )
                    results_here.loc[init_seed, :] = result_dict

                p.update(i + 1)
                # endfor inits

                # label global vs. local errors.
                min_cost = results_here.cost.min()
                rel_error = (results_here.cost - min_cost) / min_cost
                results_here.loc[
                    rel_error < TOL_GLOBAL,
                    "solution",
                ] = "global"
                results_here.loc[
                    rel_error >= TOL_GLOBAL,
                    "solution",
                ] = "local"
                results = pd.concat([results, results_here], ignore_index=True)

            # endfor it
            if save_results:
                results.to_pickle(fname)
                print("saved intermediate as", fname)
    return results


if __name__ == "__main__":
    from utils.helper_params import logs_to_file

    out_dir = "_results/"
    logfile = os.path.join(out_dir, "simulation_noise.log")
    with logs_to_file(logfile):
        params_dir = "simulation_noise/"
        print(f"Running experiment {params_dir}")
        generate_results(params_dir, out_dir=out_dir)
