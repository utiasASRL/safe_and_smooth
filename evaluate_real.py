""" Evaluate real dataset. This code requires access to the drone UWB dataset referred to in [1]. 
While waiting for the dataset to be published, please contact the authors to get access. 
"""
import itertools
import time
import os

import numpy as np
import pandas as pd

from certificate import get_certificate, get_rho_and_lambdas
from gauss_newton import gauss_newton
from helper_params import load_parameters, parse_log_argument
from problem import Problem

ANCHOR_CHOICE = {"top": [0, 2, 3, 5], "diagonal": [0, 1, 3, 4], "all": range(6)}
DEFAULT_FILE = "default_real.json"


def evaluate_datasets(params_dir, out_dir, save_results=True, calibrate=True):
    """
    Detect local minima using a subset of anchors and
    a fixed regularization parameter.
    """
    print("running results", os.path.join(out_dir, params_dir))
    # from sdp_setup import get_f, get_prob_matrices

    fname = os.path.join(out_dir, params_dir, "results.pkl")
    params = load_parameters(params_dir, out_dir, default_file=DEFAULT_FILE)

    # do not change:
    use_gt = False
    n_random = 1

    time_range = params["time_range"]
    save_estimate = params["save_estimate"]
    sigma_dist_est = params["sigma_dist_est"]

    datasets = params["datasets"]
    inits = params["inits"]
    use_anchors_list = params["use_anchors_list"]
    sigma_acc_est_list = params["sigma_acc_est_list"]

    columns = [
        "dataset",
        "sigma",
        "regularization",
        "init",
        "seed init",
        "RMSE",
        "MAE",
        "RMSE unbiased",
        "error_xyz",
        "var_xyz",
        "cost",
        "cost data",
        "cost prior",
        "certificate",
        "time solve",
        "time cert",
    ]
    if save_estimate:
        columns += ["estimate", "ground truth", "inverse covariance"]

    results = pd.DataFrame(columns=columns)

    for dataset_i in datasets:
        if type(dataset_i) == int:
            dataset = f"trial{dataset_i}"
        else:
            dataset = dataset_i

        print(f"\n ========= \ndataset: {dataset}")
        for use_anchors, sigma_acc_est in itertools.product(
            use_anchors_list, sigma_acc_est_list
        ):
            print("anchors:", use_anchors)
            print("sigma:", sigma_acc_est)
            prob = Problem.init_from_dataset(
                dataset,
                time_range=time_range,
                sigma_dist_est=sigma_dist_est,
                sigma_acc_est=sigma_acc_est,
                use_gt=use_gt,
                use_anchors=ANCHOR_CHOICE[use_anchors],
                calibrate=calibrate,
            )

            for regularization in params["regularization"]:
                print(f"regularization: {regularization}")

                for init in inits:
                    n_init = n_random if init == "random" else 1
                    for n in range(n_init):
                        print(f"{init} {n+1}/{n_init}")
                        np.random.seed(n)

                        if init == "random":
                            # generate random trajectory
                            theta_0 = prob.random_traj_init(
                                regularization, sigma_acc_est=sigma_acc_est
                            )
                            # generate random points
                            # theta_0 = prob.random_traj(
                            #    regularization, sigma_acc_est=0.0
                            # )
                        elif init == "flip":
                            theta_0 = prob.reflected_init(regularization, fraction=1.0)
                        elif init == "half-flip":
                            theta_0 = prob.reflected_init(regularization, fraction=0.5)
                        elif init == "gt":
                            theta_0 = prob.gt_init(regularization)

                        theta_hat, stats = gauss_newton(
                            theta_0,
                            prob,
                            regularization=regularization,
                            progressbar=True,
                            verbose=False,
                        )

                        if not stats["success"]:
                            print("Warning: Gauss-Newton did not converge.")
                            # print(stats)

                        t1 = time.time()
                        rho, lambdas = get_rho_and_lambdas(
                            theta_hat,
                            prob,
                            regularization=regularization,
                        )
                        assert abs(rho + stats["cost"]) / rho < 1e-4

                        cert = get_certificate(
                            prob,
                            rho,
                            lambdas,
                            regularization=regularization,
                            verbose=False,
                        )
                        time_cert = time.time() - t1

                        traj_hat = theta_hat[:, : prob.d]
                        rmse = prob.get_rmse(traj_hat)
                        mae = prob.get_mae(traj_hat)
                        rmse_unbiased = prob.get_rmse_unbiased(traj_hat)
                        error_xyz = prob.get_mean_error_xyz(traj_hat)
                        var_xyz = prob.get_var_error_xyz(traj_hat)

                        results_dict = {
                            "dataset": dataset,
                            "regularization": regularization,
                            "init": init,
                            "seed_init": n,
                            "RMSE": rmse,
                            "MAE": mae,
                            "RMSE unbiased": rmse_unbiased,
                            "error_xyz": error_xyz,
                            "var_xyz": var_xyz,
                            "sigma": sigma_acc_est,
                            "cost": stats["cost"],
                            "cost data": stats["cost dist"],
                            "cost prior": stats["cost reg"],
                            "certificate": cert,
                            "time solve": stats["time"],
                            "time cert": time_cert,
                        }
                        if save_estimate:
                            results_dict.update(
                                {
                                    "estimate": traj_hat,
                                    "ground truth": prob.trajectory,
                                    "inverse covariance": stats["cov"],
                                }
                            )
                        results.loc[len(results), :] = results_dict
                        print(f"RMSE: {rmse:.2f}, cert:{cert}")
                if save_results:
                    results.to_pickle(fname)
                    print("saved intermediate as", fname)
    return results


if __name__ == "__main__":
    import sys

    save_results = True
    out_dir = "_results/"

    logging = parse_log_argument(description="Run simulation experiments.")
    if logging:
        old_stdout = sys.stdout
        logfile = os.path.join(out_dir, "evaluate_real.log")
        f = open(logfile, "w")
        sys.stdout = f


    params_dir = "real_top_estimate/"
    evaluate_datasets(params_dir, out_dir, calibrate=True, save_results=save_results)

    params_dir = "real_top_calib/"
    evaluate_datasets(params_dir, out_dir, calibrate=True, save_results=save_results)

    params_dir = "real_top/"
    evaluate_datasets(params_dir, out_dir, calibrate=True, save_results=save_results)

    if logging:
        sys.stdout = old_stdout
        f.close()
