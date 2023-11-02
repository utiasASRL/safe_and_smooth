""" Evaluate range-only drone localization dataset.

This code requires access to the drone UWB dataset referred to in [1].
While waiting for the dataset to be published, please contact the authors to get access.

"""
import itertools
import time
import os

import numpy as np
import pandas as pd

# from cert_matrix import get_centered_matrix, get_original_matrix
# from certificate import get_mineig_sparse
from poly_certificate.certificate import get_certificate, get_rho_and_lambdas
from poly_certificate.gauss_newton import gauss_newton
from poly_certificate.problem import Problem
from poly_certificate.datasets import ANCHOR_CHOICE

from utils.helper_params import load_parameters

DEFAULT_FILE = "default_real.json"


def evaluate_datasets(
    params_dir, out_dir, save_results=True, calibrate=True, test=False
):
    """
    Detect local minima using a subset of anchors and
    a fixed regularization parameter.
    """
    print("running results", os.path.join(out_dir, params_dir))
    # from sdp_setup import get_f, get_prob_matrices

    fname = os.path.join(out_dir, params_dir, "results.pkl")
    params = load_parameters(params_dir, out_dir, default_file=DEFAULT_FILE)

    datasets = params["datasets"]
    if test:
        datasets = datasets[:1]

    # do not change:
    use_gt = False
    n_random = 1

    data = []
    for dataset_i in datasets:
        if type(dataset_i) is int:
            dataset = f"trial{dataset_i}"
        else:
            dataset = dataset_i

        print(f"\n ========= \ndataset: {dataset}")
        for use_anchors, sigma_acc_est in itertools.product(
            params["use_anchors_list"], params["sigma_acc_est_list"]
        ):
            print("anchors:", use_anchors)
            print("sigma:", sigma_acc_est)
            prob = Problem.init_from_dataset(
                dataset,
                time_range=params["time_range"],
                sigma_dist_est=params["sigma_dist_est"],
                sigma_acc_est=sigma_acc_est,
                use_gt=use_gt,
                use_anchors=ANCHOR_CHOICE[use_anchors],
                calibrate=calibrate,
            )

            for regularization in params["regularization"]:
                print(f"regularization: {regularization}")

                for init in params["inits"]:
                    n_init = n_random if init == "random" else 1
                    for n in range(n_init):
                        print(f"{init} {n+1}/{n_init}")
                        np.random.seed(n)

                        results_dict = {
                            "dataset": dataset,
                            "regularization": regularization,
                            "init": init,
                            "seed_init": n,
                            "sigma": sigma_acc_est,
                        }

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

                        theta_est, stats = gauss_newton(
                            theta_0,
                            prob,
                            regularization=regularization,
                            progressbar=True,
                            verbose=False,
                        )

                        if not stats["success"]:
                            print("Warning: Gauss-Newton did not converge.")
                            # print(stats)

                        # original LDL certificate
                        t1 = time.time()
                        rho, lambdas = get_rho_and_lambdas(
                            theta_est,
                            prob,
                            regularization=regularization,
                        )
                        assert abs(rho + stats["cost"]) / rho < 1e-4
                        cert = get_certificate(
                            prob,
                            rho,
                            lambdas,
                            regularization=regularization,
                        )
                        time_cert = time.time() - t1
                        traj_hat = theta_est[:, : prob.d]
                        rmse = prob.get_rmse(traj_hat)

                        results_dict.update(
                            {
                                "RMSE": rmse,
                                "MAE": prob.get_mae(traj_hat),
                                "RMSE unbiased": prob.get_rmse_unbiased(traj_hat),
                                "error_xyz": prob.get_mean_error_xyz(traj_hat),
                                "var_xyz": prob.get_var_error_xyz(traj_hat),
                                "cert": cert,
                                "cost": stats["cost"],
                                "cost data": stats["cost dist"],
                                "cost prior": stats["cost reg"],
                                "time solve": stats["time"],
                                "time cert": time_cert,
                            }
                        )
                        if params["save_estimate"]:
                            results_dict.update(
                                {
                                    "estimate": traj_hat,
                                    "ground truth": prob.trajectory,
                                    "inverse covariance": stats["cov"],
                                }
                            )

                        data.append(results_dict)
                        print(f"RMSE: {rmse:.2f}, cert:{cert}")
                if save_results:
                    results = pd.DataFrame(data)
                    results.to_pickle(fname)
                    print("saved intermediate as", fname)
    results = pd.DataFrame(data)
    return results


if __name__ == "__main__":
    from utils.helper_params import logs_to_file, parse_arguments

    save_results = True
    args = parse_arguments("Generate dataset results.")
    out_dir = args.resultdir

    logfile = os.path.join(out_dir, "evaluate_real.log")
    with logs_to_file(logfile):
        params_dir = "real_top_estimate/"
        evaluate_datasets(
            params_dir,
            out_dir,
            calibrate=True,
            save_results=save_results,
            test=args.test,
        )

        params_dir = "real_top_calib/"
        evaluate_datasets(
            params_dir,
            out_dir,
            calibrate=True,
            save_results=save_results,
            test=args.test,
        )

        params_dir = "real_top/"
        evaluate_datasets(
            params_dir,
            out_dir,
            calibrate=True,
            save_results=save_results,
            test=args.test,
        )
