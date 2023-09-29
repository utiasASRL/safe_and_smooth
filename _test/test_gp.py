"""
Gaussian Process interpolation tests (WIP)
"""

import numpy as np
import scipy as sp
import scipy.sparse.linalg as spl

from poly_certificate.gaussian_process import *
from poly_certificate.gauss_newton import gauss_newton, setup_gp
from poly_certificate.problem import Problem


def test_query():
    np.random.seed(4)
    N = 2
    K = 3
    d = 2
    prob = Problem(N=N, K=K, d=d)
    prob.generate_random()

    regularization = "constant-velocity"
    theta_0 = prob.gt_init(regularization)
    theta_hat, stats = gauss_newton(theta_0, prob, regularization=regularization)

    K_post = stats["cov"]
    K_prior_ii, K_prior_ij = get_prior_covariances(prob, regularization)
    for tau in np.arange(prob.times[0], prob.times[-1], step=0.1):
        x_tau, K_tau = query_trajectory(
            tau,
            prob,
            theta_hat,
            regularization=regularization,
            return_covariance=True,
            K_prior_ii=K_prior_ii,
            K_prior_ij=K_prior_ij,
            K_post=K_post,
        )
        # assert np.all(np.linalg.eigvalsh(K_tau) >= 1e-10)


def test_covariances():
    np.random.seed(4)
    N = 2
    K = 3
    d = 2
    prob = Problem(N=N, K=K, d=d)
    prob.generate_random()

    regularization = "constant-velocity"
    k = 4
    A_inv, Q_inv, K_prior_inv, v = setup_gp(prob, regularization)

    np.testing.assert_allclose(
        (A_inv.T @ Q_inv @ A_inv).toarray(), K_prior_inv.toarray()
    )


if __name__ == "__main__":
    test_covariances()
    test_query()
