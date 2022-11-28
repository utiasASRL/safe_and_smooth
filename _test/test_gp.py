"""
Gaussian Process interpolation tests (WIP)
"""

import numpy as np

import sys
from os.path import dirname

sys.path.append(dirname(__file__) + "/../")

from gaussian_process import *
from gauss_newton import gauss_newton
from problem import Problem


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

    from gauss_newton import setup_gp
    import scipy.sparse.linalg as spl

    regularization = "constant-velocity"
    k = 4
    A_inv, Q_inv, K_prior_inv, v = setup_gp(prob, regularization)

    np.testing.assert_allclose(
        (A_inv.T @ Q_inv @ A_inv).toarray(), K_prior_inv.toarray()
    )

    K_prior_ii, K_prior_ij = get_prior_covariances(prob, regularization)

    import scipy.sparse as sp

    K_prior = spl.inv(K_prior_inv + 1e-3 * sp.identity(K_prior_inv.shape[0]))
    for i in range(prob.N):
        K_prior_ii_est = K_prior[i * k : (i + 1) * k, i * k : (i + 1) * k].toarray()
        # np.testing.assert_allclose(K_prior_ii_est, K_prior_ii[i])


if __name__ == "__main__":
    test_covariances()
    test_query()
