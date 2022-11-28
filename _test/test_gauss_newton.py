import numpy as np

import sys
from os.path import dirname

sys.path.append(dirname(__file__) + "/../")

from gauss_newton import *
from problem import Problem, generate_distances

N = 10
d = 3
K = 4
regularization = "constant-velocity"
sigma_acc_est = 1e-3
sigma_dist_est = 1e-2
add_W = True  # False


def generate_problem(
    sigma_dist_real=sigma_dist_est, sigma_acc_real=sigma_acc_est, seed=0
):
    np.random.seed(seed)
    prob = Problem(N, d, K, sigma_acc_est=1e-3, sigma_dist_est=1e-2)
    prob.generate_random(sigma_acc_real=sigma_acc_real, sigma_dist_real=sigma_dist_real)
    if add_W:
        k = prob.theta.shape[1]
        prob.W = np.random.choice(
            [0, 1], replace=True, size=(prob.K, prob.N), p=[0.2, 0.8]
        )
        prob.E = np.sum(prob.W)
    return prob


def test_finite_differences(seed=0):
    prob = generate_problem(seed=seed)
    # prob.generate_random()  # np.random.rand(prob.N, prob.d * 2)
    # theta = prob.theta.copy()
    theta = np.random.rand(prob.N, prob.d * 2)
    fun_finite_differences(get_grad_hess_cost_f, theta, prob)
    fun_finite_differences(
        get_grad_hess_cost_r, theta, prob, regularization=regularization
    )

    k = theta.shape[1]
    eps = 1e-10
    for n in range(N):
        for i in range(k):
            eps_vec = np.zeros((N, k))
            eps_vec[n, i] = eps
            theta_eps = theta.copy() + eps_vec

            G_eps = generate_distances(theta_eps[:, :d], prob.anchors)  # K x N
            g_eps = (G_eps**1).flatten("F")

            G = generate_distances(theta[:, :d], prob.anchors)  # K x N
            g = (G**1).flatten("F")

            J = jacobian_distances(theta, prob.anchors, prob.W)  # K x N*1d

            theta_eps_flat = theta_eps.flatten()
            np.testing.assert_allclose(theta_eps_flat[:k], theta_eps[0, :])

            g_eps_est = g + J @ eps_vec.flatten()
            np.testing.assert_allclose(g_eps, g_eps_est)


def fun_finite_differences(fun, theta, prob, **kwargs):
    grad_f, cost_f = fun(theta, prob, return_cost=True, **kwargs)
    grad_f = grad_f.reshape((N, -1))

    eps = 1e-7
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            theta_eps = theta.copy()
            theta_eps[i, j] += eps

            __, cost_f_eps = fun(theta_eps, prob, return_cost=True, **kwargs)
            np.testing.assert_allclose(
                eps * grad_f[i, j], cost_f_eps - cost_f, atol=eps, rtol=eps
            )


def test_hess(seed=0):
    # assert that the cost at the real solution is zero.
    prob = generate_problem(sigma_dist_real=0.0, seed=seed)
    theta_gt = prob.theta
    grad_f, hess_f = get_grad_hess_cost_f(
        theta_gt, prob, return_cost=False, return_hess=True
    )
    grad_r, hess_r = get_grad_hess_cost_r(
        theta_gt,
        prob,
        regularization=regularization,
        return_cost=False,
        return_hess=True,
    )
    hess = hess_f + hess_r
    eigvalsh = np.linalg.eigvalsh(hess.toarray())
    print(eigvalsh)
    assert np.all(eigvalsh > -1e-10)


def test_cost(seed=0):
    # assert that the cost at the real solution is zero.
    prob = generate_problem(sigma_dist_real=0.0, seed=seed)
    theta_gt = prob.gt_init(regularization=regularization)

    grad_f, cost_f = get_grad_hess_cost_f(theta_gt, prob, return_cost=True)
    grad_r, cost_r = get_grad_hess_cost_r(
        theta_gt, prob, regularization=regularization, return_cost=True
    )
    assert cost_f == 0
    assert cost_r != 0
    np.testing.assert_almost_equal(grad_f, 0)

    prob = generate_problem(sigma_acc_real=0.0)
    theta_gt = prob.gt_init(regularization=regularization)

    grad_r, cost_r = get_grad_hess_cost_r(
        theta_gt, prob, regularization=regularization, return_cost=True
    )
    assert cost_r == 0
    np.testing.assert_almost_equal(grad_r, 0)

    # assert that the cost is minimal at the optimum.
    prob = generate_problem()
    theta_0 = prob.gt_init(regularization=regularization)
    theta_est, stats = gauss_newton(theta_0, prob, regularization=regularization)

    delta = 1e-3
    for i in range(theta_est.shape[0]):
        for j in range(theta_est.shape[1]):
            theta_delta = theta_est.copy()
            theta_delta[i, j] += delta

            __, cost_f = get_grad_hess_cost_f(theta_delta, prob, return_cost=True)
            __, cost_r = get_grad_hess_cost_r(
                theta_delta, prob, return_cost=True, regularization=regularization
            )
            assert cost_f + cost_r > stats["cost"]


def test_certificate(seed=0):
    # calculate cost at optimum and rho at optimum --> they should be equal.
    from certificate import get_rho_and_lambdas

    prob = generate_problem(seed=seed)
    theta_0 = prob.gt_init(regularization=regularization)
    theta_est, stats = gauss_newton(theta_0, prob, regularization=regularization)

    rho, lamdas = get_rho_and_lambdas(theta_est, prob, regularization)

    # TODO(FD) not sure if these are good enough errors
    np.testing.assert_allclose(stats["cost"], -rho, rtol=1e-3, atol=1e-5)


def test_optimizer(seed=0):
    """Test that the algorithm converges"""
    # make sure Gauss-Newton converges.
    prob = generate_problem(seed=seed)

    theta_0 = prob.random_init(regularization=regularization)
    theta_est, stats = gauss_newton(theta_0, prob, regularization=regularization)
    assert stats["success"]
    assert theta_est is not None


def test_gradients(seed=0):
    """Test that the gradients are ascent directions and that they are zero at the optimum."""
    prob = generate_problem(seed=seed)

    # test that gradient is direction of steepest ascent.
    delta = 1e-5
    for reg in ["no", "constant-velocity", "zero-velocity"]:
        if reg == "constant-velocity":
            k = 2 * d
        else:
            k = d

        theta = prob.random_init(regularization=reg)
        grad_f, cost_f = get_grad_hess_cost_f(
            theta,
            prob,
            return_cost=True,
        )
        grad_r, cost_r = get_grad_hess_cost_r(
            theta, prob, return_cost=True, regularization=reg
        )

        grad_f = grad_f.reshape((-1, k))
        grad_r = grad_r.reshape((-1, k))
        theta_new = theta + delta * grad_f
        *_, cost_f_new = get_grad_hess_cost_f(
            theta_new,
            prob,
            return_cost=True,
        )
        assert cost_f_new >= cost_f

        theta_new = theta + delta * grad_r
        *_, cost_r_new = get_grad_hess_cost_r(
            theta_new, prob, return_cost=True, regularization=reg
        )
        assert cost_r_new >= cost_r


if __name__ == "__main__":
    # for seed in range(10):
    for seed in range(1):
        np.random.seed(seed)
        print(f"test {seed+1}/{10}")
        print("hess...", end="")
        test_hess(seed=seed)
        print("finite diff...", end="")
        test_finite_differences(seed=seed)
        print("cost...", end="")
        test_cost(seed=seed)
        print("certificate...", end="")
        test_certificate(seed=seed)
        print("optimizer...", end="")
        test_optimizer(seed=seed)
        print("gradients...", end="")
        test_gradients(seed=seed)
    print("all tests passed")
