"""
Gauss-Newton solver for GP regression.
"""

import time

import numpy as np
import progressbar as progressbar_module
import scipy.sparse as sp

from poly_certificate.sdp_setup import get_A_inv, fill
from poly_certificate.problem import Problem, generate_distances

MAX_ITER = 200

# if cost increases by more than TOL_COST instead of decreasing,
# use the regularization below (Levenberg-Marquardt scheme)
TOL_COST = 1e-5
LAMBDAS = [0.0]  # no effect
# LAMBDAS = [0.0] + list(np.logspace(-3, 3, 7))

TOL_STEP = 1e-10
TOL_G = -np.inf


def get_hess(
    theta,
    prob,
    regularization,
    precompute=None,
    mu_0=None,
):
    """Compute exact Hessian.

    See e.g. Nocedal & Wright, equation (10.5) for the analytical expression.
    """

    assert isinstance(prob, Problem)
    Hessian = get_hess_approx(theta, prob, regularization, precompute, mu_0)

    # compute the second part of the Hessian
    k = prob.get_dim(regularization)
    for n in range(prob.N):

        # We are only using the top-left dxd block of the em hessian,
        # becaues the rest is zero and we don't want to change the sparsity structure
        # of the Hessian because that would be expensive.
        Hessian[n * k : n * k + prob.d, n * k : n * k + prob.d] += (
            2 / prob.E * prob.get_em_hess(n, theta)[: prob.d, : prob.d]
        )
    return Hessian


def jacobian_distances(theta, anchors, W):
    """Fill the sparse Jacobian of the measurement function.

    J = [
        J_1, 0,  0, ...
        0,  J_2, 0, ...
    ]
    where J_n is the Jacobian with respect to theta_n.

    :param theta: N x k state matrix
    :param anchors: M x d anchor coordinates

    :return: sparse Jacobian of shape (M * N, N * k)
    """

    M, d = anchors.shape
    N = theta.shape[0]
    k = theta.shape[1]

    I = []
    J = []
    data = []
    counter = 0
    for n in range(N):
        G = 2 * (anchors - theta[n, :d][None, :])  # M x k
        G[W[:, n] == 0] = 0.0
        counter = fill(
            G, range(n * M, (n + 1) * M), range(n * k, n * k + d), I, J, data, counter
        )
        # jac[n * M : (n + 1) * M, n * k : n * k + d] = G
    jac = sp.csr_array((data, (I, J)), shape=(M * N, N * k))
    return jac


def setup_gp(prob, regularization, mu_0=None):
    """
    Generate the matrices relevant for Gaussian Process inference.

    See [2] for implementation details.
    """

    k = prob.get_dim(regularization)

    if mu_0 is None:
        Q_inv = sp.block_diag(
            [prob.get_Q_inv(n, regularization=regularization) for n in range(1, prob.N)]
        )
        A_inv = get_A_inv(prob, regularization=regularization, reduced=True)
        v = sp.csr_array((prob.N - 1, k))
    else:
        K_0 = sp.identity(k)
        Q_inv = sp.block_diag(
            [K_0]
            + [
                prob.get_Q_inv(n, regularization=regularization)
                for n in range(1, prob.N)
            ]
        )
        assert len(mu_0) == k, (len(mu_0), k)
        v = sp.csr_array((mu_0, ([0] * k, range(k))), shape=(prob.N, k))
        A_inv = get_A_inv(prob, regularization=regularization, reduced=False)
    K_inv = A_inv.T @ Q_inv @ A_inv
    return A_inv, Q_inv, K_inv, v


def total_cost(theta, prob, regularization, precompute=None, mu_0=None):

    Sig_inv = sp.block_diag([prob.Sig_inv] * prob.N) / prob.E
    G = generate_distances(theta[:, : prob.d], prob.anchors)  # K x N
    error = prob.D_noisy - G**2
    error[prob.W == 0] = 0.0

    # column-major, want to have N groups of [M, M, M] elements
    error = error.flatten("F").reshape(-1, 1)

    cost_f = float((error.T @ Sig_inv @ error))
    if precompute is None:
        A_inv, Q_inv, R, v = setup_gp(prob, regularization, mu_0=mu_0)
    else:
        A_inv, Q_inv, R, v = precompute
    error = v.reshape(-1, 1) - (A_inv @ theta.flatten()).reshape(-1, 1)
    cost_r = float(error.T @ Q_inv @ error) / prob.N
    return cost_r + cost_f


def get_hess_approx(
    theta,
    prob,
    regularization,
    precompute=None,
    mu_0=None,
):
    """Compute approximate Hessian as used by Gauss-Newton."""
    Sig_inv = sp.block_diag([prob.Sig_inv] * prob.N)
    J = jacobian_distances(theta, prob.anchors, prob.W)  # K x N*k
    if precompute is None:
        A_inv, Q_inv, R, v = setup_gp(prob, regularization, mu_0=mu_0)
    else:
        A_inv, Q_inv, R, v = precompute
    return 2 * (J.T @ Sig_inv @ J / prob.E + R / prob.N)


def get_grad_hess_cost_f(
    theta,
    prob,
    return_grad=True,
    return_hess=False,
    return_cost=False,
    exact_hess=False,
):
    """Calculate gradient and Hessian of distance error cost."""

    Sig_inv = (
        sp.block_diag([prob.Sig_inv] * theta.shape[0]) / prob.E
    )  # (N * M) x (N * M)
    J = jacobian_distances(theta, prob.anchors, prob.W)  # K x Nk
    G = generate_distances(theta[:, : prob.d], prob.anchors)  # K x N
    # print(prob.W[:4, :20].astype(float))
    error = prob.D_noisy - G**2
    error[prob.W == 0] = 0.0

    # column-major, want to have N groups of [M, M, M] elements
    error = error.flatten("F").reshape(-1, 1)  # NM x 1

    if exact_hess:
        # exact hessian has at each block -2 1'Sig_n e
        k = theta.shape[1]
        hess_corr = -2 * Sig_inv @ error  #  N * M vector
        elements = [
            e
            for i in range(prob.N)
            for e in [np.sum(hess_corr[i * prob.K : (i + 1) * prob.K])] * prob.d
            + [0] * (k - prob.d)
        ]
        exact_hess_add = sp.diags(elements)

    # nnz = np.abs(error) > 0
    # ii, jj = np.where(nnz)
    # error = sp.csr_array((error[nnz], [ii, jj]), shape=error.shape)
    grad_f = 2 * J.T @ Sig_inv @ error
    output = []
    if return_grad:
        output.append(grad_f)

    if return_hess and exact_hess:
        output.append(2 * J.T @ Sig_inv @ J + exact_hess_add)
    elif return_hess and not exact_hess:
        output.append(2 * J.T @ Sig_inv @ J)

    if return_cost:
        output.append(float((error.T @ Sig_inv @ error)))
    return output


def get_grad_hess_cost_r(
    theta,
    prob,
    regularization=None,
    precompute=None,
    mu_0=None,
    return_grad=True,
    return_hess=False,
    return_cost=False,
):
    """Calculate gradient and Hessian of regularization cost."""
    if precompute is None:
        assert regularization is not None
        A_inv, Q_inv, R, v = setup_gp(prob, regularization, mu_0=mu_0)
    else:
        A_inv, Q_inv, R, v = precompute

    error = v.reshape(-1, 1) - (A_inv @ theta.flatten()).reshape(-1, 1)
    grad_r = -2 * A_inv.T @ Q_inv @ error / prob.N
    output = []
    if return_grad:
        output.append(grad_r)
    if return_hess:
        output.append(2 * R / prob.N)
    if return_cost:
        output.append(float(error.T @ Q_inv @ error) / prob.N)
    return output


def gauss_newton(
    theta_0,
    prob,
    regularization,
    tol=TOL_STEP,
    gtol=TOL_G,
    verbose=0,
    max_iter=MAX_ITER,
    progressbar=False,
    mu_0=None,
    exact_hess=False,
):
    """
    :param theta_0: starting point of shape N x k
    :param tol: break when stepsize (RMSE) is smaller is smaller than this
    :param gtol: break when maximum absolute value of gradient is smaller than this.

    :returns: theta_est, stats.
        theta_est: estimate after convergence.
        stats: different convergence statistics.

    """
    # precompute these to save time later
    A_inv, Q_inv, R, v = setup_gp(prob, regularization, mu_0=mu_0)

    theta = theta_0.copy()
    stats = {"steps": []}

    ttot = 0
    converged = False

    use_lm = len(LAMBDAS) == 1
    current_cost = np.inf
    if progressbar:
        p = progressbar_module.ProgressBar(max_value=max_iter)
        p.start()

    for i in range(max_iter):

        if progressbar:
            p.update(i + 1)

        if verbose > 1:
            print(f"{i}/{max_iter}")

        for lamda in LAMBDAS:
            grad_f, Hess_f, *cost_f = get_grad_hess_cost_f(
                theta,
                prob,
                return_hess=True,
                exact_hess=exact_hess,
                return_cost=use_lm,
            )
            if regularization == "no":
                r = -grad_f
                L = Hess_f
            else:
                grad_r, Hess_r, *cost_r = get_grad_hess_cost_r(
                    theta,
                    prob,
                    precompute=(A_inv, Q_inv, R, v),
                    return_hess=True,
                    return_cost=use_lm,
                )

                r = -grad_r - grad_f
                L = Hess_r + Hess_f
            if lamda > 0:
                L += lamda * sp.identity(Hess_f.shape[0], dtype=float)

            abs_grad = np.abs(r)
            if np.all(abs_grad < gtol):
                stats["status"] = f"converged in gradient at {i+1}"
                stats["success"] = True
                converged = True
                break
            elif verbose:
                print("maximum grad:", np.max(abs_grad))

            t1 = time.time()
            dtheta = sp.linalg.spsolve(L, r)
            ttot += time.time() - t1

            step = None
            if np.any(np.isnan(dtheta)):
                stats["status"] = f"singular matrix in update calculation"
                stats["success"] = False
                converged = False
                break

            step = np.sqrt(np.mean(dtheta**2))
            if step < tol:
                stats["status"] = f"converged in stepsize at {i+1}"
                stats["success"] = True
                converged = True
                break

            if use_lm:
                continue

            # Levenberg-Marquardt step size correction

            new_cost = total_cost(
                theta + dtheta.reshape(prob.N, -1),
                prob,
                precompute=(A_inv, Q_inv, R, v),
            )
            error_cost = (new_cost - current_cost) / current_cost

            # This is not a valid update because the cost increases.
            # Try with the next bigger lamda.
            if error_cost > TOL_COST:
                if verbose:
                    print(f"    {lamda} cost did not decrease:", end="\t")
                    print(error_cost)
                continue
            elif error_cost < 0:
                if verbose:
                    print("cost decreased")
                current_cost = new_cost
                break
            else:
                if verbose:
                    print(f"    {lamda} cost stayed roughly same:", end="\t")
                    print(error_cost)
                current_cost = new_cost
                break

        theta = theta + dtheta.reshape((prob.N, -1))
        ttot += time.time() - t1

        stats["steps"].append(step)
        if converged:
            break

    # calculate values at converged solution (for reporting only)
    grad_f, hess_f, cost_f = get_grad_hess_cost_f(
        theta, prob, return_cost=True, return_hess=True, exact_hess=exact_hess
    )
    grad_r, hess_r, cost_r = get_grad_hess_cost_r(
        theta,
        prob,
        regularization,
        precompute=(A_inv, Q_inv, R, v),
        return_cost=True,
        return_hess=True,
    )
    stats["cost dist"] = cost_f
    stats["cost reg"] = cost_r
    stats["cost"] = cost_f + cost_r
    stats["grad dist"] = grad_f
    stats["grad reg"] = grad_r
    stats["grad"] = grad_f + grad_r
    stats["cov dist"] = hess_f
    stats["cov reg"] = hess_r
    stats["cov"] = hess_f + hess_r

    stats["success"] = converged
    stats["n it"] = i + 1
    stats["time"] = ttot
    stats["time per it"] = ttot / (i + 1)
    if i + 1 == max_iter:
        stats["status"] = f"reached max iterations {max_iter}"
    return theta, stats
