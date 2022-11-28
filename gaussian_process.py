import numpy as np


def get_Qi(Q_c, dt, regularization="constant-velocity"):
    assert np.ndim(Q_c) > 1
    if regularization == "constant-velocity":
        return np.r_[
            np.c_[1 / 3 * dt**3 * Q_c, 1 / 2 * dt**2 * Q_c],
            np.c_[1 / 2 * dt**2 * Q_c, dt * Q_c],
        ]
    else:
        return Q_c


def get_Qi_inv(Q_c_inv, dt, regularization="constant-velocity", sparse=False):
    assert np.ndim(Q_c_inv) > 1

    if regularization == "constant-velocity":
        if sparse:
            import scipy.sparse as sp

            return sp.vstack(
                [
                    sp.hstack([12 / dt**3 * Q_c_inv, -6 / dt**2 * Q_c_inv]),
                    sp.hstack([-6 / dt**2 * Q_c_inv, 4 / dt * Q_c_inv]),
                ]
            )
        else:
            return np.r_[
                np.c_[12 / dt**3 * Q_c_inv, -6 / dt**2 * Q_c_inv],
                np.c_[-6 / dt**2 * Q_c_inv, 4 / dt * Q_c_inv],
            ]
    elif regularization == "zero-velocity":
        return Q_c_inv
    elif regularization == "no":
        return sp.csr_array(Q_c_inv.shape) if sparse else np.zeros_like(Q_c_inv)
    else:
        raise ValueError


def get_phi(d, dt, regularization="constant-velocity", sparse=False):
    if regularization == "constant-velocity":
        # TODO(FD) consider returning indices instead of the sparse matrix
        k = 2 * d
        if sparse:
            import scipy.sparse as sp

            data = [1.0] * k + [dt] * d
            row = list(range(k)) + list(range(d))
            col = list(range(k)) + list(range(d, k))
            return sp.csr_array((data, [row, col]), shape=(k, k))
        else:
            return np.r_[
                np.c_[np.eye(d), (dt) * np.eye(d)], np.c_[np.zeros((d, d)), np.eye(d)]
            ]
    elif regularization == "zero-velocity":
        return sp.identity(d) if sparse else np.eye(d)
    else:
        return sp.csr_array((d, d)) if sparse else np.zeros((d, d))


def get_posterior_covariances(prob, Cov_inv, regularization, ns=None):
    import scipy.sparse as sp

    if ns is None:
        ns = range(prob.N)

    k = prob.get_dim(regularization)

    # TODO(FD): can calculate the inverse of a block-tridiagonal
    # matrix more efficiently.
    print(f"inverting information matrix (size:{Cov_inv.shape[0]})...")
    Cov = sp.linalg.inv(Cov_inv)
    covariances = []

    for n in ns:
        covariances.append(Cov[n * k : n * k + 2, n * k : n * k + 2].toarray())
    return covariances


def get_prior_covariance(prob, i, j, regularization):
    """
    Use equation (13) in [2] to set up the diagonal and off-diagonal
    blocks of prior Covariance matrix. (These are the only ones needed for inference)
    """
    k = prob.get_dim(regularization)
    Cov = np.zeros((k, k))
    if i == j:  # e.g. i=j=3
        for n in range(i + 1):  # n=0,1,2,3,
            dt = prob.times[i] - prob.times[n]  # t3-t0, t3-t1, ..., 0
            Phi = get_phi(prob.d, dt, regularization)
            if dt == 0:
                Q_n = get_Qi(prob.Q, 0.1, regularization=regularization)  # k x k
            else:
                Q_n = get_Qi(prob.Q, dt, regularization=regularization)  # k x k
            Cov += Phi @ Q_n @ Phi.T
        return Cov
    elif i > j:
        for n in range(j + 1):
            dt = prob.times[i] - prob.times[n]
            Phi = get_phi(prob.d, dt, regularization)
            Q_n = get_Qi(prob.Q, dt, regularization=regularization)  # k x k
            Cov += Phi @ Q_n @ Phi.T
        dt = prob.times[i] - prob.times[j]
        Phi_ij = get_phi(prob.d, dt, regularization)
        return Phi_ij @ Cov
    else:
        return get_prior_covariance(prob, j, i, regularization).T


def get_prior_covariances(prob, regularization):
    K_ii = []
    K_ij = []
    for i in range(prob.N):
        K_ii.append(get_prior_covariance(prob, i, i, regularization))
        if i > 0:
            j = i - 1
            # start at (1, 0), (2, 1), ...
            K_ij.append(get_prior_covariance(prob, i, j, regularization))
    return K_ii, K_ij


def query_trajectory(
    tau,
    prob,
    x,
    regularization,
    return_covariance=False,
    K_post=None,
    K_prior_ii=None,
    K_prior_ij=None,
    x_prior=None,
):
    if x_prior is None:
        x_prior = np.zeros_like(x)

    assert x.shape[0] == prob.N
    k = x.shape[1]

    # find interval
    i = np.where(tau >= prob.times)[0][-1]
    j = i + 1
    ti = prob.times[i]
    tj = prob.times[j]
    assert tau >= prob.times[i]
    assert tau < prob.times[j]

    Q_c_inv = prob.Q_inv  # d x d
    Q_c = prob.Q  # d x d

    Q_tau = get_Qi(Q_c, tau - ti, regularization=regularization)  # k x k
    Q_inv_tj = get_Qi_inv(Q_c_inv, tj - ti, regularization=regularization)  # k x k

    Phi_tau_ti = get_phi(prob.d, tau - ti, regularization)  # k x k
    Phi_tj_tau = get_phi(prob.d, tj - tau, regularization)
    Phi_tj_ti = get_phi(prob.d, tj - ti, regularization)

    # use equation (3.208) in [3]
    Lambda = Phi_tau_ti - Q_tau @ Phi_tj_tau.T @ Q_inv_tj @ Phi_tj_ti
    Psi = Q_tau @ Phi_tj_tau.T @ Q_inv_tj
    mat = np.c_[Lambda, Psi]  # k x 2k

    x_prior_tau = Phi_tau_ti @ x_prior[i]  # k

    # use equation (3.209b)
    x_tau = x_prior_tau + mat @ (x[[i, j], :].flatten() - x_prior[[i, j], :].flatten())
    if not return_covariance:
        return x_tau

    K_post_ij = K_post[
        i * k : (j + 1) * k,
        i * k : (j + 1) * k,
    ]
    K_prior_ij = np.c_[
        np.r_[K_prior_ii[i], K_prior_ij[i]], np.r_[K_prior_ij[i], K_prior_ii[j]]
    ]
    # use equation (3.201) in [3]
    K_prior_i = K_prior_ii[i]
    K_tau_prior = Phi_tau_ti @ K_prior_i @ Phi_tau_ti.T + Q_tau
    print("K_tau prior", np.linalg.eigvalsh(K_tau_prior)[0])

    diff = K_post_ij - K_prior_ij
    print("difference", np.linalg.eigvalsh(diff)[0])

    # use equation (3.209b)
    # print(K_post_ij - K_prior_ij)
    K_tau = K_tau_prior + mat @ diff @ mat.T
    print("K_tau", np.linalg.eigvalsh(K_tau)[0])
    return x_tau, K_tau
