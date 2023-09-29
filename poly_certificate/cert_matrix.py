import numpy as np

from poly_matrix.poly_matrix import PolyMatrix

from poly_certificate.sdp_setup import get_prob_matrices, get_H
from poly_certificate.problem import Problem


def get_centered_f(theta_est):
    k = theta_est.shape[1]
    if k in [4, 6]:
        d = k // 2
    else:
        d = k
    zs = np.linalg.norm(theta_est[:, :d], axis=1) ** 2
    xs = theta_est.flatten()
    np.testing.assert_allclose(xs[:k], theta_est[0, :])
    return np.r_[xs, zs]


def get_original_f(theta_est):
    k = theta_est.shape[1]
    if k in [4, 6]:
        d = k // 2
    else:
        d = k
    F = np.c_[theta_est, np.linalg.norm(theta_est[:, :d], axis=1) ** 2]
    f = np.r_[F.flatten(), 1.0]
    return f


def get_original_old_matrix(prob, rho, lamdas, regularization):

    Q, A_0_list, A_list, R = get_prob_matrices(prob, regularization=regularization)
    if R is not None:
        Q += R
    return get_H(Q, A_0_list, A_list, rho, lamdas)


def get_original_matrix(prob, rho, lamdas, regularization, reg=0.0):
    k = prob.get_dim(regularization)
    dim = k + 1

    mat = PolyMatrix()
    mat["l", "l"] = rho
    for m in range(prob.N):
        H_mm, H_nm = prob.get_R_matrices(
            m, dim, regularization
        )  # H_nm are elements H_[1, 2], H_[2, 3], etc.
        Q_mm, q_m, q_0m = prob.get_Q_matrices(m, dim)
        mat["l", "l"] += q_0m

        B_mm = np.zeros((dim, dim))
        B_mm[range(prob.d), range(prob.d)] = lamdas[m]
        b_m = np.zeros((dim))
        b_m[-1] = -0.5 * lamdas[m]

        H_mm += Q_mm + B_mm
        if reg > 0:
            H_mm += reg * np.eye(dim)
        mat[f"x{m}", f"x{m}"] = H_mm
        mat[f"x{m}", "l"] = q_m + b_m
        if H_nm is not None:
            assert m < prob.N
            mat[f"x{m}", f"x{m+1}"] = H_nm
    return mat


def get_centered_matrix(prob, theta_est, lamdas, regularization, test_jac=False):
    if type(theta_est) == float:
        raise ValueError("Second argument is theta_est")

    print(
        "get_centered_matrix: This method is deprecated, use get_centered_matrix_blocks instead!"
    )

    k = prob.get_dim(regularization)

    Hessian = PolyMatrix()
    Others = PolyMatrix()
    for m in range(prob.N):

        # Measurement cost
        J_m = prob.get_Jm(m, k, theta_est)  # M x d

        if test_jac:
            e_m = prob.get_em(m, theta_est)
            Jac = 1 / prob.E * J_m.T @ prob.Sig_inv @ e_m

        B_m = np.zeros((k, k))
        B_m[range(prob.d), range(prob.d)] = lamdas[m]

        Hessian[f"x{m}", f"x{m}"] = (
            1 / prob.E * (J_m.T @ prob.Sig_inv @ J_m) + B_m
        )  # B_m has 1/E in lambda
        Others[f"x{m}", f"z{m}"] = 1 / prob.E * np.sum(J_m.T @ prob.Sig_inv, axis=1)
        Others[f"z{m}", f"z{m}"] = 1 / prob.E * np.sum(prob.Sig_inv)

        # Regularization cost
        k = prob.get_dim(regularization)
        H_mm = np.zeros((k, k))
        if m < prob.N - 1:
            Phi_i = prob.get_phi(m + 1, regularization)
            Q_i = prob.get_Q_inv(m + 1, regularization)
            H_mm += Phi_i.T @ Q_i @ Phi_i

            H_mn = -Phi_i.T @ Q_i  # H_m{m+1}
            Hessian[f"x{m}", f"x{m+1}"] = 1 / prob.N * H_mn

            if test_jac:
                u_i_tilde = (
                    Phi_i @ theta_est[m] - theta_est[m + 1]
                )  # TODO(FD) would also have u here, if it wasn't zero
                Jac += 1 / prob.N * Phi_i.T @ Q_i @ u_i_tilde

        if m > 0:
            Q_im1 = prob.get_Q_inv(m, regularization)  # Q_{i-1}
            H_mm += Q_im1

            if test_jac:
                Phi_im1 = prob.get_phi(m, regularization)  # Phi_{i-1}
                u_im1_tilde = (
                    Phi_im1 @ theta_est[m - 1] - theta_est[m]
                )  # TODO(FD) would also have u here, if it wasn't zero
                Jac += -1 / prob.N * Q_im1 @ u_im1_tilde

        if test_jac:
            np.testing.assert_almost_equal(Jac, 0.0, decimal=4)

        Hessian[f"x{m}", f"x{m}"] += 1 / prob.N * H_mm
    return Hessian, Others


def get_hessian(prob, theta_est, regularization, reg=0):
    """Compmute exact Hessian."""

    # Sig_inv = sp.block_diag([prob.Sig_inv] * prob.N) / prob.E

    # J = jacobian_distances(theta, prob.anchors, prob.W)  # K x N*k
    k = theta_est.shape[1]
    H = PolyMatrix()
    for n in range(prob.N):
        J = np.zeros((prob.K, k))
        J[:, : prob.d] = 2 * (prob.anchors - theta_est[n, : prob.d][None, :])  # K x d
        J[prob.W[:, n] == 0, :] = 0.0

        R_nn = prob.get_R_nn(n, k, regularization)  # already has 1/N
        if n > 0:
            R_nm = prob.get_R_nm(n, k, regularization)  # already has 1/N
            H[f"x{n-1}", f"x{n}"] = 2 * R_nm
        H[f"x{n}", f"x{n}"] = (
            2 / prob.E * (J.T @ prob.Sig_inv @ J + prob.get_em_hess(n, theta_est))
            + 2 * R_nn
        ) + reg * np.eye(k)
    return H


def get_centered_matrix_blocks(prob, theta_est, regularization, reg=0.0):
    k = prob.get_dim(regularization)

    C = PolyMatrix()
    B = PolyMatrix()

    for m in range(prob.N):
        nnz = prob.W[:, m] > 0

        # extract the active matrix
        Sig_inv = prob.Sig_inv[nnz, :][:, nnz]
        B[f"z{m}", f"z{m}"] = 1 / prob.E * np.sum(Sig_inv) + reg

        assert prob.anchors.shape[1] == prob.d
        C_block = np.zeros(k)
        C_block[: prob.d] = np.sum(
            2
            / prob.E
            * (-prob.anchors[nnz, :].T + theta_est[m, : prob.d][:, None])
            @ Sig_inv,
            axis=1,
        )
        C[f"x{m}", f"z{m}"] = C_block
    return B, C
