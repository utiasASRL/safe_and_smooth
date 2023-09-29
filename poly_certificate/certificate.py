import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spl
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence

from poly_certificate.cert_matrix import get_centered_matrix_blocks, get_hessian
from poly_certificate.decompositions import tri_block_ldl
from poly_certificate.sdp_setup import get_prob_matrices, get_H

REG = 1e-3  # tolerance for p.s.d.-ness. if minimum eigenvalue is more than -REG, it is considered still p.s.d


def chompack_cholesky(A):
    from cvxopt import spmatrix, amd
    from chompack import symbolic, cspmatrix, cholesky

    # generate sparse matrix and compute symbolic factorization
    # I = [0, 1, 3, 1, 5, 2, 6, 3, 4, 5, 4, 5, 6, 5, 6]
    # J = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6]
    # A = spmatrix([0.1*(i+1) for i in range(15)], I, J, (7,7)) + spmatrix(10.0,range(7),range(7))
    symb = symbolic(A, p=amd.order)

    # create cspmatrix
    L = cspmatrix(symb)
    L += A

    # compute numeric factorization
    cholesky(L)


def get_rho_and_lambdas(theta_est, prob, regularization):
    lambdas = []
    rho = 0.0

    for n in range(theta_est.shape[0]):
        nnz = prob.W[:, n] > 0
        Sig_inv_n = prob.Sig_inv[nnz, :][:, nnz]

        d_n = np.linalg.norm(
            theta_est[None, n, : prob.d] - prob.anchors[nnz, :], axis=1
        )
        eps = prob.D_noisy[nnz, n] - d_n**2  # M
        lambdas.append(-2 * np.sum(Sig_inv_n @ eps) / prob.E)

        if regularization == "no":
            rho += -eps.T @ Sig_inv_n @ eps / prob.E
        else:
            xnorm = np.linalg.norm(theta_est[n, : prob.d]) ** 2
            b_n = (
                prob.D_noisy[nnz, n] - np.linalg.norm(prob.anchors[nnz, :], axis=1) ** 2
            )
            rho += -(xnorm + b_n.T) @ Sig_inv_n @ eps / prob.E
    return float(rho), lambdas


def get_block_matrices_poly(prob, H):
    H_ii = H.get_block_matrices([f"x{n}" for n in range(prob.N)])
    # This is now donw in the matrix creationg directly)
    # if reg > 0:
    #    H_ii = [H_i + reg * np.eye(H_i.shape[0]) for H_i in H_ii]
    H_ij = H.get_block_matrices([(f"x{n}", f"x{n+1}") for n in range(prob.N - 1)])
    return H_ii, H_ij


def get_block_matrices(prob, rho, lamdas, regularization, reg=0, only_reg=False):
    k = prob.get_dim(regularization)
    dim = k + 1

    # preparation for sparse array
    H_ii_list = []
    H_ij_list = []
    h_i_list = []
    h = rho
    for m in range(prob.N):
        H_mm, H_nm = prob.get_R_matrices(m, dim, regularization)

        if not only_reg:
            Q_mm, q_m, q_0m = prob.get_Q_matrices(m, dim)

            B_mm = np.zeros((dim, dim))
            B_mm[range(prob.d), range(prob.d)] = lamdas[m]
            b_m = np.zeros((dim))
            b_m[-1] = -0.5 * lamdas[m]

            H_mm += Q_mm + B_mm

            h_i_list.append(q_m + b_m)
            h += q_0m

        if reg > 0:
            H_mm += reg * np.eye(dim)

        H_ii_list.append(H_mm)
        if H_nm is not None:
            H_ij_list.append(H_nm)
    return H_ii_list, H_ij_list, h_i_list, h


def get_centered_certificate(
    prob,
    theta_est,
    regularization="constant-velocity",
    schur="H",
    method="lanczos",
    return_time=False,
):

    xvar = prob.get_xvar()
    zvar = prob.get_zvar()
    var_dict = prob.get_var_dict(regularization)

    Hessian = get_hessian(prob, theta_est, regularization)
    B, C = get_centered_matrix_blocks(prob, theta_est, regularization)
    if schur is None:
        H = 0.5 * Hessian + B + C
        return get_mineig_sparse(H.get_matrix(var_dict), v0=None)

    if schur == "H":
        B.invert_diagonal(inplace=True)
        Test = 0.5 * Hessian - C.multiply(B).multiply(C.transpose())

        # TODO(FD): could also do LDL here.
        if method == "lanczos":
            return get_mineig_sparse(Test.get_matrix(xvar), v0=None)
        elif method == "ldl":
            return

    elif schur == "B":
        B_sparse = B.get_matrix(zvar)
        C_sparse = C.get_matrix((xvar, zvar))
        Hess_sparse = Hessian.get_matrix(xvar)

        # eigsh, LinearOperator
        def matvec(x):
            """linear operator: given x, output:
            (B - C.T @ Hess_inv @ C)x
            """
            # Hess @ y = A @ x
            # y = spl.spsolve(Hess_sparse, C_sparse @ x.flatten())
            y = solve(C_sparse @ x.flatten())
            return (
                B_sparse @ x.flatten() - 2 * C_sparse.T @ y
            )  # need flatten to handle (Nx1) case.

        try:
            from sksparse.cholmod import cholesky, CholmodNotPositiveDefiniteError
            solve = cholesky(Hess_sparse)
        except:
            solve = spl.factorized(Hess_sparse)
        lin_op = spl.LinearOperator(shape=B_sparse.shape, matvec=matvec)

        if method == "lanczos":
            t1 = time.time()
            cert = spl.eigsh(
                lin_op, k=1, return_eigenvectors=False, which="SA", v0=None
            )[0]
            ttot = time.time() - t1
        elif method == "cholesky":
            t1 = time.time()
            Hess_inv = spl.inv(Hess_sparse)
            Schur = B_sparse - 2 * C_sparse.T @ Hess_inv @ C_sparse
            try:
                cholesky(Schur)
                cert = True
            except CholmodNotPositiveDefiniteError:
                print("Warning: cholesky failed")
                cert = False
            ttot = time.time() - t1
        if return_time:
            return cert, ttot
        else:
            return cert
    else:
        raise ValueError(schur)


def compute_ldl_tridiag(H_blocks):
    """Compute ldl decomposition of tridiagonal (and optionally arrowhead) matrix.

    :param H_blocks: list of lists of diagonal / off-diagonal blocks, etc. for example:
    - H_ii, H_ij: [H_11, H_22, ...], [H_12, H_23] (main diagonal and off-diagonal blocks)
    - H_ii, H_ij, h_ij, h: (same as above, plus last column and last element)

    :return: list of diagonal elemenst of D, or None if LDL does not exist (negative matrix)
    """

    res, status = tri_block_ldl(*H_blocks, tol=1e-10, early_stopping=True)
    if res is not None:
        D, *_ = res
        ds = sorted([d for Dii in D for d in np.diag(Dii)])
        return ds, status
    return None, status


def get_mineig_sparse(H, v0=None, method="scipy", strategy="LA"):
    if method == "scipy":
        package = spl
        kwargs = dict(k=1, return_eigenvectors=False, which=strategy, v0=v0)
    elif method == "primme":
        import primme

        package = primme
        kwargs = dict(k=1, return_eigenvectors=False, which=strategy)
    else:
        raise ValueError(method)

    try:
        if strategy == "LA":
            eig_max = package.eigsh(H, **kwargs)[0]
            H_inv = (
                sp.csr_array(
                    (
                        np.full(H.shape[0], 2 * eig_max),
                        ((np.arange(H.shape[0]), np.arange(H.shape[0]))),
                    )
                )
                - H
            )
            # l_inv = 2*l_max - l
            eigs = package.eigsh(H_inv, **kwargs)
            return 2 * eig_max - eigs[0]
        elif strategy == "SA":
            eigs = package.eigsh(H, **kwargs)
            return eigs
        else:
            raise ValueError(strategy)
    except ArpackNoConvergence as e:
        # print(f"Scipy eigsh did not not converge!")
        return None


def get_minimum_eigenvalue(
    prob, rho, lamdas, regularization, use_sparse=False, verbose=False, return_all=False
):

    Q, A_0_list, A_list, R = get_prob_matrices(prob, regularization=regularization)
    if R is not None:
        Q += R

    H = get_H(Q, A_0_list, A_list, rho, lamdas)
    if use_sparse:

        try:
            cert = get_mineig_sparse(H)
        except Exception as e:
            print("Warning: eigsh failed with", e)
            eigs = spl.eigsh(H, k=1, return_eigenvectors=False, which="SM")
            if return_all:
                cert = eigs
            else:
                cert = eigs[0]
    else:
        eigs = np.linalg.eigvalsh(H.toarray())
        if return_all:
            cert = eigs
        else:
            cert = eigs[0]
    return cert, H


def get_certificate(
    prob,
    rho,
    lamdas,
    regularization="constant-velocity",
    cert_type=None,
    reg=REG,
    return_time=False,
):
    if cert_type is not None:
        print("cert_type is depcreated")

    # we add regularization to make sure
    # that the LDL is well defined even if the smallest
    # eigenvalue is weakly negative (we tolerate >= -REG).
    # in that case, we still want to consider it to be zero.
    # (and thus the matrix to be p.s.d.)
    H_blocks = get_block_matrices(
        prob, rho, lamdas, regularization=regularization, reg=reg
    )

    t1 = time.time()
    values, status = compute_ldl_tridiag(H_blocks)
    ttot = time.time() - t1

    if values is not None:
        cert = values[0]
    else:
        cert = -np.inf

    if return_time:
        return cert, ttot
    else:
        return cert
