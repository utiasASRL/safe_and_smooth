import numpy as np

# tolerance for p.s.d.-ness. if minimum eigenvalue is more than -REG, it is considered still p.s.d
REG = 1e-3


def get_rho_and_lambdas(
    theta_hat,
    prob,
    regularization,
):
    lambdas = []
    rho = 0.0

    for n in range(theta_hat.shape[0]):
        d_n = np.linalg.norm(theta_hat[None, n, : prob.d] - prob.anchors, axis=1)
        eps = prob.W[:, n] * (prob.D_noisy[:, n] - d_n**2)  # M
        lambdas.append(-2 * np.sum(prob.Sig_inv @ eps) / prob.E)

        if regularization == "no":
            rho += -eps.T @ prob.Sig_inv @ eps / prob.E
        else:
            xnorm = np.linalg.norm(theta_hat[n, : prob.d]) ** 2
            b_n = prob.W[:, n] * (
                prob.D_noisy[:, n] - np.linalg.norm(prob.anchors, axis=1) ** 2
            )
            rho += -(xnorm * np.ones((1, prob.K)) + b_n.T) @ prob.Sig_inv @ eps / prob.E
    return float(rho), lambdas


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


def get_minimum_eigenvalue(
    prob,
    rho,
    lamdas,
    regularization,
    use_sparse=False,
    verbose=False,
):
    from poly_certificate.sdp_setup import get_prob_matrices, get_H

    if verbose:
        print("setup matrices...", end=" ")
    Q, A_0_list, A_list, R = get_prob_matrices(prob, regularization=regularization)
    if R is not None:
        Q += R
    if verbose:
        print("done")

    if verbose:
        print("get H...", end=" ")
    H = get_H(Q, A_0_list, A_list, rho, lamdas)
    if verbose:
        print("done")

    if verbose:
        print("computing certificate...", end=" ")
    if use_sparse:
        import scipy.sparse as sp
        from scipy.sparse import linalg as spl

        try:
            eig_max = spl.eigsh(H, k=1, return_eigenvectors=False, which="LM")[0]
            H_inv = (
                sp.csr_array(
                    (
                        np.full(H.shape[0], eig_max),
                        ((np.arange(H.shape[0]), np.arange(H.shape[0]))),
                    )
                )
                - H
            )
            # l_inv = l_max - l
            eigs = spl.eigsh(H_inv, k=1, return_eigenvectors=False, which="LM")
            cert = eig_max - eigs[0]
        except Exception as e:
            print("Warning: eigsh faild", e)
            eigs = spl.eigsh(H, k=1, return_eigenvectors=False, which="SM")
            cert = eigs[0]
    else:
        eigs = np.linalg.eigvalsh(H.toarray())
        cert = eigs[0]
    return cert


def get_certificate(
    prob,
    rho,
    lamdas,
    regularization="constant-velocity",
    verbose=False,
    cert_type=None,
    reg=REG,
):
    if cert_type is not None:
        print("cert_type is depcreated")

    from poly_certificate.decompositions import tri_block_ldl

    # we add this regularization to make sure
    # that the LDL is well defined even if the smallest
    # eigenvalue is weakly negative (we tolerate >= -REG).
    # in that caes, we still want to consider it to be zero.
    # (and the matrix to be p.s.d.)
    H_ii, H_ij, h_i_list, h = get_block_matrices(
        prob, rho, lamdas, regularization=regularization, reg=reg
    )

    result, status = tri_block_ldl(
        H_ii,
        H_ij,
        h_i_list=h_i_list,
        h=h,
        verbose=verbose,
        early_stopping=True,
    )
    if result is None:
        # print("certificate failed:", status)
        return -np.inf
    D, J, L, l, d = result
    ds = np.array([d_ii for D_ii in D for d_ii in np.diag(D_ii)] + [d])
    if verbose:
        print(np.sort(ds))
    result = np.sort(ds)[0]
    return result
