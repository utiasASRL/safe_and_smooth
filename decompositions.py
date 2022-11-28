import numpy as np
import scipy.sparse as sp

TOL = 1e-10  # tolerance for p.s.d. inside LDL decomposition


def ldl(A, tol=TOL, use_scipy=False, binary=False, verbose=False):
    """Compute ldl decomposition, without permutation.

    :param tol: what to consider as zero.

    :returns: L, D if LDL decomposition exists (A is p.s.d), None, None otherwise.
    """
    if use_scipy:
        from scipy.linalg import ldl as ldl_scipy

        l, d, p = ldl_scipy(A, overwrite_a=True, check_finite=False)
        if binary:
            if np.any(np.diag(d) < -tol):
                if verbose:
                    print(f"found negative diagonal element in {np.sort(np.diag(d))}")
                return None, None
        return l, d

    N = A.shape[0]
    D = np.zeros((N, N))
    L = np.eye(N)
    for j in range(N):
        if j > 0:
            D[j, j] = A[j, j] - np.sum(L[j, :j] ** 2 * D[range(j), range(j)])
        else:
            D[j, j] = A[j, j]

        if (D[j, j] < -tol) and binary:
            # print(f"found negative at {j}/{N}:", D[j, j])
            return None
        # else:
        # print("non-negative:", D[j, j])

        for i in range(j + 1, N):
            if j > 0:
                if abs(D[j, j]) > tol:
                    L[i, j] = (
                        1
                        / D[j, j]
                        * (
                            A[i, j]
                            - np.sum(L[i, :j] * L[j, :j] * D[range(j), range(j)])
                        )
                    )
                else:
                    L[i, j] = 0
            else:
                if abs(D[j, j]) > tol:
                    L[i, j] = 1 / D[j, j] * (A[i, j])
                else:
                    L[i, j] = 0
    try:
        np.testing.assert_allclose(L @ D @ L.T, A, rtol=1e-3, atol=1e-5)
    except TypeError:  # A is sparse
        np.testing.assert_allclose(L @ D @ L.T, A.toarray(), rtol=1e-3, atol=1e-5)
    except:
        return None, None
    return L, D


def get_blocks_from_matrix(A, size=4, return_h_i=False):
    """Assuming A is block-tridiagonal of size 4, return list of diagonal blocks.

    :param return_h_i: return also the last column and element, assuming matrix is block-tridiagonal and arrow-head (as in [1])

    :return: H_ii (list of diagonal blocks), H_ij (list of off-diagonal blocks)
    """
    H_ii = []
    H_ij = []
    Nk = A.shape[0]
    N = Nk // size
    for n in range(N):
        H_ii.append(A[n * size : (n + 1) * size, n * size : (n + 1) * size])
        if n < N - 1:
            H_ij.append(A[n * size : (n + 1) * size, (n + 1) * size : (n + 2) * size])

    if not return_h_i:
        return H_ii, H_ij

    h_i_list = []
    for n in range(N):
        h_i_list.append(A[n * size : (n + 1) * size, -1])
    return H_ii, H_ij, h_i_list, A[-1, -1]


# TODO(FD): convert below to sparse, as in compose_ldl. Can also combine both in one function (sparsity patterns are similar)
def get_matrix_from_blocks(H_ii, H_ij, h_i_list=None, h=None):
    """
    Given blocks of matrix, construct the full matrix (dense)
    """
    size = H_ii[0].shape[0]
    N = len(H_ii)
    if h_i_list is not None:
        assert h is not None
        A = np.zeros((size * N + 1, size * N + 1))
    else:
        print("Warning: in new format, need also h_i_list")
        A = np.zeros((size * N, size * N))

    for n in range(N):
        A[n * size : (n + 1) * size, n * size : (n + 1) * size] = H_ii[n]
        if n < N - 1:
            A[(n) * size : (n + 1) * size, (n + 1) * size : (n + 2) * size] = H_ij[n]
            A[(n + 1) * size : (n + 2) * size, (n) * size : (n + 1) * size] = H_ij[n].T

    if h_i_list is not None:
        for n in range(N):
            A[-1, n * size : (n + 1) * size] = h_i_list[n]
            A[n * size : (n + 1) * size, -1] = h_i_list[n]
        A[-1, -1] += h
    return A


def get_ldl_from_blocks(D_ii, J_ii, L_ij, l_i_list=None, l=None):
    """Given L and D in blocks, construct the full L and D matrices."""
    N = len(D_ii)
    k = D_ii[0].shape[0]
    ds = [d for D in D_ii for d in np.diag(D)]
    if l_i_list is None:
        D = sp.csr_array((ds, [range(k * N), range(k * N)]))
    else:
        ds += [l]
        D = sp.csr_array(
            (ds, [range(k * N + 1), range(k * N + 1)]), (k * N + 1, k * N + 1)
        )

    I = []
    J = []
    data = []
    for n in range(N):
        ii, jj = np.mgrid[n * k : (n + 1) * k, n * k : (n + 1) * k]
        I += list(ii.flatten())
        J += list(jj.flatten())
        data += list(J_ii[n].flatten())

    for n in range(N - 1):
        ii, jj = np.mgrid[(n + 1) * k : (n + 2) * k, n * k : (n + 1) * k]
        I += list(ii.flatten())
        J += list(jj.flatten())
        data += list(L_ij[n].flatten())

    if l_i_list is not None:
        assert l is not None

        for n in range(N):
            ii, jj = np.mgrid[N * k : N * k + 1, n * k : (n + 1) * k]
            I += list(ii.flatten())
            J += list(jj.flatten())
            data += list(l_i_list[n])
        I += [N * k]
        J += [N * k]
        data += [1]
    L = sp.csr_array((data, [I, J]))
    return L, D


def tri_block_ldl(
    H_ii, H_ij, h_i_list=None, h=None, tol=TOL, verbose=False, early_stopping=False
):
    """
    Returns the block-ldl-decompositions
    :return:
        - D_ii, J_ii, L_ij if h_i_list, h are not given
        - D_ii, J_ii, L_ij, l_i_list, l if h_i_list, h are given
        - None if the decomposition doesn't exist (matrix is not p.s.d.)
    """
    D_ii = []
    J_ii = []
    L_ij = []
    status = ""
    for n, H_nn in enumerate(H_ii):
        if n == 0:
            J_nn, D_nn = ldl(H_nn, verbose=verbose, tol=tol)
            if J_nn is None:
                status = f"dense ldl failed at time {n}"
                return None, status

            if early_stopping and np.any(np.diag(D_nn) < -tol):
                status = f"found zero diagonal element at time {n}: {np.diag(D_nn)}"
                return None, status

            D_ii.append(D_nn)
            J_ii.append(J_nn)

        else:
            mat = H_nn - L_ij[n - 1] @ D_ii[n - 1] @ L_ij[n - 1].T
            J_nn, D_nn = ldl(mat, verbose=verbose, tol=tol)
            if J_nn is None:
                status = f"dense ldl failed at time {n}"
                return None, status

            if early_stopping and np.any(np.diag(D_nn) < -tol):
                status = f"found zero diagonal element at time {n}: {np.diag(D_nn)}"
                return None, status

            D_ii.append(D_nn)
            J_ii.append(J_nn)

        if n < len(H_ii) - 1:
            try:
                DJ_inv = np.linalg.pinv(D_ii[n] @ J_ii[n].T)
            except:
                status = f"error computing inv(D @ J.T) at {n}"
                return None, status
            L_ij.append(H_ij[n].T @ DJ_inv)

    if h_i_list is None:
        assert h is None, "cannot use h without h_i_list"
        return (D_ii, J_ii, L_ij), status

    l_i = []
    sum_ = 0
    for n, h_n in enumerate(h_i_list):
        try:
            JD_inv = np.linalg.pinv(J_ii[n] @ D_ii[n])
        except:
            status = f"error computing inv(J @ D) at {n}"
            return None, status

        if n == 0:
            l_i.append(JD_inv @ h_n)
        elif n > 0:
            l_i.append(JD_inv @ (h_n - L_ij[n - 1] @ D_ii[n - 1] @ l_i[n - 1]))

        sum_ += l_i[n].T @ D_ii[n] @ l_i[n]

    d = h - sum_

    rel_d = (h - sum_) / sum_
    if early_stopping and rel_d < -tol:
        status = f"error in final value: h={h}, sum_={sum_}, d={d}"
        return None, status
    elif abs(rel_d) < tol:
        d = 0.0

    status = "success"
    l_i.append(1)
    return (D_ii, J_ii, L_ij, l_i, d), status
