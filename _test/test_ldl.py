import numpy as np

import sys
from os.path import dirname

sys.path.append(dirname(__file__) + "/../")

from decompositions import *


def get_random_A(N, k, psd=True, arrowhead=False, sparse=False):
    if arrowhead:
        size = N * k + 1
    else:
        size = N * k

    I = []
    J = []
    data = []

    for n in range(N):
        (
            jj,
            ii,
        ) = np.meshgrid(range(n * k, (n + 1) * k), range(n * k, (n + 1) * k))
        data_here = np.random.randint(1, N, (k, k))
        I += list(ii.flatten())
        J += list(jj.flatten())
        data += list(data_here.flatten())

        if n <= N - 2:
            jj, ii = np.meshgrid(
                range(n * k, (n + 1) * k), range((n + 1) * k, (n + 2) * k)
            )
            data_here = np.random.randint(1, N, (k, k))
            I += list(ii.flatten())
            J += list(jj.flatten())
            data += list(data_here.flatten())

    if arrowhead:

        # below fills last row (otherwise A @ A.T is not arrowhead!)
        jj, ii = np.meshgrid(range(size), [size - 1])
        data_here = np.random.randint(1, N, size)

        I += list(ii.flatten())
        J += list(jj.flatten())
        data += list(data_here.flatten())

    if not sparse:
        A = np.zeros((size, size))
        A[I, J] = data
    else:
        A = sp.csc_array((data, [I, J]), shape=(size, size))

    if psd:
        return A @ A.T
    else:
        return A + A.T


def test_matrix_generation():
    N = 5
    k = 4
    seed = 1

    np.random.seed(seed)
    A_sparse = get_random_A(N, k, psd=True, arrowhead=True, sparse=True)
    np.random.seed(seed)
    A_dense = get_random_A(N, k, psd=True, arrowhead=True, sparse=False)

    np.testing.assert_allclose(A_dense, A_sparse.toarray())
    assert np.linalg.eigvalsh(A_dense)[0] > -1e-10

    np.random.seed(seed)
    A_sparse = get_random_A(N, k, psd=True, arrowhead=False, sparse=True)
    np.random.seed(seed)
    A_dense = get_random_A(N, k, psd=True, arrowhead=False, sparse=False)

    np.testing.assert_allclose(A_dense, A_sparse.toarray())
    assert np.linalg.eigvalsh(A_dense)[0] > -1e-10

    np.random.seed(seed)
    A_sparse = get_random_A(N, k, psd=False, arrowhead=False, sparse=True)
    np.random.seed(seed)
    A_dense = get_random_A(N, k, psd=False, arrowhead=False, sparse=False)

    np.testing.assert_allclose(A_dense, A_sparse.toarray())

    np.random.seed(seed)
    A_sparse = get_random_A(N, k, psd=False, arrowhead=True, sparse=True)
    np.random.seed(seed)
    A_dense = get_random_A(N, k, psd=False, arrowhead=True, sparse=False)

    np.testing.assert_allclose(A_dense, A_sparse.toarray())


def test_ldl_dense():
    N = 4
    A = np.random.randint(1, N, (N, N))
    A = A.T @ A
    L, D = ldl(A)
    np.testing.assert_allclose(np.diag(L), np.ones(N))
    np.testing.assert_allclose(L @ D @ L.T, A)


def test_ldl_block_arrow():
    test_ldl_block(arrowhead=True)


def test_ldl_block(arrowhead=False):
    N = 3
    k = 2
    A = get_random_A(N=N, k=k, psd=True, arrowhead=arrowhead)
    if arrowhead:
        H_ii, H_ij, h_i_list, h = get_blocks_from_matrix(A, size=k, return_h_i=True)
    else:
        H_ii, H_ij = get_blocks_from_matrix(A, size=k, return_h_i=False)
        h_i_list = h = None

    res, status = tri_block_ldl(H_ii, H_ij, h_i_list=h_i_list, h=h)
    assert res is not None

    if arrowhead:
        D_ii, J_ii, L_ij, l_i_list, l = res
    else:
        D_ii, J_ii, L_ij = res
        l_i_list = l = None

    for n in range(N - 1):
        if n == 0:
            np.testing.assert_allclose(J_ii[n] @ D_ii[n] @ J_ii[n].T, H_ii[n])
            if arrowhead:
                np.testing.assert_allclose(J_ii[n] @ D_ii[n] @ l_i_list[n], h_i_list[n])
        else:
            np.testing.assert_allclose(
                L_ij[n - 1] @ D_ii[n - 1] @ L_ij[n - 1].T
                + J_ii[n] @ D_ii[n] @ J_ii[n].T,
                H_ii[n],
            )
            if arrowhead:
                np.testing.assert_allclose(
                    L_ij[n - 1] @ D_ii[n - 1] @ l_i_list[n - 1]
                    + J_ii[n] @ D_ii[n] @ l_i_list[n],
                    h_i_list[n],
                )
        np.testing.assert_allclose(J_ii[n] @ D_ii[n] @ L_ij[n].T, H_ij[n])
    if arrowhead:
        sum_ = l
        for n in range(N):
            sum_ += l_i_list[n].T @ D_ii[n] @ l_i_list[n]
        np.testing.assert_allclose(sum_, h)

    L, D = get_ldl_from_blocks(D_ii, J_ii, L_ij, l_i_list=l_i_list, l=l)
    np.testing.assert_allclose(L.diagonal(), np.ones(L.shape[0]))

    A_test = L @ D @ L.T
    np.testing.assert_allclose(A_test.toarray(), A)

    if not arrowhead:
        # construct a slightly negative matrix
        # surprisingly, it still works!
        D[-1, -1] = -0.07
        A = L @ D @ L.T
        H_ii, H_ij = get_blocks_from_matrix(A, size=k)
        res, status = tri_block_ldl(H_ii, H_ij, h_i_list=None, h=None)
        D_ii, J_ii, L_ij = res
        L, D = get_ldl_from_blocks(D_ii, J_ii, L_ij)
        A_test = L @ D @ L.T
        np.testing.assert_allclose(A_test.toarray(), A.toarray())


def test_any():
    N = 5
    k = 3
    tol = 1e-10

    psd_test = 0
    nd_test = 0

    for i in range(100):
        np.random.seed(i)

        A_nd = get_random_A(N, k, psd=False)
        H_ii, H_ij = get_blocks_from_matrix(A_nd, k)
        res, status = tri_block_ldl(
            H_ii, H_ij, h_i_list=None, h=None, early_stopping=True
        )

        if np.linalg.eigvalsh(A_nd)[0] <= -tol:
            nd_test += 1
            assert res is None, "L D L.T should not exist because matrix is negative."

            # if res is not None:
            #    L, D = get_ldl_from_blocks(*res)
            #    np.testing.assert_allclose((L @ D @ L.T).toarray(), A_nd)
        else:
            psd_test += 1
            assert res is not None, "L D L.T should exist because matrix is p.s.d."
    print(f"Tested {nd_test} negative, {psd_test} positive definite matrices")


def test_psd():
    N = 5
    k = 3
    tol = 1e-10

    for i in range(100):
        np.random.seed(i)

        A_psd = get_random_A(N, k, psd=True)
        H_ii, H_ij = get_blocks_from_matrix(A_psd, k)
        res, status = tri_block_ldl(
            H_ii, H_ij, h_i_list=None, h=None, early_stopping=True, tol=1e-8
        )

        assert np.linalg.eigvalsh(A_psd)[0] >= -tol
        assert res is not None, "L D L.T should exist because matrix is p.s.d."
        L, D = get_ldl_from_blocks(*res)
        np.testing.assert_allclose((L @ D @ L.T).toarray(), A_psd)


if __name__ == "__main__":
    np.random.seed(0)

    test_matrix_generation()
    test_psd()
    test_any()
    test_ldl_dense()
    test_ldl_block()
    test_ldl_block_arrow()
    print("all tests passed")
