import scipy.sparse as sp
import numpy as np

from cvxopt.lapack import potrf, sytrf
from cvxopt import matrix, spmatrix
from cvxopt.cholmod import options, symbolic, numeric, spsolve

methods = {"ldl": 0, "ll": 2}  # LDL using simplical,  # LL using supernodal


def general_solve(A, method="ldl"):
    options["supernodal"] = methods[method]  # LL using supernodal
    if isinstance(A, spmatrix):
        B = spmatrix(1.0, range(A.size[0]), range(A.size[0]))
        try:
            # solution returned as output argument
            # B = splinsolve(A, B)
            # cert = True

            sym_factor = symbolic(A)

            # factorization is stored in sym_factor
            numeric(A, sym_factor)

            # returns D
            if method == "ldl":
                D = spsolve(sym_factor, B, sys=6)
                d = [D[i, i] for i in range(D.size[0])]
                # print("diagonal elements:", d)
                if np.all(np.array(d) >= 0):
                    cert = True
                else:
                    cert = False
            elif method == "ll":
                cert = True
        except ArithmeticError as e:
            cert = False
    else:
        raise TypeError("Types other than spmatrix not supported")
    return cert


def dense_compute_cholesky(A):
    try:
        # general matrix:
        potrf(A, uplo="L")

        # band matrix:
        # pbtrf(A, uplo="L")

        # on exit, A contains L in lower-triangular part.
        cert = True
    except ArithmeticError:
        cert = False
    return cert


def dense_compute_ldl(A):
    try:
        ipiv = matrix(1, size=A.size, tc="i")  # 'i' matrix
        sytrf(A, ipiv)
        # on exit, A and ipiv contain factorization
        cert = True
    except ArithmeticError:
        cert = False
    return cert


def generate_matrix(psd=False):
    np.random.seed(0)  # eigs has negative elements
    I = [0, 1, 2, 3, 4, 5]
    J = [1, 1, 3, 4, 5, 5]
    data = list(np.random.rand(6))
    if psd:
        I += list(range(6))
        J += list(range(6))
        data += list(np.ones(6))
    A_scipy = sp.csc_matrix((data, (I, J)), (6, 6))
    A_scipy = A_scipy + A_scipy.T
    eigs = np.linalg.eigvalsh(A_scipy.toarray())
    print("eigenvalues:", eigs)
    if psd:
        assert np.all(eigs >= 0)
    return A_scipy


if __name__ == "__main__":
    A_scipy = generate_matrix(psd=False)
    A = matrix(A_scipy.toarray())
    print("dense routines:")
    print("cholesky:", dense_compute_cholesky(A))
    print("ldl:     ", dense_compute_ldl(A))

    print("sparse routines:")
    A_scipy = generate_matrix(psd=False)
    A_coo = A_scipy.tocoo()
    A_sp = spmatrix(
        A_coo.data.tolist(), A_coo.row.tolist(), A_coo.col.tolist(), size=A_coo.shape
    )
    print("cholesky:", general_solve(A_sp, method="ll"))
    print("ldl:", general_solve(A_sp, method="ldl"))

    A_scipy = generate_matrix(psd=True)
    A_coo = A_scipy.tocoo()
    A_sp = spmatrix(
        A_coo.data.tolist(), A_coo.row.tolist(), A_coo.col.tolist(), size=A_coo.shape
    )
    print("cholesky:", general_solve(A_sp, method="ll"))
    print("ldl:", general_solve(A_sp, method="ldl"))
