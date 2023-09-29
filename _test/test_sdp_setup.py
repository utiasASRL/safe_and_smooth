import scipy.sparse as sp
import numpy as np

from poly_certificate.sdp_setup import *
from poly_certificate.problem import Problem


def test_sdp_setup():
    tol = 1e-10

    mat = np.arange(9).reshape((3, 3)).astype(float)
    I = []
    J = []
    data = []
    fill(mat, range(3), range(3, 6), I, J, data, counter=0)

    A = sp.csr_matrix((data, (I, J)), shape=(6, 6))
    A_test = np.zeros((6, 6))
    A_test[:3, 3:6] = mat
    np.testing.assert_allclose(A.toarray(), A_test)

    prob = Problem(N=3, d=2, K=5)
    prob.generate_random()
    for regularization in ["no", "zero-velocity", "constant-velocity"]:

        Q, A_0, A_list, R = get_prob_matrices(prob, regularization=regularization)

        eigs = np.linalg.eigvalsh(Q.toarray())
        assert eigs[0] >= -tol, f"Q not psd: {eigs[0]}"

        if R is not None:
            eigs = np.linalg.eigvalsh(R.toarray())
            assert eigs[0] >= -tol, f"R not psd: {eigs[0]}"

        theta = prob.gt_init(regularization=regularization)
        f = get_f(theta)
        assert f.T @ A_0 @ f == 1.0
        for A_n in A_list:
            assert abs(f.T @ A_n @ f) <= tol


if __name__ == "__main__":
    test_sdp_setup()
    print("passed all tests")
