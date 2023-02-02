import numpy as np

import sys
from os.path import dirname

sys.path.append(dirname(__file__) + "/../")

from problem import Problem

# time_range = [0, 50]
time_range = None


def test_calibrate():
    try:
        prob = Problem.init_from_dataset(
            "trial1",
            traj_only=False,
            time_range=time_range,
            use_anchors=None,
            use_gt=False,
            calibrate=False,
        )
        assert np.all(prob.get_biases(prob.D_noisy, squared=True) != 0)

        prob = Problem.init_from_dataset(
            "trial1",
            traj_only=False,
            time_range=time_range,
            use_anchors=None,
            use_gt=False,
            calibrate=True,
        )
        np.testing.assert_almost_equal(prob.get_biases(prob.D_noisy, squared=True), 0.0)

        # calibration with gt should not be necessary
        prob = Problem.init_from_dataset(
            "trial1",
            traj_only=False,
            time_range=time_range,
            use_anchors=None,
            use_gt=True,
            calibrate=False,
        )
        np.testing.assert_almost_equal(prob.get_biases(prob.D_noisy, squared=True), 0.0)

        # calibration below has no effect, but we can still test it
        prob = Problem.init_from_dataset(
            "trial1",
            traj_only=False,
            time_range=time_range,
            use_anchors=None,
            use_gt=True,
            calibrate=True,
        )
        np.testing.assert_almost_equal(prob.get_biases(prob.D_noisy, squared=True), 0.0)
    except FileNotFoundError:
        print("Skipping calibration test because dataset is not available")


def test_blocks():
    from sdp_setup import get_prob_matrices, get_H
    from certificate import get_block_matrices

    np.random.seed(0)
    regularization = "constant-velocity"
    prob = Problem(N=5, d=2, K=3)
    prob.generate_random()

    rho = -1.0
    lamdas = np.random.rand(prob.N)

    Q, A_0, A_list, R = get_prob_matrices(prob, regularization=regularization)
    A_0 = A_0.toarray()
    A_list = [A_n.toarray() for A_n in A_list]
    if R is not None:
        Q_all = Q.toarray() + R.toarray()
    else:
        Q_all = Q.toarray()
    H = get_H(Q_all, A_0, A_list, rho, lamdas)

    H_ii, H_ij, h_i_list, h = get_block_matrices(
        prob, rho, lamdas, regularization=regularization
    )

    k = prob.theta.shape[1]
    dim = k + 1
    for n in range(prob.N):
        H_nn = H[n * dim : (n + 1) * dim, n * dim : (n + 1) * dim]
        np.testing.assert_allclose(H_nn, H_ii[n])

        if n < prob.N - 1:
            m = n + 1
            H_nm = H[n * dim : (n + 1) * dim, m * dim : (m + 1) * dim]
            np.testing.assert_allclose(H_nm, H_ij[n])

        h_n = H[n * dim : (n + 1) * dim, -1]
        np.testing.assert_allclose(h_n, h_i_list[n])

    np.testing.assert_allclose(h, H[-1, -1])


def test_reflect_points():
    from problem import reflect_points

    anchors = np.c_[[3, -3, 3], [-3, -3, 3], [-3, 3, 3], [3, 3, 3]]  # d x M
    points = np.c_[
        [0, 0, 0],
        [0, 1, 1],
    ]  # d x N
    points_target = np.c_[[0, 0, 6], [0, 1, 5]]
    points_ref = reflect_points(points, anchors)
    np.testing.assert_allclose(points_ref, points_target)

    # vertical line through (1, 0)
    anchors = np.c_[
        [1, 0],
        [1, 1],
        [1, 2],
    ]

    points = np.c_[
        [0, 0],
        [-1, -1],
        [-2, -2],
    ]  # d x N
    points_target = np.c_[[2, 0], [3, -1], [4, -2]]
    points_ref = reflect_points(points, anchors)
    np.testing.assert_allclose(points_ref, points_target)


if __name__ == "__main__":
    # test_calibrate()
    test_blocks()
    test_reflect_points()
    print("all tests passed")
