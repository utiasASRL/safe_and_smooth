import numpy as np
from numpy.linalg import norm
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

from poly_certificate.datasets import read_anchors, read_dataset
from poly_certificate.gaussian_process import get_phi, get_Qi_inv


def generate_random_trajectory(
    N,
    d,
    times,
    v_sigma=0.2,
    fix_x0=True,
    x0=None,
    return_velocities=False,
    fix_v0=True,
):
    if fix_x0 and x0 is None:
        x0 = np.ones(d)
    elif fix_x0 is False:
        x0 = np.random.rand(d) * 2 - 1  # uniform between -1, 1

    if fix_v0:
        v0 = np.ones(d)
    else:
        v0 = np.random.rand(d) * 2 - 1  # uniform between -1, 1

    trajectory = np.empty((N, d))
    velocities = np.empty((N, d))
    if times is None:
        times = np.arange(N, step=1.0)

    trajectory[0] = x0
    velocities[0] = v0
    for i in range(1, N):
        dt = times[i] - times[i - 1]
        trajectory[i] = trajectory[i - 1] + velocities[i - 1] * dt
        velocities[i] = velocities[i - 1] + np.random.normal(scale=v_sigma, size=d)
    if return_velocities:
        return trajectory, velocities
    else:
        return trajectory


def generate_anchors(M, d, trajectory=None, method="scale"):
    anchors = np.random.rand(M, d)
    if method == "scale":
        assert trajectory is not None
        assert trajectory.shape[1] == d
        size = np.max(trajectory, axis=0) - np.min(trajectory, axis=0)
        anchors *= size[None, :]  # [xmax - xmin, ymax - ymin]
        anchors += np.min(trajectory, axis=0)
    elif method == "square":
        assert trajectory is not None
        assert trajectory.shape[1] == d
        size = np.max(trajectory) - np.min(trajectory)
        traj_min = np.min(trajectory, axis=0)
        anch_min = np.min(anchors, axis=0)
        anch_max = np.max(anchors, axis=0)
        anchors = (anchors - anch_min) / (anch_max - anch_min) * size
        anchors += traj_min
    return anchors


def reflect_points(points, anchors, verbose=False):
    """
    reflect points across the plane defined by anchors
    """
    assert (
        points.shape[0] == anchors.shape[0]
    ), f"{points.shape} and {anchors.shape} should be dxM."
    # centre anchors
    c = np.mean(anchors, axis=1, keepdims=True)
    if verbose:
        print("centre", c)

    # fit plane to anchors
    U, S, V = np.linalg.svd(anchors - c)
    normal = U[:, -1]
    if verbose:
        print("normal", normal)

    # reflect points
    return points - 2 * normal @ (points - c) * normal[:, None]


def generate_distances(trajectory, anchors):
    return np.linalg.norm(trajectory[None, :, :] - anchors[:, None, :], axis=2)


class Problem(object):
    SIGMA_ACC_EST = 1.0
    SIGMA_DIST_EST = 1.0

    SIGMA_ACC_REAL = 0.2
    SIGMA_DIST_REAL = 0.1

    @staticmethod
    def init_from_dataset(
        fname,
        traj_only=False,
        time_range=None,
        use_anchors=None,
        sigma_acc_est=SIGMA_ACC_EST,
        sigma_dist_est=SIGMA_DIST_EST,
        use_gt=False,
        calibrate=False,
        cut_ends=False,
    ):
        def interpolate(y_given, x_given, x_new):

            inter = interp1d(x_given, y_given, fill_value="extrapolate")
            return inter(x_new)

        data_gt, data_uwb = read_dataset(fname)
        anchors, anchor_names = read_anchors("_data/config.yaml", use_anchors)
        # anchors_dict = dict(zip(anchor_names, anchors))

        assert anchors.shape[1] == 3
        if (time_range is None) and cut_ends:
            t_min = data_uwb.times[0] + 8
            t_max = data_uwb.times[len(data_uwb.times) - 1] - 8
            time_range = [t_min, t_max]

        if time_range is not None:
            data_gt = data_gt.loc[
                (data_gt.times > time_range[0]) & (data_gt.times < time_range[1])
            ]
            data_uwb = data_uwb.loc[
                (data_uwb.times > time_range[0]) & (data_uwb.times < time_range[1])
            ]

        M = anchors.shape[0]
        N = len(data_uwb)
        d = 3
        prob = Problem(
            N, d, M, sigma_acc_est=sigma_acc_est, sigma_dist_est=sigma_dist_est
        )
        prob.anchors = anchors
        prob.times = data_uwb.times.unique()

        prob.trajectory = np.c_[
            interpolate(data_gt.x.values, data_gt.times.values, prob.times),
            interpolate(data_gt.y.values, data_gt.times.values, prob.times),
            interpolate(data_gt.z.values, data_gt.times.values, prob.times),
        ]
        assert prob.trajectory.shape == (prob.N, prob.d)

        if traj_only:
            return prob

        D_noisy = np.zeros((M, N))
        prob.W = np.zeros((M, N), dtype=bool)
        for j, t in enumerate(prob.times):
            df = data_uwb.loc[data_uwb.times == t]
            for _, row in df.iterrows():
                try:
                    i = np.where(anchor_names == row.anchor)[0][0]
                except Exception as e:
                    # print(row.anchor, "not in", anchor_names)
                    continue
                prob.W[i, j] = 1

                if use_gt:
                    a = prob.anchors[i]
                    D_noisy[i, j] = norm(a - prob.trajectory[j])
                else:
                    D_noisy[i, j] = row.distance
        prob.E = np.sum(prob.W)

        if calibrate:
            biases = prob.get_biases(D_noisy)
            D_noisy[D_noisy > 0] = (D_noisy - biases[:, None])[D_noisy > 0]
            prob.D_noisy = D_noisy**2
        else:
            prob.D_noisy = D_noisy**2

        # prune points where we don't have distance measurements.
        prob.prune()

        return prob

    def prune(self):
        indices = np.any(self.W, axis=0)
        self.N = np.sum(indices)
        self.W = self.W[:, indices]
        if self.trajectory is not None:
            self.trajectory = self.trajectory[indices, :]
        if self.times is not None:
            self.times = self.times[indices]
        if self.D_noisy is not None:
            self.D_noisy = self.D_noisy[:, indices]

    def add_noise(self, sigma):
        D_noisy = self.D_noisy.copy()
        D_noisy[D_noisy > 0] = np.sqrt(D_noisy[D_noisy > 0])
        D_noisy += np.random.normal(scale=sigma, loc=0, size=D_noisy.shape)
        self.D_noisy = D_noisy**2

    def __init__(
        self, N, d, K=None, sigma_acc_est=SIGMA_ACC_EST, sigma_dist_est=SIGMA_DIST_EST
    ):
        self.N = N
        self.d = d
        self.K = K
        self.E = None  # number of measurements, needs to be initialized later

        # measurements
        self.anchors = None
        self.D_noisy = None
        self.times = None
        self.W = None

        self.Q = sigma_acc_est**2 * np.eye(d)
        self.Q_inv = 1 / sigma_acc_est**2 * np.eye(d)
        if self.K is not None:
            self.Sig_inv = 1 / sigma_dist_est**2 * np.eye(K)

        # ground truth
        self.velocities = None
        self.trajectory = None
        self.theta = None

    def extract_timerange(self, mini, maxi):
        self.trajectory = self.trajectory[mini:maxi]
        self.times = self.times[mini:maxi]
        self.theta = self.theta[mini:maxi]
        self.D_noisy = self.D_noisy[:, mini:maxi]
        self.W = self.W[:, mini:maxi]
        self.N = self.trajectory.shape[0]

    def calculate_noise_level(self, squared=True):
        D_gt = generate_distances(self.trajectory, self.anchors)
        if squared:
            Eps = np.abs(self.D_noisy - D_gt**2)
        else:
            Eps = np.abs(np.sqrt(self.D_noisy) - D_gt)
        if self.W is not None:
            Eps = self.W.astype(float) * Eps
        noise_level = np.sqrt(np.mean(Eps[Eps > 0] ** 2))
        return noise_level

    def generate_random(
        self,
        sigma_acc_real=SIGMA_ACC_REAL,
        sigma_dist_real=SIGMA_DIST_REAL,
        anchor_method="square",
        verbose=False,
    ):
        if verbose:
            print("sigma_acc_real, sigma_dist_real:")
            print(sigma_acc_real, sigma_dist_real)

        self.times = np.arange(self.N)
        self.trajectory, self.velocities = generate_random_trajectory(
            self.N, self.d, self.times, v_sigma=sigma_acc_real, return_velocities=True
        )
        self.theta = np.c_[self.trajectory, self.velocities]
        if self.K is not None:
            self.anchors = generate_anchors(
                self.K, self.d, trajectory=self.trajectory, method=anchor_method
            )  # K x d
            self.generate_distances(sigma_dist_real)

        if verbose:
            print("first two points:", self.trajectory[:3, :])
            print("anchors:", self.anchors)

    def generate_distances(self, sigma_dist_real=SIGMA_DIST_REAL):
        self.D_noisy = (
            generate_distances(self.trajectory, self.anchors)
            + np.random.normal(scale=sigma_dist_real, size=(self.K, self.N))
        ) ** 2
        self.W = np.ones(self.D_noisy.shape, dtype=bool)
        self.E = self.W.size

    def generate_W_roundrobin(self, use_anchors=None, measurements_per_time=1):
        """Initialize W with only limited measurement at a time.
        Optionally use only a subset of anchors.
        :param use_anchors: list of indices to use, or None to use all.
        """
        if use_anchors is None:
            remove_anchors = {}
        else:
            all_anchors = set(np.arange(self.K))
            remove_anchors = all_anchors.difference(use_anchors)

        self.W = np.zeros((self.K, self.N))
        for i in range(measurements_per_time):
            i_indices = np.mod(np.arange(self.N) + i, self.K)
            self.W[i_indices, range(self.N)] = 1.0
        self.W[list(remove_anchors), :] = 0.0

        self.prune()

    def random_traj_init(self, regularization, sigma_acc_est=SIGMA_ACC_EST):
        traj, vel = generate_random_trajectory(
            self.N, self.d, self.times, v_sigma=sigma_acc_est, return_velocities=True
        )
        if regularization == "constant-velocity":
            return np.c_[traj, vel]
        return traj

    def random_init(self, regularization):
        if regularization == "constant-velocity":
            k = 2 * self.d
        else:
            k = self.d
        return np.random.rand(self.N, k)

    def gt_init(self, regularization, extra_noise=0.0):
        if extra_noise > 0:
            traj = self.trajectory + np.random.normal(
                loc=0, scale=extra_noise, size=self.trajectory.shape
            )
        else:
            traj = self.trajectory
        if regularization == "constant-velocity":
            vel = np.ones((self.N, self.d))
            return np.c_[traj, vel]
        return traj

    def reflected_init(
        self, regularization=None, fraction=0.5, verbose=False, extra_noise=0.0
    ):
        if fraction == 0:
            return self.gt_init(regularization=regularization)
            # raise ValueError("use gt_init")
        n_reflected = int(self.N * fraction)
        n_original = self.N - n_reflected
        traj_refl = reflect_points(
            self.trajectory[-n_reflected:, :].T, self.anchors.T, verbose=verbose
        ).T
        traj = np.r_[self.trajectory[:n_original, :], traj_refl]

        if extra_noise > 0:
            traj += np.random.normal(loc=0, scale=extra_noise, size=traj.shape)

        if regularization == "constant-velocity":
            vel = np.ones((self.N, self.d))
            return np.c_[traj, vel]
        return traj

    def get_dim(self, regularization):
        if regularization == "no":
            return self.d
        elif regularization == "zero-velocity":
            return self.d
        elif regularization == "constant-velocity":
            return 2 * self.d
        else:
            raise ValueError(f"unknown regularization: {regularization}")

    def get_rmse_unbiased(self, trajectory_est):
        mean_error = self.get_mean_error(trajectory_est)
        return np.sqrt(self.get_mse(trajectory_est, mu=mean_error))

    def get_rmse(self, trajectory_est):
        return np.sqrt(self.get_mse(trajectory_est))

    def get_mean_error(self, trajectory_est):
        return np.sum(trajectory_est - self.trajectory, axis=0) / self.N

    def get_mse(self, trajectory_est, mu=None):
        if mu is None:
            return np.sum((trajectory_est - self.trajectory) ** 2) / self.N
        else:
            return (
                np.sum((trajectory_est - self.trajectory - mu[None, :]) ** 2) / self.N
            )

    def get_mae(self, trajectory_est):
        return np.sum(np.abs(trajectory_est - self.trajectory)) / self.N

    def get_mean_error_xyz(self, trajectory_est):
        return np.mean(trajectory_est - self.trajectory, axis=0)

    def get_var_error_xyz(self, trajectory_est):
        return np.var(trajectory_est - self.trajectory, axis=0)

    def get_biases(self, D_noisy, squared=False):
        if squared:
            D_noisy = np.sqrt(D_noisy)
        biases = np.empty(D_noisy.shape[0])
        for i in range(D_noisy.shape[0]):
            indices = D_noisy[i, :] > 0  # only consider existing distances
            D_real = generate_distances(self.trajectory[indices, :], self.anchors[[i]])
            biases[i] = np.mean(D_noisy[i, indices] - D_real)
        return np.array(biases).flatten()

    def get_Jm(self, m, k, theta_est):
        """Calculate Jacobian"""
        Jm = 2 * (self.anchors - theta_est[m, : self.d][None, :])  # (M x d)
        Jm_aug = np.c_[Jm, np.zeros((Jm.shape[0], k - self.d))]  # M x K
        assert Jm_aug.shape == (self.K, k)
        return Jm_aug

    def get_em(self, m, theta_est):
        """Calculate error vector"""
        em = self.W[:, m] * (
            self.D_noisy[:, m]  # already squared
            - norm(self.anchors - theta_est[m, : self.d][None, :], axis=1) ** 2
        )  # M
        return em

    def get_em_hess(self, n, theta):
        r"""Get the non-linear contribution to the gradient:

        .. math::
            \sum_{m=1}^{M_n} \nabla^2 e_{nm} (\Sigma^{-1}_n \mathbf{e}_n)_m
        """
        # need the r" above to avoid warning

        k = theta.shape[1]
        hess = np.zeros((k, k))

        errors = self.Sig_inv @ self.get_em(n, theta)
        for m in range(self.K):
            hess[range(self.d), range(self.d)] += -2 * errors[m]
        return hess

    def get_Q_matrices(self, m, dim):
        """Compute m-th component of Q matrices: Q_mm, q_m, and the m-th element of q
        :param m: timestamp
        :param dim: total dimension (including e.g. velocity)
        """
        Q_mm = np.zeros((dim, dim))
        q_m = np.zeros(dim)
        assert self.W is not None

        nnz = np.where(self.W[:, m] > 0)[0]
        # this shouldn't happen because we assume to have at least one distance measurement
        # at each time, but just in case...
        if len(nnz) == 0:
            # print("Warning: no measurement at time:", m, self.W[:, m])
            return Q_mm, q_m, 0.0

        Sig_inv_n = self.Sig_inv[nnz, :][:, nnz]

        Q_n = np.c_[2 * self.anchors[nnz], -np.ones(len(nnz))]
        Q_mm[np.ix_(list(range(self.d)) + [-1], list(range(self.d)) + [-1])] = (
            Q_n.T @ Sig_inv_n @ Q_n
        )

        bm = self.D_noisy[nnz, m] - norm(self.anchors[nnz], axis=1) ** 2

        q_m[: self.d] = 2 * self.anchors[nnz].T @ Sig_inv_n @ bm
        q_m[-1] = -np.sum(Sig_inv_n @ bm)
        q_0m = bm.T @ Sig_inv_n @ bm

        return Q_mm / self.E, q_m / self.E, q_0m / self.E

    def get_phi(self, m, regularization):
        assert m > 0
        dt = self.times[m] - self.times[m - 1]
        return get_phi(self.d, dt, regularization)

    def get_Q_inv(self, m, regularization):
        assert m > 0
        dt = self.times[m] - self.times[m - 1]
        # print("Q_inv:", self.Q_inv)
        return get_Qi_inv(self.Q_inv, dt, regularization)

    def get_R_nn(self, n, dim, regularization, verbose=False):
        R_nn = np.zeros((dim, dim))
        if n == 0:
            # phi_12 @ Q_2 @ phi_21
            Q_inv_m = self.get_Q_inv(n + 1, regularization=regularization)
            phi_m = self.get_phi(n + 1, regularization=regularization)
            if verbose:
                print(f"R_nn: phi_{n,n+1} Q_{n+1} phi_{n, n+1}")
            R_nn[: phi_m.shape[1], : phi_m.shape[1]] = phi_m.T @ Q_inv_m @ phi_m
        elif n <= self.N - 2:
            # Q_2 + phi_23 @ Q_3 @ phi_32
            # etc.
            Q_inv_n = self.get_Q_inv(n, regularization=regularization)
            Q_inv_m = self.get_Q_inv(n + 1, regularization=regularization)
            phi_m = self.get_phi(n + 1, regularization=regularization)
            if verbose: 
                print(f"R_nn: Q_{n} + phi_{n,n+1} Q_{n+1} phi_{n, n+1}")
            R_nn[: phi_m.shape[1], : phi_m.shape[1]] = (
                Q_inv_n + phi_m.T @ Q_inv_m @ phi_m
            )
        elif n == self.N - 1:
            # Q_N
            Q_inv_n = self.get_Q_inv(n, regularization=regularization)
            R_nn[: Q_inv_n.shape[0], : Q_inv_n.shape[1]] = Q_inv_n
        return R_nn / self.N

    def get_R_nm(self, m, dim, regularization):
        """returns the element R_{m-1, m}"""
        assert m > 0
        R_nm = np.zeros((dim, dim))
        Q_inv_m = self.get_Q_inv(m, regularization=regularization)
        phi_m = self.get_phi(m, regularization=regularization)
        R_nm[: phi_m.shape[1], : phi_m.shape[1]] = -phi_m.T @ Q_inv_m
        # -phi_12.T @ Q_2
        return R_nm / self.N

    def get_R_matrices(self, m, dim, regularization):
        """
        n goes from 0 to N-1.
        - at n=0, the function returns the first element, R_00
            this becomes R_nn in the outer loop.
        - at time n==1, the function returns R_mm=R_11 and R_nm=R_01.
        - at time n==N-1, the function returns R_{N-1,N-1}
        """
        R_mm = self.get_R_nn(m, dim, regularization)
        if m < self.N - 1:
            R_nm = self.get_R_nm(m + 1, dim, regularization)
        else:
            R_nm = None
        return R_mm, R_nm

    def __repr__(self):
        outstr = "Problem with parameters:\n"
        outstr += f"{self.N, self.K, self.d}\n"
        outstr += f"Sig_inv[0,0] (distance covariance):{self.Sig_inv[0, 0]:.1f}\n"
        outstr += f"Q_inv[0,0] (motion covariance): {self.Q_inv[0, 0]:.1f}"
        return outstr

    # polymat stuff
    def get_xvar(self):
        return [f"x{i}" for i in range(self.N)]

    def get_zvar(self):
        return [f"z{i}" for i in range(self.N)]

    def get_var_dict(self, regularization):
        k = self.get_dim(regularization)
        var_dict = {key: k for key in self.get_xvar()}
        var_dict.update({key: 1 for key in self.get_zvar()})
        return var_dict

    def plot(self, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        if self.anchors is not None:
            ax.scatter(*self.anchors.T, color="k", marker="x", label="anchors")

        if self.trajectory is not None:
            ax.scatter(*self.trajectory.T, color="k", marker="o", label="real")
        ax.axis("equal")
        if show:
            plt.show()
        return fig, ax

    def plot_estimates(self, points_list, show=True, ax=None, **kwargs):
        if ax is None:
            fig, ax = self.plot(show=False)
        for i, p in enumerate(points_list):
            ax.plot(*p.T, marker=".", **kwargs)
        ax.legend()
        if show:
            plt.show()
