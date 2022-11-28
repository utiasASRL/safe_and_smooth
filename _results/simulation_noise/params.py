import numpy as np

params = dict(
    N=100,
    K=6,
    d=2,
    sigma_dist_est=None,
    sigma_acc_real=0.2,
    setup_seed=np.arange(100),
    init_seed=np.arange(10),
    sigma_dist_real=np.logspace(-4, 2, 7),
    sigma_acc_est=[0.2],
    regularization=["no", "zero-velocity", "constant-velocity"],
)
