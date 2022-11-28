params = dict(
    sigma_dist_est=0.05,
    save_estimate=True,
    sigma_acc_est_list=[1e-3],
    datasets=[1, 2],
    inits=["gt", "half-flip", "flip"],
    regularization=["constant-velocity"],
)
