params = dict(
    sigma_dist_est=0.05,
    time_range=[10, 50],
    save_estimate=True,
    sigma_acc_est_list=[1e-4, 1e-3, 1e-2],
    datasets=range(1, 9),
    inits=["gt"],
    regularization=["zero-velocity", "constant-velocity"],
)
