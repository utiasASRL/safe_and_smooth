params = dict(
    sigma_dist_est=0.05,
    sigma_acc_est_list=[1e-3],
    datasets=range(1, 17),
    inits=["gt", "half-flip", "flip"],
    use_anchors_list=["top"],
    regularization=["constant-velocity"],
)
