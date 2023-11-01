#!/usr/bin/env python
import os

import numpy as np
import matplotlib.pylab as plt
import pandas as pd

fn_tp_styles = {
    "t.p.": ("C2", "<"),  # green
    "t.n.": ("C0", ">"),  # blue
    "f.n.": ("C1", "o"),  # orange
    "f.p.": ("C3", "x"),  # red
}


def add_colorbar(fig, ax, im, title=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    if title is not None:
        cax.set_ylabel(title)
    return cax


def add_scalebar(
    ax, size=5, size_vertical=1, loc="lower left", fontsize=8, color="black", pad=0.1
):
    """Add a scale bar to the plot.

    :param ax: axis to use.
    :param size: size of scale bar.
    :param size_vertical: height (thckness) of the bar
    :param loc: location (same syntax as for matplotlib legend)
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import matplotlib.font_manager as fm

    fontprops = fm.FontProperties(size=fontsize)
    scalebar = AnchoredSizeBar(
        ax.transData,
        size,
        "{} m".format(size),
        loc,
        pad=pad,
        color=color,
        frameon=False,
        size_vertical=size_vertical,
        fontproperties=fontprops,
    )
    ax.add_artist(scalebar)


def add_labels(ax, coords, **kwargs):
    # no annotation if only one element.
    if not (coords.shape[0] > 1):
        return

    avg_dist = np.linalg.norm(coords[None, :, :] - coords[:, None, :], axis=2)
    label_dist = min(min(avg_dist[avg_dist > 0]) / 5, 0.1)
    for i, coord in enumerate(coords):
        ax.annotate(f"{i}", coord + label_dist, **kwargs)


def add_lines(ax, trajectory, anchors, W):
    assert W.shape == (anchors.shape[0], trajectory.shape[0])
    for n, x in enumerate(trajectory):
        for m, a in enumerate(anchors):
            if W[m, n] > 0:
                ax.plot([x[0], a[0]], [x[1], a[1]], color=f"C{n}", ls=":")


def make_dirs_safe(path):
    """Make directory of input path, if it does not exist yet."""
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def savefig(fig, name, verbose=True):
    make_dirs_safe(name)
    ext = name.split(".")[-1]
    fig.savefig(name, bbox_inches="tight", pad_inches=0, dpi=200)
    if verbose:
        print(f"saved plot as {name}")


def remove_ticks(ax):
    """Remove all ticks and margins from plot."""
    for ax_name in ["x", "y"]:
        ax.tick_params(
            axis=ax_name,
            which="both",
            bottom=False,
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.1)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())


def plot_decomposition(
    log=False,
    colorbar=False,
    vmin=None,
    vmax=None,
    shape=None,
    cmap="viridis",
    return_all=False,
    titles=None,
    **matrix_dict,
):
    import matplotlib

    old = plt.rcParams["axes.ymargin"]
    plt.rcParams["axes.ymargin"] = 0.0

    fig = plt.figure()
    if shape is None:
        shape = (1, len(matrix_dict))
    else:
        assert shape[0] * shape[1] >= len(matrix_dict)
    fig.set_size_inches(3 * shape[1], 3 * shape[0])
    gs = matplotlib.gridspec.GridSpec(
        shape[0],
        shape[1],
        figure=fig,
        width_ratios=[m.shape[1] for m in matrix_dict.values()][: shape[1]],
    )
    axs = []
    for i, (key, matrix) in enumerate(matrix_dict.items()):
        if titles is not None:
            title = titles[key]
        else:
            title = key
        im = None
        row_i = i // shape[1]
        i_here = i - row_i * shape[1]
        ax0 = fig.add_subplot(gs[row_i, i_here])
        ax0.set_anchor("S")
        axs.append(ax0)
        ny, nx = matrix.shape
        ax0.set_xticks(np.arange(-0.5, nx, step=1.0), minor=True)
        ax0.set_yticks(np.arange(-0.5, ny, step=1.0), minor=True)
        if log:
            plot_matrix = np.full(matrix.shape, np.nan)
            plot_matrix[np.abs(matrix) > 1e-10] = np.log10(
                np.abs(matrix[np.abs(matrix) > 1e-10])
            )
            im = ax0.matshow(
                plot_matrix, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest"
            )
            ax0.set_title(title + " (log)", y=1.0)
        else:
            im = ax0.matshow(
                matrix, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest"
            )
            ax0.set_title(title, y=1.0)
        if colorbar:
            add_colorbar(fig, ax0, im)
        ax0.set_xticks([], minor=False)
        ax0.set_yticks([], minor=False)
        ax0.xaxis.set_ticklabels([])
        ax0.grid(which="minor", linestyle="-", linewidth=0.2, color="k")
    fig.tight_layout()
    # plt.rcParams['axes.ymargin'] = old
    if return_all:
        return fig, axs
    return fig


def plot_aligned(trajectory, anchors, trajectory_est, anchors_est, ax=None):
    from scipy.linalg import orthogonal_procrustes

    # find optimal t, R based on anchors.
    t = np.mean(anchors, axis=0)
    anchors_centered = anchors - t
    trajectory_centered = trajectory - t
    t_est = np.mean(anchors_est, axis=0)
    anchors_est_centered = anchors_est - t_est
    R_optimal, scale = orthogonal_procrustes(anchors_est_centered, anchors_centered)
    # print('determinant of R_optimal:', round(np.linalg.det(R_optimal)))

    # transfrom estimated trajectory and anchors
    anchors_est_rot = R_optimal.T.dot(anchors_est_centered.T).T
    trajectory_rot = R_optimal.T.dot((trajectory_est - t_est).T).T

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
    ax.scatter(*trajectory_rot[:, :2].T, color="C0")
    ax.scatter(*anchors_est_rot[:, :2].T, color="C1")
    ax.scatter(*trajectory_centered[:, :2].T, color="C0", marker="x", s=100)
    ax.scatter(*anchors_centered[:, :2].T, color="C1", marker="x", s=100)
    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def plot_setup(trajectory, anchors, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
    ax.scatter(*trajectory[:, :2].T, **kwargs)
    ax.scatter(*anchors[:, :2].T, color="k", marker="x")
    ax.axis("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax


def jitter(values):
    return 10 ** (np.log10(values.astype(float)) + np.random.rand(*values.shape) * 0.3)


def fill_fn_tp(row):
    if pd.isna(row["cert value"]):
        row.result = None
    elif (row["cert value"] >= 0) and (row.solution == "global"):
        row.result = "t.p."
    elif (row["cert value"] >= 0) and (row.solution == "local"):
        row.result = "f.p."
    elif (row["cert value"] < 0) and (row.solution == "local"):
        row.result = "t.n."
    elif (row["cert value"] < 0) and (row.solution == "global"):
        row.result = "f.n."
    return row


def plot_cert_fn_tp(
    results, xlabel="sigma dist real", ylabel="rmse", ax=None, xticks=None, **kwargs
):
    if not "result" in results.columns:
        results.loc[:, "result"] = ""
        results = results.apply(fill_fn_tp, axis=1)

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 3)
    else:
        fig = plt.gcf()

    for label, (c, s) in fn_tp_styles.items():
        df = results[results.result == label]
        ax.scatter(
            df[xlabel].values,
            df[ylabel].values,
            color=c,
            marker=s,
            label=label,
            **kwargs,
        )
    ax.set_xlabel("distance noise", fontsize=12)
    ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        # ax.set_xticklabels([f"$10^{{{np.log10(e):.0f}}}$" for e in xticks])
    ax.set_yscale("log")
    ax.grid(which="major")
    ax.set_axisbelow(True)  # make sure grid is below points
    plt.minorticks_off()
    fig.tight_layout()
    return fig, ax


def plot_3d_curves(
    trajs, anchors=None, show=True, plotly=True, ax=None, styles={}, zlabel=False
):
    # generate nice interactive graphic for notebook
    if plotly:
        import plotly
        import plotly.graph_objs as go

        plotly.offline.init_notebook_mode()

        traces = []
        for label, traj in trajs.items():
            trace = go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="markers",
                marker={
                    "size": 1,
                    "opacity": 0.8,
                },
                name=label,
            )
            traces.append(trace)
        if anchors is not None:
            K = anchors.shape[0]
            trace_anchors = go.Scatter3d(
                x=anchors[list(range(K)) + [0], 0],
                y=anchors[list(range(K)) + [0], 1],
                z=anchors[list(range(K)) + [0], 2],
                marker={
                    "size": 1,
                    "opacity": 0.8,
                },
                name="anchors",
            )
            traces.append(trace_anchors)
            # layout = go.Layout()
        fig = go.Figure(data=traces)
        if show:
            plotly.offline.iplot(fig)
        return fig
    # generate good graphic for saving
    else:
        if ax is None:
            fig = plt.figure()
            ax = plt.add_subplot(1, 1, 1, projection="3d")
        for label, traj in trajs.items():
            kwargs = styles.get(label, {})
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                label=label,
                rasterized=True,
                **kwargs,
            )

        if anchors is not None:
            ax.scatter(
                anchors[:, 0],
                anchors[:, 1],
                anchors[:, 2],
                marker="x",
                s=20,
                color="black",
                label="UWB anchor",
            )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_zticks([0, 2, 4])
        if zlabel:
            ax.set_zlabel("z [m]", labelpad=-15, fontsize=12)
        else:
            ax.set_zticklabels([])
        ax.view_init(elev=10, azim=-45)
        ax.set_xlabel("x [m]", labelpad=-15, fontsize=12)
        ax.set_ylabel("y [m]", labelpad=-15, fontsize=12)
        ax.dist = 8


def plot_covariance(Cov, centre, scaling, ax, log=False, **kwargs):
    from matplotlib.patches import Ellipse

    lamda_, vec = np.linalg.eig(Cov)
    if log:
        lamda_ = np.log10(np.sqrt(lamda_)) * scaling
    else:
        lamda_ = np.sqrt(lamda_) * scaling
    # print(lamda_)
    ell = Ellipse(
        xy=centre,
        width=lamda_[0],
        height=lamda_[1],
        angle=np.rad2deg(np.arccos(vec[0, 0])),
        **kwargs,
    )
    ax.add_artist(ell)
    # ax.scatter(*centre, color="C2", marker=".")
