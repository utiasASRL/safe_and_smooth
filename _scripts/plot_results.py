import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from poly_certificate.datasets import ANCHOR_CHOICE
from poly_certificate.gauss_newton import gauss_newton
from poly_certificate.problem import Problem
from poly_certificate.sdp_setup import get_prob_matrices

from poly_certificate.utils.plotting_tools import add_scalebar
from poly_certificate.utils.plotting_tools import fill_fn_tp, fn_tp_styles
from poly_certificate.utils.plotting_tools import plot_3d_curves
from poly_certificate.utils.plotting_tools import plot_cert_fn_tp
from poly_certificate.utils.plotting_tools import plot_decomposition
from poly_certificate.utils.plotting_tools import savefig
from poly_certificate.utils.helper_params import parse_arguments
from poly_certificate.utils.helper_params import load_parameters



import shutil
latex = False
if shutil.which('latex'):
    latex = True
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )
    plt.rc("text.latex", preamble=r"\usepackage{bm}\usepackage{color}")
figsize = 7


# Problem setup
def plot_problem_setup(plotdir):
    # N = 100
    N = 3
    d = 3
    K = 7
    sigma_dist_real = 0.0
    sigma_acc_real = 0.2

    np.random.seed(0)

    prob = Problem(N, d, K)
    prob.generate_random(sigma_acc_real=sigma_acc_real, sigma_dist_real=sigma_dist_real)

    regs = ["no", "zero-velocity", "constant-velocity"]
    all_dict = {}
    for reg in regs:
        Q, A_0, A_list, R = get_prob_matrices(prob, regularization=reg)
        if type(A_0) is list:
            mat_dict = {f"$$A_{n}$$": A_n.toarray() for n, A_n in enumerate(A_0)}
        mat_dict = dict(
            Q=Q.toarray(),
            A1=A_list[0].toarray(),
        )
        if R is not None:
            mat_dict["R"] = R.toarray()

        if (R is not None) and (N < 10):
            all_dict.update(
                {
                    f"Q\_{reg[0]}": np.abs(Q.toarray()) > 1e-10,
                    f"R\_{reg[0]}": np.abs(R.toarray()) > 1e-10,
                }
            )

    fig, axs = plot_decomposition(
        **all_dict,
        colorbar=False,
        log=False,
        cmap="Greys",
        return_all=True,
        titles=None,
    )
    [ax.set_title("") for ax in axs]
    if latex:
        [ax.set_ylabel("$\\bm{{Q}}^{{(g)}}$", rotation=0) for ax in [axs[0], axs[2]]]
        [ax.set_ylabel("$\\bm{{R}}^{{(g)}}$", rotation=0) for ax in [axs[1], axs[3]]]
    fig.set_size_inches(6, 2.0)
    fig.tight_layout()
    axs[0].xaxis.set_label_coords(1.3, -0.09)
    axs[2].xaxis.set_label_coords(1.2, -0.05)
    axs[0].yaxis.set_label_coords(1.25, 0)
    axs[1].yaxis.set_label_coords(1.25, 0)
    axs[2].yaxis.set_label_coords(1.15, 0)
    axs[3].yaxis.set_label_coords(1.15, 0)
    [ax.yaxis.set_ticks_position("none") for ax in axs]
    [ax.xaxis.set_ticks_position("none") for ax in axs]
    savefig(fig, os.path.join(plotdir, "Q_all.pdf"))


# Certificate study with fixed N, increasing noise
def plot_noise(outdir, plotdir):
    def remove_invalid_rows(results, verbose=False):
        """For very few noise levels and setup seeds, all initializations lead to local minima,
        accoding to the certificate.
        Since we don't have global minima to compare to, we remove these from the results.
        """
        remove_index = []
        for (reg, noise), df_noise in results.groupby(
            ["regularization", "sigma dist real"]
        ):
            if verbose:
                print(reg, noise)
            for seed, df in df_noise.groupby("setup seed"):
                local_ = df.loc[df["cert value"] < 0]
                inits_local = local_["init seed"].unique()
                if len(inits_local) == 0:
                    continue

                if verbose:
                    print(
                        f"for seed {seed}, inits leading to local minima:", inits_local
                    )
                if len(inits_local) == len(df):
                    print(f"noise {noise} invalid seed:", seed)
                    remove_index += list(df.index.values)
        return results.drop(index=remove_index, inplace=False)

    name = "simulation_noise"
    fname = os.path.join(outdir, name, "results.pkl")
    try:
        results = pd.read_pickle(fname)
        print("read", fname)
    except FileNotFoundError:
        print(f"{fname} not found, generate with simulate_noise.py")

    print("number of rows before:", len(results))
    results = remove_invalid_rows(results, verbose=False)
    print("number of rows after:", len(results))

    regularization = results.regularization.unique()

    fig_bin, axs_bin = plt.subplots(1, len(regularization), sharey=True, squeeze=False)
    fig_bin.set_size_inches(figsize * 0.85, figsize * 0.35)

    ylabel = "error" if "error" in results.columns else "rmse"

    labels = {key: 0 for key in fn_tp_styles.keys()}
    for i, m in enumerate(regularization):
        df = results[results.regularization == m].copy()
        axs_bin[0, i].set_xscale("log")
        axs_bin[0, i].set_yscale("log")

        df.loc[:, "result"] = ""
        df = df.apply(fill_fn_tp, axis=1)

        # reduce number of datapoints because otherwise the
        # final plot will be too large
        df["error_id"] = [f"{y:.5e}" for y in df[ylabel].values]
        df["error_id"] = df["error_id"].astype(float)
        df_small = df.groupby("error_id", as_index=False).aggregate("first")
        plot_cert_fn_tp(
            df_small,
            xlabel="noise",
            ylabel="error_id",
            ax=axs_bin[0, i],
            s=20.0,
            alpha=0.5,
        )
        # plot_cert_fn_tp(df, xlabel="sigma dist real", ax=axs_bin[0, i], s=20.0, alpha=0.5)

        xticks = [1e-2, 1e0, 1e2, 1e4]
        axs_bin[0, i].set_xticks(xticks)
        # axs_bin[0, i].set_xticklabels(xticks)
        axs_bin[0, i].set_title(f"{m} prior", fontsize=12)
        for label in labels.keys():
            labels[label] += len(df[df.result == label])

    labels_with_num = [f"{key}: {val}" for key, val in labels.items()]
    fig_bin.legend(
        labels=labels_with_num,
        loc="lower right",
        bbox_to_anchor=[0.98, 0.21],
        fontsize=10,
        handlelength=0.3,
    )

    # axs_bin[0, 0].set_ylim(1e-4, 1e4)
    # axs_bin[0, 0].set_yticks([1e-3, 1e-1, 1e1, 1e3])
    axs_bin[0, 0].set_ylabel("RMSE", fontsize=12)
    fig_bin.tight_layout()
    fig_bin.subplots_adjust(wspace=0.1)

    savefig(fig_bin, os.path.join(plotdir, f"{name}.pdf"))
    savefig(fig_bin, os.path.join(plotdir, f"{name}.jpg"))

    params = load_parameters(name, outdir)
    regularization = "constant-velocity"
    # regularization = "no"
    sigma_dist_real = params["sigma_dist_real"][1]
    sigma_acc_est = params["sigma_acc_est"][0]
    print("sigmas:", sigma_dist_real, sigma_acc_est)

    df = results.loc[
        (results.regularization == regularization)
        & (results["sigma dist real"] == sigma_dist_real)
    ].copy()

    counter = 0

    N = 100
    d = 2
    K = 6

    n_local = 3
    n_examples = 6
    n_examples = min(n_examples, len(df[df.solution == "local"]["setup seed"].unique()))
    print("number of setups with local solutions:", n_examples)

    fig, axs = plt.subplots(2, n_examples, sharex="col", squeeze=False)
    # fig.set_size_inches(size * n_examples, size * 2.3)
    fig.set_size_inches(figsize, figsize / n_examples * 2.3)
    fig.subplots_adjust(wspace=0.05)
    for n_it, df_ in df.groupby("setup seed"):
        if not len(df_[df_.solution == "local"]):
            continue

        n_total = len(df_)
        n_local_sol = len(df_[df_.solution == "local"])
        n_global_sol = len(df_[df_.solution == "global"])

        print(f"found {n_local_sol} local solutions for setup", n_it)
        if n_local_sol == 1:
            continue

        print("plotting...")
        np.random.seed(n_it)
        prob = Problem(
            N, d, K, sigma_acc_est=sigma_acc_est, sigma_dist_est=sigma_dist_real
        )
        prob.generate_random(
            sigma_dist_real=sigma_dist_real,
            sigma_acc_real=sigma_acc_est,
            verbose=False,
        )

        # sanity check that we are generating the same setup
        test_noise = prob.calculate_noise_level()
        np.testing.assert_allclose(test_noise, df_.noise.unique()[0])

        axs[0, counter].plot(
            prob.trajectory[:, 0],
            prob.trajectory[:, 1],
            color="k",
            label="ground truth",
        )
        axs[1, counter].plot(
            prob.trajectory[:, 0],
            prob.trajectory[:, 1],
            color="k",
            label="ground truth",
        )

        df_tn = df_[df_.solution == "global"]
        row = df_tn.iloc[0]
        assert row["cert value"] >= 0

        np.random.seed(row["init seed"])
        theta_0 = prob.random_traj_init(regularization)
        theta_est, stats = gauss_newton(theta_0, prob, regularization=regularization)

        axs[0, counter].plot(
            theta_est[:, 0], theta_est[:, 1], color="C1", ls="--", label="estimates"
        )

        df_tp = df_[df_.solution == "local"]
        for i in range(min(n_local, len(df_tp))):
            row = df_tp.iloc[i]
            assert row["cert value"] < 0

            np.random.seed(row["init seed"])
            theta_0 = prob.random_traj_init(regularization)
            theta_est, stats = gauss_newton(
                theta_0, prob, regularization=regularization
            )

            label = "estimates" if i == 1 else None
            axs[1, counter].plot(
                theta_est[:, 0], theta_est[:, 1], color=f"C{i}", ls="--", label=label
            )

        axs[0, counter].scatter(
            *prob.anchors[:, :2].T, marker="x", color="k", label="anchor points"
        )
        axs[1, counter].scatter(
            *prob.anchors[:, :2].T, marker="x", color="k", label="anchor points"
        )
        axs[0, counter].axis("equal")
        axs[1, counter].axis("equal")

        [ax.set_xticks([]) for ax in axs.flatten()]
        [ax.set_yticks([]) for ax in axs.flatten()]
        # [add_scalebar(ax, size=50, loc="upper right") for ax in axs[0, :]]

        axs[0, 0].set_ylabel("certificate \nholds", fontsize=12)
        axs[1, 0].set_ylabel("certificate \nfails", fontsize=12)

        axs[0, counter].set_xlabel(f"{n_global_sol}/{n_total}", y=0.0, fontsize=12)
        axs[1, counter].set_xlabel(f"{n_local_sol}/{n_total}", y=0.0, fontsize=12)
        axs[0, counter].xaxis.set_label_coords(0.5, -0.03)
        axs[1, counter].xaxis.set_label_coords(0.5, -0.03)

        for ax in axs.flatten():
            for spine in ax.spines.values():
                spine.set_edgecolor("lightgray")
            ax.set_facecolor("lightgray")

        counter += 1
        if counter >= n_examples:
            break
    leg = axs[1, -1].legend(
        ncol=3, loc="upper right", bbox_to_anchor=[0.8, -0.15], fontsize=12
    )
    # leg.get_frame().set_facecolor("lightgray")
    leg.get_frame().set_edgecolor("lightgray")
    savefig(fig, os.path.join(plotdir, "local_minimum_sim.pdf"))


def plot_timing(outdir, plotdir):
    labels = {
        "solve": "solve (GN)",
        "duals": "compute dual variables",
        "certificate": "compute certificate",
    }
    try:
        fname = os.path.join(outdir, "simulation_time.pkl")
        results = pd.read_pickle(fname)

        fig, ax = plt.subplots()
        fig.set_size_inches(figsize / 2, 2)
        for method, df in results.groupby("method", sort=False):
            ax.plot(df.N, df.time, ls="--", label=labels[method])

        # ax.legend(loc="lower right")
        # ax.legend(loc="upper left", bbox_to_anchor=[1.0, 1.0])
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(which="major")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_off()
        yticks = np.logspace(-2, 2, 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"$10^{{{np.log10(y):.0f}}}$" for y in yticks])
        ax.set_ylabel("runtime [s]", fontsize=12)
        ax.set_xlabel("number of positions $N$", fontsize=12)
        savefig(fig, os.path.join(plotdir, "certificates_timing.pdf"))
    except Exception as e:
        print(e)
        print(f"{fname} not found, generate with simulate_time.py")


def plot_real_top_estimate(outdir, plotdir):
    name = "real_top_estimate"
    fname = os.path.join(outdir, name, "results.pkl")
    results = pd.read_pickle(fname)

    inits = ["gt", "half-flip"]
    solutions = ["global", "local"]

    for (dataset, regularization), df in results.groupby(["dataset", "regularization"]):
        prob = Problem.init_from_dataset(
            dataset, traj_only=True, use_anchors=ANCHOR_CHOICE["top"]
        )

        fig3d = plt.figure()
        fig3d.set_size_inches(figsize, figsize / len(inits))

        zlabel = False
        titles = [
            "certified global minimum",
            "uncertified local minimum",
        ]
        for i, (init, sol) in enumerate(zip(inits, solutions)):
            rows = df[df.init == init]
            if len(rows) != 1:
                print("skipping", rows)

            row = rows.iloc[0]
            assert len(rows) == 1

            ax3d = fig3d.add_subplot(1, len(inits), i + 1, projection="3d")
            styles = {
                "estimate": dict(color="C1", s=5, marker="."),
                "ground truth": dict(color="k", s=5, marker="."),
            }
            plot_3d_curves(
                {"ground truth": row["ground truth"], "estimate": row["estimate"]},
                anchors=prob.anchors,
                plotly=False,
                styles=styles,
                ax=ax3d,
                zlabel=zlabel,
            )
            ax3d.set_zlim(0, 5)
            ax3d.set_title(titles[i], y=0.96, loc="center", fontsize=12)
            # ax3d.set_rasterized(True)
            zlabel = True
        ax3d.legend(
            ncol=3, bbox_to_anchor=[-0.2, 0.05], loc="upper center", fontsize=12
        )
        # fig3d.subplots_adjust(wspace=0.1)
        savefig(fig3d, os.path.join(plotdir, f"real_certificate_{dataset}.pdf"))


def plot_real_top(outdir, plotdir):
    name = "real_top"
    fname = os.path.join(outdir, name, "results.pkl")
    # fname = "_results_server/real_top/results.pkl"
    # fname = "_results/real_top/results.pkl"
    results = pd.read_pickle(fname)

    print("average cert time:", results["time cert"].mean())
    print("average solve time:", results["time solve"].mean())
    print(
        "average total time:",
        results["time cert"].mean() + results["time solve"].mean(),
    )

    def plot_rmse_cost(results):
        results = results.apply(pd.to_numeric, errors="ignore", axis=1)
        try:  # older version
            results.loc[:, "certificate binary"] = results.certificate >= 0
        except AttributeError:  # newer version
            results.loc[:, "certificate binary"] = results.cert >= 0

        fig, ax = plt.subplots()
        styles = {True: "x", False: "o"}
        sizes = {True: 50, False: 20}
        cmap = plt.get_cmap("viridis", len(results.dataset.unique()))
        # for i, (dataset, df) in enumerate(results.groupby("dataset")):
        for i in range(1, 17)[::-1]:
            dataset = f"trial{i}"
            df = results[results.dataset == dataset]
            if dataset == "trial1":
                color = "C1"
                label = dataset
            elif dataset == "trial13":
                color = "C2"
                label = dataset
            elif dataset == "trial16":
                color = "C0"
                label = dataset
            else:
                color = "lightgray"
                label = None
            for cert, df_cert in df.groupby("certificate binary"):
                ax.scatter(
                    df_cert.RMSE.values,
                    df_cert.cost.values,
                    color=color,
                    marker=styles[cert],
                    s=sizes[cert],
                )
            ax.scatter(
                [], [], color=color, marker=styles[True], s=sizes[True], label=label
            )
        ax.scatter(
            [],
            [],
            color="lightgray",
            marker=styles[True],
            s=sizes[True],
            label="all others",
        )
        # ax.scatter([], [], color="gray", marker=styles[True], label="certificate holds")
        # ax.scatter([], [], color="gray", marker=styles[False], label="certificate fails")
        ax.legend(loc="upper center", bbox_to_anchor=[0.4, 0.95], fontsize=10)
        ax.set_xlabel("RMSE $[$m$]$", fontsize=12)
        ax.set_ylabel("total cost", fontsize=12)
        ax.grid()
        fig.set_size_inches(figsize / 2, 2)
        return fig

    fig = plot_rmse_cost(results)
    savefig(fig, os.path.join(plotdir, "real-cost-rmse.pdf"))


def plot_real_top_calib(outdir, plotdir):
    name = "real_top_calib"
    fname = os.path.join(outdir, name, "results.pkl")
    results = pd.read_pickle(fname)

    init = "gt"

    datasets = results.dataset.unique()[:1]
    # sigmas = results.sigma.unique()
    sigmas = results.sigma.unique()[-3:]
    regularization = results.regularization.unique()

    fig, axs = plt.subplots(
        len(datasets), len(sigmas) * len(regularization), squeeze=False
    )
    fig.set_size_inches(figsize, 1 * len(datasets))

    for i, dataset in enumerate(datasets):
        df = results.loc[results.dataset == dataset]
        # to get the anchors locations
        prob = Problem.init_from_dataset(
            dataset, traj_only=True, use_anchors=ANCHOR_CHOICE["top"]
        )
        for j1, reg in enumerate(regularization):
            for j2, sigma in enumerate(sigmas):
                j = j1 * len(sigmas) + j2

                rows = df.loc[
                    (df.regularization == reg) & (df.sigma == sigma) & (df.init == init)
                ]

                if len(rows) != 1:
                    print("skipping:", rows)
                    continue

                row = rows.iloc[0]
                color = f"C{j1}"
                axs[i, j].plot(
                    row["ground truth"][:, 0],
                    row["ground truth"][:, 1],
                    color="k",
                    ls="-",
                    rasterized=True,
                )
                axs[i, j].scatter(
                    prob.anchors[:, 0], prob.anchors[:, 1], color="k", marker="x"
                )

                axs[i, j].scatter(
                    row.estimate[:, 0],
                    row.estimate[:, 1],
                    color=color,
                    s=1.0,
                    rasterized=True,
                )

                if i == 0:
                    axs[i, j].set_title(f"$\\sigma_a$={sigma}", y=0.95)
                print(f"dataset{row.dataset} reg:{row.regularization}", end=" ")
                try:
                    print(f"rmse: {row.RMSE} cert: {row.certificate}")
                except AttributeError:
                    print(f"rmse: {row.RMSE} cert: {row.cert}")

                axs[i, j].grid("on", which="both")
                axs[i, j].xaxis.set_ticks_position("none")
                axs[i, j].yaxis.set_ticks_position("none")
                axs[i, j].set_xticklabels([])
                axs[i, j].set_yticklabels([])
                axs[i, j].set_xlim(-4.5, 4.5)
                axs[i, j].set_ylim(-4.5, 4.5)

                labels = {
                    "MAE": row.MAE,
                    "RMSE": row["RMSE unbiased"],
                }
                if j == 0:
                    label = "\n".join([li + f" {vi:.2f}m" for li, vi in labels.items()])
                    print(label)
                    axs[i, j].set_xlabel(label, fontsize=12)
                    axs[i, j].xaxis.set_label_coords(0.4, -0.05)
                else:
                    label = "\n".join([f" {vi:.2f}m" for vi in labels.values()])
                    print(label)
                    axs[i, j].set_xlabel(label, fontsize=12)
                    axs[i, j].xaxis.set_label_coords(0.5, -0.05)
        # axs[i, 0].set_xlabel(f"RMSE: {row.RMSE:.2f} m", fontsize=12)
        if len(datasets) > 1:
            axs[i, 0].set_ylabel(dataset, fontsize=12)
        axs[i, 0].yaxis.set_label_coords(-0.02, 0.5)
    add_scalebar(
        axs[0, 0],
        size=2,
        size_vertical=2.0,
        loc="upper left",
        fontsize=10,
        color="white",
        pad=0.01,
    )
    add_scalebar(
        axs[0, 0],
        size=2,
        size_vertical=0.2,
        loc="upper left",
        fontsize=10,
        color="black",
        pad=0.2,
    )
    plt.figtext(
        0.17, 1.08, regularization[0] + " prior ($\\sigma_a$ in m/s)", fontsize=12
    )  # ,color="C0")
    plt.figtext(
        0.54, 1.08, regularization[1] + " prior ($\\sigma_a$ in m/s$^{2}$)", fontsize=12
    )  # ,color="C1")
    # savefig(fig, "_plots/real-solutions.jpg")
    savefig(fig, os.path.join(plotdir, "real-solutions.pdf"))

    results
    datasets = results.dataset.unique()

    ls = {"zero-velocity": ":", "constant-velocity": "-"}
    sigma = 1e-3

    df = results.loc[results.sigma == sigma]

    times = {"zero-velocity": [], "constant-velocity": []}

    fig, axs = plt.subplots(2, sharex=True)
    fig.set_size_inches(figsize, 3)

    ax = axs[0]
    ax_var = axs[1]

    legends = []
    for reg, df_reg in df.groupby("regularization"):
        error_means = {0: [], 1: [], 2: []}  # xyz
        error_vars = {0: [], 1: [], 2: []}
        for dataset_i, (dataset, df) in enumerate(df_reg.groupby("dataset")):
            assert len(df) == 1
            row = df.iloc[0]
            for i in range(3):
                error_means[i].append(abs(row.error_xyz[i]))
                error_vars[i].append(0.5 * np.sqrt(row.var_xyz[i]))
            times[reg].append(row["time cert"] + row["time solve"])

        ax.semilogy(
            range(len(datasets)), error_means[0], label="x", ls=ls[reg], color="C0"
        )
        ax.semilogy(
            range(len(datasets)), error_means[1], label="y", ls=ls[reg], color="C1"
        )
        ax.semilogy(
            range(len(datasets)), error_means[2], label="z", ls=ls[reg], color="C2"
        )

        (linex,) = ax_var.semilogy(
            range(len(datasets)), error_vars[0], label="x", ls=ls[reg], color="C0"
        )
        (liney,) = ax_var.semilogy(
            range(len(datasets)), error_vars[1], label="y", ls=ls[reg], color="C1"
        )
        (linez,) = ax_var.semilogy(
            range(len(datasets)), error_vars[2], label="z", ls=ls[reg], color="C2"
        )
        legends.append(
            {"title": reg, "handles": [linex, liney, linez], "labels": ["x", "y", "z"]}
        )

    ax_var.set_ylabel("$\\sigma$ [m]", fontsize=12)
    ax.set_ylabel("MAE [m]", fontsize=12)

    ax.set_ylim(1e-4, 0.2)
    ax.grid()
    ax_var.grid()
    ax_var.set_xticks(range(len(datasets)))
    ax_var.set_ylim(0.02, 0.3)
    ax_var.set_yticks([0.05, 0.1, 0.2])
    ax_var.set_yticklabels([0.05, 0.1, 0.2])

    ax_var.set_xticklabels(f"{i+1}" for i in range(len(datasets)))
    ax_var.set_xlabel("trial number", fontsize=12)

    kwargs = {
        "borderpad": 0.1,
        "labelspacing": 0.1,
        "handlelength": 0.5,
        "ncol": 3,
        "edgecolor": "white",
    }
    leg = ax.legend(**legends[0], bbox_to_anchor=[0.0, 0.0], loc="lower left", **kwargs)
    ax.legend(**legends[1], bbox_to_anchor=[1.0, 0.0], loc="lower right", **kwargs)
    ax.add_artist(leg)
    plt.minorticks_off()
    savefig(fig, os.path.join(plotdir, "real-analysis.pdf"))
    savefig(fig, os.path.join(plotdir, "real-analysis.jpg"))

    for key, val in times.items():
        print(f"average time for {key} (reduced dataset): {np.mean(val):.2f}s")


if __name__ == "__main__":

    args = parse_arguments("Plot all results")

    plot_problem_setup(args.plotdir)

    plot_noise(args.resultdir, args.plotdir)
    plot_timing(args.resultdir, args.plotdir)

    plot_real_top(args.resultdir, args.plotdir)
    plot_real_top_calib(args.resultdir, args.plotdir)
    plot_real_top_estimate(args.resultdir, args.plotdir)

