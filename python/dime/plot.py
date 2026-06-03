"""
Plotting utilities for DIME simulation results.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mlp.rcParams["font.family"] = "Times New Roman"
mlp.rcParams.update({"font.size": 13})


# ---------------------------------------------------------------------------
# CV plots
# ---------------------------------------------------------------------------

def get_cv(path: str | Path, b_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute coefficient of variation of STE signal across rotations.

    Parameters
    ----------
    path  : path to .npz file from dime.simulate.simulate()
    b_idx : b-value index to use (default: 0)

    Returns
    -------
    cv     : CV per radius (std/mean), shape [n_radii]
    radii  : radii [m], shape [n_radii]
    """
    data  = np.load(path, allow_pickle=True)
    sig   = data["STE_sig"][:, b_idx, :]  # (n_radii, n_rot)
    radii = data["radii"]
    cv    = np.std(sig, axis=1) / np.mean(sig, axis=1)
    return cv, radii


def plot_cv_vs_radius(
    paths_80mT: list,
    paths_200mT: list,
    labels: list | None = None,
    save_path: str | Path | None = None,
    ax=None,
):
    """
    Line plot of STE signal CV vs. cylinder radius for two scanner systems.

    Parameters
    ----------
    paths_80mT  : list of .npz paths for 80 mT/m system
    paths_200mT : list of .npz paths for 200 mT/m system
    labels      : waveform labels (one per path pair)
    save_path   : if given, save figure to this path at 600 dpi
    ax          : existing Axes to plot into

    Returns
    -------
    ax : matplotlib Axes
    """
    if labels is None:
        labels = [f"wf {i+1}" for i in range(len(paths_80mT))]
    if len(paths_80mT) != len(paths_200mT):
        raise ValueError("paths_80mT and paths_200mT must have the same length")

    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots()

    colors = ["black", "#A9A9A9", "#4b91e2", "#8E1616"]
    family_handles = []

    for i, (p80, p200, label) in enumerate(zip(paths_80mT, paths_200mT, labels)):
        color = colors[i % len(colors)]
        cv_80,  r80  = get_cv(p80)
        cv_200, r200 = get_cv(p200)
        ax.plot(r80  * 1e6, cv_80  * 100, color=color, linestyle="--")
        ax.plot(r200 * 1e6, cv_200 * 100, color=color, linestyle="-")
        family_handles.append(Line2D([0], [0], color=color, linestyle="-", label=label))

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  label="200 mT/m"),
        Line2D([0], [0], color="black", linestyle="--", label="80 mT/m"),
    ]
    leg1 = ax.legend(handles=family_handles, title="Waveform",
                     loc="upper right", frameon=False, fontsize=10,
                     title_fontproperties={"weight": "bold"})
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title="System", loc="upper center",
              frameon=False, fontsize=10, title_fontproperties={"weight": "bold"})

    ax.set_xlabel("Radius [µm]")
    ax.set_ylabel("CV [%]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight")
    if created_ax:
        plt.show()
    return ax


def boxplot_cv(
    paths_80mT: list,
    paths_200mT: list,
    labels: list | None = None,
    save_path: str | Path | None = None,
    ax=None,
):
    """
    Grouped box plot of STE signal CV for two scanner systems.

    Parameters
    ----------
    paths_80mT  : list of .npz paths for 80 mT/m system
    paths_200mT : list of .npz paths for 200 mT/m system
    labels      : waveform labels
    save_path   : if given, save figure
    ax          : existing Axes

    Returns
    -------
    ax : matplotlib Axes
    """
    if labels is None:
        labels = [f"wf {i+1}" for i in range(len(paths_80mT))]

    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots()

    c200 = "#3a3a3a"
    c80  = "#b24b4b"
    box_width, pair_sep, jitter = 0.25, 0.28, 0.015

    centers = np.arange(len(labels)) + 1
    pos200  = centers - pair_sep / 2
    pos80   = centers + pair_sep / 2
    data200 = [get_cv(p)[0] * 100 for p in paths_200mT]
    data80  = [get_cv(p)[0] * 100 for p in paths_80mT]

    bp200 = ax.boxplot(data200, positions=pos200, widths=box_width, whis=(0, 100),
                       patch_artist=True, showfliers=False)
    bp80  = ax.boxplot(data80,  positions=pos80,  widths=box_width, whis=(0, 100),
                       patch_artist=True, showfliers=False)

    for box in bp200["boxes"]:
        box.set(facecolor=c200, edgecolor="black", alpha=0.95)
    for box in bp80["boxes"]:
        box.set(facecolor=c80,  edgecolor="darkred", alpha=0.9)
    for bp in [bp200, bp80]:
        for key in ["whiskers", "caps", "medians"]:
            for a in bp[key]:
                a.set(color="black", linewidth=1.0)

    rng = np.random.default_rng(123)
    for x, y in zip(pos200, data200):
        ax.plot(x + rng.uniform(-jitter, jitter, len(y)), y,
                "o", color="black", alpha=0.35, markersize=2.0)
    for x, y in zip(pos80, data80):
        ax.plot(x + rng.uniform(-jitter, jitter, len(y)), y,
                "o", color="black", alpha=0.35, markersize=2.0)

    ax.legend(handles=[
        Patch(facecolor=c200, edgecolor="black",   label="200 mT/m"),
        Patch(facecolor=c80,  edgecolor="darkred", label="80 mT/m"),
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=5, label="Samples"),
    ], loc="upper left", frameon=False)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coefficient of variation [%]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight")
    if created_ax:
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# MD vs. radius plots
# ---------------------------------------------------------------------------

def load_md_from_npz_files(paths: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Load D_STE and D_LTE from a list of fitted .npz files.

    Returns
    -------
    data   : [n_radii x 2 x n_wf] array (axis 1: STE=0, LTE=1)
    radii  : [n_radii] array [m]
    """
    with np.load(paths[0], allow_pickle=True) as f0:
        radii  = np.asarray(f0["radii"])
        n_radii = len(radii)

    data = np.zeros((n_radii, 2, len(paths)), dtype=float)
    for w, p in enumerate(paths):
        with np.load(p, allow_pickle=True) as f:
            data[:, 0, w] = f["D_STE"].astype(float)
            data[:, 1, w] = f["D_LTE"].astype(float)
    return data, radii


def plot_md_vs_radius(
    data_80mT: np.ndarray,
    data_200mT: np.ndarray,
    radii: np.ndarray,
    wf_labels: list | None = None,
    save_path: str | Path | None = None,
):
    """
    Four-panel plot of MD_STE, MD_LTE, and |delta-MD| vs. radius for each waveform.

    Parameters
    ----------
    data_80mT  : [n_radii x 2 x n_wf] from load_md_from_npz_files() for 80 mT/m
    data_200mT : same for 200 mT/m
    radii      : [n_radii] array [m]
    wf_labels  : subplot titles (default: wf 1, wf 2, ...)
    save_path  : if given, save figure

    Returns
    -------
    fig, axes
    """
    n_wf = data_80mT.shape[2]
    if wf_labels is None:
        wf_labels = [f"wf {i+1}" for i in range(n_wf)]

    colors = ["black", "#A9A9A9", "#4b91e2", "#8E1616"]

    fig, axes = plt.subplots(1, n_wf, figsize=(3 * n_wf, 3))
    fig.tight_layout()

    for i, ax in enumerate(axes):
        c80  = colors[3]
        c200 = colors[0]
        r_um = radii * 1e6

        ax.set_title(wf_labels[i])
        ax.plot(r_um, data_200mT[:, 0, i], color=c200, linestyle="-")
        ax.plot(r_um, data_200mT[:, 1, i], color=c200, linestyle="--")
        ax.plot(r_um, np.abs(data_200mT[:, 1, i] - data_200mT[:, 0, i]), color=c200, linestyle=":")
        ax.plot(r_um, data_80mT[:, 0, i],  color=c80,  linestyle="-")
        ax.plot(r_um, data_80mT[:, 1, i],  color=c80,  linestyle="--")
        ax.plot(r_um, np.abs(data_80mT[:, 1, i] - data_80mT[:, 0, i]),  color=c80,  linestyle=":")

        ax.set_xlabel("Radius [µm]")
        if i != 0:
            ax.tick_params(labelleft=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("MD [µm²/ms]")
    axes[0].legend(handles=[
        Line2D([0], [0], color="black", linestyle="-",  label=r"MD$_\mathrm{STE}$"),
        Line2D([0], [0], color="black", linestyle="--", label=r"MD$_\mathrm{LTE}$"),
        Line2D([0], [0], color="black", linestyle=":",  label=r"$\Delta$MD"),
    ], title="Encoding", loc="center right", frameon=False,
       title_fontproperties={"weight": "bold"})
    axes[1].legend(handles=[
        Line2D([0], [0], color=colors[0], linestyle="-", label="200 mT/m"),
        Line2D([0], [0], color=colors[3], linestyle="-", label="80 mT/m"),
    ], title="System", loc="center right", frameon=False,
       title_fontproperties={"weight": "bold"})

    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig, axes
