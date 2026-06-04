"""
Plotting utilities for DIME simulation results.

Covers:
  - Gradient waveform and encoding power spectrum visualization
  - CV vs. radius (cylinders, Figure 7)
  - MD vs. radius (spheres, Figure 8)
  - Encoding efficiency scatter (Figure 6)
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .waveform import to_q_spectrum, GAMMA

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


# ---------------------------------------------------------------------------
# Waveform and encoding spectrum visualization
# ---------------------------------------------------------------------------

def gwf_plot(
    gwf: np.ndarray,
    dt: float,
    colors: list | None = None,
    linestyles: list | None = None,
    ax=None,
) -> tuple:
    """
    Plot gradient waveform axes as filled area traces.

    Parameters
    ----------
    gwf        : [n x 3] gradient waveform [T/m]
    dt         : raster time [s]
    colors     : list of colours per axis (default: grey, dark grey, red)
    linestyles : list of linestyles per axis (default: all '-')
    ax         : existing Axes

    Returns
    -------
    ax, handles
    """
    gwf = np.atleast_2d(np.asarray(gwf, dtype=float))
    if gwf.ndim == 1:
        gwf = gwf[:, None]

    n_comp = gwf.shape[1]
    t_ms   = np.arange(gwf.shape[0]) * dt * 1e3

    _colors = (colors or ["#A6A6A6", "#4d4d4d", "#bf2626"])[:n_comp]
    _ls     = (linestyles or ["-"] * n_comp)[:n_comp]

    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots()

    handles = []
    for c in range(n_comp):
        y = gwf[:, c] * 1e3  # T/m -> mT/m
        h = ax.fill_between(t_ms, 0, y, alpha=0.35,
                            facecolor=_colors[c], edgecolor=_colors[c],
                            linewidth=1.5, linestyle=_ls[c])
        ax.plot(t_ms, y, color=_colors[c], linewidth=1.5, linestyle=_ls[c])
        handles.append(h)

    gmax = np.max(np.abs(gwf)) * 1.1e3 + 1e-9
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(-gmax, gmax)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("g [mT/m]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created_ax:
        plt.tight_layout()
        plt.show()
    return ax, handles


def plot_qspectra(
    gwf: np.ndarray,
    rf: np.ndarray,
    dt: float,
    colors: list | None = None,
    linestyles: list | None = None,
    ps_scale: float = 1.0,
    xlim: tuple | None = None,
    ax=None,
    gamma: float = GAMMA,
) -> tuple:
    """
    Plot the encoding power spectrum ||Q(omega)||^2 per gradient axis.

    Each component is normalised by its own integral before plotting, so the
    y-axis reflects spectral shape rather than absolute magnitude.

    Parameters
    ----------
    gwf        : [n x 3] gradient waveform [T/m]
    rf         : [n] refocusing sign vector
    dt         : raster time [s]
    colors     : colours per axis (default: grey, dark grey, red)
    linestyles : linestyles per axis
    ps_scale   : uniform scale applied after normalisation (default 1.0)
    xlim       : (f_min, f_max) for x-axis [Hz]; default (0, f_nyquist/4)
    ax         : existing Axes
    gamma      : gyromagnetic ratio [rad/s/T]

    Returns
    -------
    ax, f, spectra  (spectra shape: [n_freq x n_comp])
    """
    gwf = np.asarray(gwf, dtype=float)
    if gwf.ndim == 1:
        gwf = gwf[:, None]

    n_comp = gwf.shape[1]
    _colors = (colors or ["#A6A6A6", "#4d4d4d", "#bf2626"])[:n_comp]
    _ls     = (linestyles or ["-"] * n_comp)[:n_comp]

    p, f = to_q_spectrum(gwf, rf, dt, gamma=gamma)
    spectra = p[:, :n_comp].real.copy()

    # Normalise each component by its integral
    for c in range(n_comp):
        A = np.trapezoid(spectra[:, c], f)
        if A > 0:
            spectra[:, c] /= A
    spectra *= ps_scale

    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots()

    for c in range(n_comp):
        y = spectra[:, c]
        ax.fill_between(f, 0, y, alpha=0.35,
                        facecolor=_colors[c], edgecolor=_colors[c], linewidth=1.5,
                        linestyle=_ls[c])
        ax.plot(f, y, color=_colors[c], linewidth=1.5, linestyle=_ls[c])

    x_max = xlim[1] if xlim else f[len(f)//4]
    x_min = xlim[0] if xlim else 0
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("Enc. power [a.u.]")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if created_ax:
        plt.tight_layout()
        plt.show()
    return ax, f, spectra


# ---------------------------------------------------------------------------
# Encoding efficiency (Figure 6)
# ---------------------------------------------------------------------------

_GAMMA = 267.513e6   # rad/s/T
_B_TARGET_SI = 2e9   # s/m^2  (= 2 ms/um^2)


def _kappa_to_tau_ms(kappa: float, gmax_mTm: float) -> float:
    """Convert encoding efficiency kappa to encoding time [ms] at b=2 ms/um^2."""
    gmax = gmax_mTm / 1000.0
    return 1e3 * (4.0 * _B_TARGET_SI / (_GAMMA**2 * gmax**2 * kappa)) ** (1.0 / 3.0)


def plot_efficiency(
    kappa: dict,
    labels: list | None = None,
    save_path: str | Path | None = None,
    ax=None,
) -> tuple:
    """
    Scatter plot of encoding time tau_STE vs tau_LTE (Figure 6).

    Converts encoding efficiency kappa to the encoding time required to reach
    b = 2 ms/um^2 via:  tau = (4b / (gamma^2 * gmax^2 * kappa))^(1/3)

    Parameters
    ----------
    kappa : nested dict with structure
            {waveform_name: {"80 mT/m": {"STE": k, "LTE": k},
                             "200 mT/m": {"STE": k, "LTE": k}}}
            Example (from Table 1 in paper)::

                kappa = {
                    "DIME":   {"80 mT/m": {"STE": 0.0295, "LTE": 0.0295},
                               "200 mT/m": {"STE": 0.0119, "LTE": 0.0119}},
                    "NOW-OE": {"80 mT/m": {"STE": 0.0482, "LTE": 0.1088},
                               "200 mT/m": {"STE": 0.0319, "LTE": 0.0922}},
                    "NOW-RM": {"80 mT/m": {"STE": 0.0482, "LTE": 0.0184},
                               "200 mT/m": {"STE": 0.0319, "LTE": 0.0118}},
                    "NOW-ET": {"80 mT/m": {"STE": 0.0521, "LTE": 0.0793},
                               "200 mT/m": {"STE": 0.0427, "LTE": 0.0624}},
                }

    labels    : display names per waveform (default: keys of kappa)
    save_path : if given, save figure at 600 dpi
    ax        : existing Axes

    Returns
    -------
    ax
    """
    waveforms = list(kappa.keys())
    if labels is None:
        labels = waveforms

    _colors = {"DIME": "black", "NOW-OE": "#A9A9A9", "NOW-RM": "#4b91e2", "NOW-ET": "#8E1616"}
    _markers = {"80 mT/m": "o", "200 mT/m": "s"}
    gmax_by_system = {"80 mT/m": 80.0, "200 mT/m": 200.0}

    # Convert kappa -> tau [ms]
    tau = {}
    for wf in waveforms:
        tau[wf] = {}
        for sys, gmax in gmax_by_system.items():
            if sys not in kappa[wf]:
                continue
            tau[wf][sys] = {
                enc: _kappa_to_tau_ms(kappa[wf][sys][enc], gmax)
                for enc in ("STE", "LTE")
            }

    all_times = [tau[wf][sys][enc]
                 for wf in waveforms for sys in tau[wf] for enc in ("STE", "LTE")]
    tmax = np.ceil(max(all_times) / 10) * 10 + 5

    created_ax = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 5.6))

    for wf, label in zip(waveforms, labels):
        col = _colors.get(wf, "black")
        for sys in tau[wf]:
            ax.scatter(tau[wf][sys]["STE"], tau[wf][sys]["LTE"],
                       s=110, color=col, marker=_markers.get(sys, "o"),
                       edgecolor="black", linewidth=0.8, zorder=3)

    xline = np.linspace(0, tmax, 200)
    ax.plot(xline, xline, "--", color="gray", linewidth=1.2, label="equal STE/LTE time", zorder=1)

    ax.set_xlim(0, tmax)
    ax.set_ylim(0, tmax)
    ax.set_xlabel(r"$\tau_\mathrm{STE}$ [ms]")
    ax.set_ylabel(r"$\tau_\mathrm{LTE}$ [ms]")
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    wf_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=_colors.get(wf, "black"),
               markeredgecolor="black", markersize=8, label=lab)
        for wf, lab in zip(waveforms, labels)
    ]
    sys_handles = [
        Line2D([0], [0], marker=_markers[sys], color="black",
               linestyle="None", markersize=8, label=sys)
        for sys in gmax_by_system
    ]

    leg1 = ax.legend(handles=wf_handles, title="Waveform", loc="upper left",
                     frameon=False, fontsize=12,
                     title_fontproperties={"weight": "bold", "size": 13})
    ax.add_artist(leg1)
    ax.legend(handles=sys_handles, title="System", loc="center left",
              frameon=False, fontsize=12,
              title_fontproperties={"weight": "bold", "size": 13})

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight")
    if created_ax:
        plt.tight_layout()
        plt.show()
    return ax
