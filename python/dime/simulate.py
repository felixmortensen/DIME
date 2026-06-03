"""
Monte Carlo signal simulation via Disimpy.
"""

import numpy as np
from pathlib import Path
from disimpy import substrates, simulations

from .waveform import load_mat


def simulate(
    folder: str | Path,
    out_folder: str | Path | None = None,
    geometry: str = "cylinder",
    radii: tuple | np.ndarray = (10e-6,),
    D: float = 2e-9,
    n_walkers: int = int(1e6),
    do_powder: bool = True,
    seed: int = 123,
    quiet: bool = False,
) -> list[Path]:
    """
    Run Disimpy Monte Carlo simulations for all .mat waveform files in a folder.

    For each waveform file the function simulates signals for STE and LTE
    encodings across all requested radii and saves results to .npz files.

    Parameters
    ----------
    folder     : directory containing waveform .mat files
    out_folder : output directory (default: folder/signals_py)
    geometry   : 'cylinder' or 'sphere'
    radii      : iterable of substrate radii [m]
    D          : free diffusivity [m^2/s]
    n_walkers  : number of random walkers
    do_powder  : if True, also save rotation-averaged signals
    seed       : random seed
    quiet      : suppress Disimpy output

    Returns
    -------
    List of paths to the saved .npz files.
    """
    folder = Path(folder)
    if out_folder is None:
        out_folder = folder / "signals_py"
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    if np.isscalar(radii):
        radii = [radii]
    radii = np.asarray(radii, dtype=float)

    fnl_wf = sorted(folder.glob("*.mat"))
    print(f"Found {len(fnl_wf)} waveform files in {folder}")

    out_files = []

    for file in fnl_wf:
        wf = load_mat(file)
        print(f"Processing {wf.onam}")

        dt    = float(wf.dt)
        bvals = np.asarray(wf.bval).ravel().astype(float)
        n_b   = len(bvals)
        n_rot = int(wf.nrot)

        GWF_ste, GWF_lte = _split_gwf(wf)
        sig_ste = np.zeros((len(radii), n_b, n_rot), dtype=float)
        sig_lte = np.zeros((len(radii), n_b, n_rot), dtype=float)

        for r_i, radius in enumerate(radii):
            print(f"  radius {radius*1e6:.1f} um")
            substrate = _make_substrate(geometry, radius)

            for b in range(n_b):
                grad_ste = np.transpose(GWF_ste[:, :, b, :], (2, 0, 1))  # (rot, time, 3)
                grad_lte = np.transpose(GWF_lte[:, :, b, :], (2, 0, 1))

                sig_ste[r_i, b, :] = simulations.simulation(
                    n_walkers=int(n_walkers), diffusivity=D,
                    gradient=grad_ste, dt=dt, substrate=substrate,
                    seed=seed, quiet=quiet,
                )
                sig_lte[r_i, b, :] = simulations.simulation(
                    n_walkers=int(n_walkers), diffusivity=D,
                    gradient=grad_lte, dt=dt, substrate=substrate,
                    seed=seed, quiet=quiet,
                )

        save_dict = {
            "onam":     np.array(wf.onam, dtype=object),
            "geometry": np.array(geometry, dtype=object),
            "radii":    radii,
            "bvals":    bvals,
            "dt":       np.array(dt),
            "D":        np.array(D),
            "nWalkers": np.array(int(n_walkers)),
            "n_rot":    np.array(n_rot),
            "STE_sig":  sig_ste,
            "LTE_sig":  sig_lte,
        }
        if do_powder:
            save_dict["STE_powder"] = sig_ste.mean(axis=2)
            save_dict["LTE_powder"] = sig_lte.mean(axis=2)

        out_path = out_folder / f"{wf.onam}_sig.npz"
        np.savez(out_path, **save_dict)
        print(f"Saved {out_path}")
        out_files.append(out_path)

    return out_files


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _split_gwf(wf) -> tuple[np.ndarray, np.ndarray]:
    idx   = np.asarray(wf.wf_ind).ravel()
    G     = np.asarray(wf.GWF)
    n_t   = G.shape[0]
    n_b   = len(np.asarray(wf.bval).ravel())
    n_rot = int(wf.nrot)
    GWF_ste = G[:, :, idx == 1].reshape(n_t, 3, n_b, n_rot)
    GWF_lte = G[:, :, idx == 2].reshape(n_t, 3, n_b, n_rot)
    return GWF_ste, GWF_lte


def _make_substrate(geometry: str, radius: float):
    if geometry == "sphere":
        return substrates.sphere(radius=radius)
    elif geometry == "cylinder":
        return substrates.cylinder(
            radius=radius,
            orientation=np.array([0.0, 0.0, 1.0], dtype=float),
        )
    raise ValueError(f"Unknown geometry: {geometry!r}")
