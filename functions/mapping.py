import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import curve_fit
from .model import fit_mono, fit_mono_offset, fit_bi, fit_bi_offset
from .utils import compute_aic, compute_r2, compute_rmse
from .model import _estimate_p0
from .io import tqdm_joblib

# ── Utilitaires ──────────────────────────────────────────────────────────────


def _extract_corners(data):
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    nx, ny, nz = vol.shape
    cx, cy, cz = max(1, nx//20), max(1, ny//20), max(1, nz//20)
    corners = np.concatenate([
        vol[:cx,  :cy,  :cz ].flatten(), vol[-cx:, :cy,  :cz ].flatten(),
        vol[:cx,  -cy:, :cz ].flatten(), vol[-cx:, -cy:, :cz ].flatten(),
        vol[:cx,  :cy,  -cz:].flatten(), vol[-cx:, :cy,  -cz:].flatten(),
        vol[:cx,  -cy:, -cz:].flatten(), vol[-cx:, -cy:, -cz:].flatten(),
    ])
    return corners[corners > 0]


def plot_histogram(data, ax=None, bins=100):
    corners = _extract_corners(data)
    mu = np.mean(corners)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()
    ax.hist(corners, bins=bins, density=True, color="steelblue")
    ax.axvline(mu, linestyle='--', color="cyan", label=f"µ={mu:.1f}")
    ax.set_title("Histogram of background noise")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend()
    if standalone:
        plt.tight_layout()
        plt.show()


# ── Fonctions de calcul par voxel (joblib) ───────────────────────────────────

def _fit_voxel_mono(x, y, signal, te):
    p_mono, _, _ = fit_mono(te, signal)
    p_off,  _, _ = fit_mono_offset(te, signal)
    t2_mono = p_mono["T2"] if p_mono is not None else np.nan
    t2_off  = p_off["T2"]  if p_off  is not None else np.nan
    return x, y, t2_mono, t2_off


def _fit_voxel_bi(x, y, signal, te):
    p_bi,    _, _ = fit_bi(te, signal)
    p_bioff, _, _ = fit_bi_offset(te, signal)

    t2c_bi    = p_bi["T2c"]    if p_bi    is not None else np.nan
    t2l_bi    = p_bi["T2l"]    if p_bi    is not None else np.nan
    t2c_bioff = p_bioff["T2c"] if p_bioff is not None else np.nan
    t2l_bioff = p_bioff["T2l"] if p_bioff is not None else np.nan

    # T2 effectif
    if p_bi is not None:
        t2_eff_bi = p_bi["f"] * p_bi["T2c"] + (1 - p_bi["f"]) * p_bi["T2l"]
    else:
        t2_eff_bi = np.nan

    if p_bioff is not None:
        t2_eff_bioff = p_bioff["f"] * p_bioff["T2c"] + (1 - p_bioff["f"]) * p_bioff["T2l"]
    else:
        t2_eff_bioff = np.nan

    return x, y, t2c_bi, t2l_bi, t2c_bioff, t2l_bioff, t2_eff_bi, t2_eff_bioff


def _fit_voxel_utils(x, y, signal, te):
    model_labels = ["mono", "mono+offset", "bi", "bi+offset"]

    p_mono,  f_mono,  _ = fit_mono(te, signal)
    p_off,   f_off,   _ = fit_mono_offset(te, signal)
    p_bi,    f_bi,    _ = fit_bi(te, signal)
    p_bioff, f_bioff, _ = fit_bi_offset(te, signal)

    aic_dict = {
        "mono":        compute_aic(signal, f_mono,  2),
        "mono+offset": compute_aic(signal, f_off,   3),
        "bi":          compute_aic(signal, f_bi,    4),
        "bi+offset":   compute_aic(signal, f_bioff, 5),
    }
    best     = min(aic_dict, key=aic_dict.get)
    best_idx = model_labels.index(best)

    params_map = {"mono": p_mono, "mono+offset": p_off,
                  "bi": p_bi,    "bi+offset": p_bioff}
    fitted_map = {"mono": f_mono, "mono+offset": f_off,
                  "bi": f_bi,    "bi+offset": f_bioff}

    best_p = params_map[best]
    best_f = fitted_map[best]

    # I0 voxel-wise avec filtre aberrant
    if best_p is not None:
        i0_raw = best_p["I0"]
        i0_vw  = i0_raw if 0 < i0_raw < 1e7 else np.nan
    else:
        i0_vw = np.nan

    r2   = compute_r2(signal, best_f)   if best_f is not None else np.nan
    rmse = compute_rmse(signal, best_f) if best_f is not None else np.nan

    # I0 par modèle pour i0_global
    i0_per_model = {
        k: (params_map[k]["I0"] if params_map[k] is not None else np.nan)
        for k in model_labels
    }

    # Fraction f bi et bi+offset
    f_bi_val    = p_bi["f"]    if p_bi    is not None else np.nan
    f_bioff_val = p_bioff["f"] if p_bioff is not None else np.nan

    # T2 effectif = f·T2c + (1-f)·T2l pour bi+offset (plus stable)
    if p_bioff is not None:
        t2_eff = p_bioff["f"] * p_bioff["T2c"] + (1 - p_bioff["f"]) * p_bioff["T2l"]
    else:
        t2_eff = np.nan

    # Fits retournés pour calcul R²/RMSE global dans display
    all_fitted = {"mono": f_mono, "mono+offset": f_off,
                  "bi": f_bi,    "bi+offset": f_bioff}

    return (x, y, best_idx, best, i0_vw, r2, rmse,
            i0_per_model, f_bi_val, f_bioff_val, t2_eff, all_fitted)



def _fit_voxel_error(x, y, signal, te):
    """Calcule AIC, R² et RMSE voxel-wise en un seul pass."""
    model_labels = ["mono", "mono+offset", "bi", "bi+offset"]

    p_mono,  f_mono,  _ = fit_mono(te, signal)
    p_off,   f_off,   _ = fit_mono_offset(te, signal)
    p_bi,    f_bi,    _ = fit_bi(te, signal)
    p_bioff, f_bioff, _ = fit_bi_offset(te, signal)

    aic_dict = {
        "mono":        compute_aic(signal, f_mono,  2),
        "mono+offset": compute_aic(signal, f_off,   3),
        "bi":          compute_aic(signal, f_bi,    4),
        "bi+offset":   compute_aic(signal, f_bioff, 5),
    }
    best   = min(aic_dict, key=aic_dict.get)
    fitted_map = {"mono": f_mono, "mono+offset": f_off,
                  "bi": f_bi,    "bi+offset": f_bioff}
    best_f = fitted_map[best]

    r2_vw   = compute_r2(signal, best_f)   if best_f is not None else np.nan
    rmse_vw = compute_rmse(signal, best_f) if best_f is not None else np.nan

    return x, y, best, r2_vw, rmse_vw, fitted_map


def _fit_voxel_noise(x, y, signal, te):
    p_off,   _, _ = fit_mono_offset(te, signal)
    p_bioff, _, _ = fit_bi_offset(te, signal)
    c_mono = p_off["C"]   if p_off   is not None else np.nan
    c_bi   = p_bioff["C"] if p_bioff is not None else np.nan
    return x, y, c_mono, c_bi


def run_parallel(func, voxels, data, te, z, n_jobs=-1, **kwargs):
    with tqdm_joblib(tqdm(total=len(voxels), desc="Calcul")):
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(func)(x, y, data[x, y, z, :], te, **kwargs)
            for x, y in voxels
        )
    


# ── Fonctions volume entier (étape 4) ────────────────────────────────────────

def compute_mono_maps(data, te, mask, z):
    nx, ny = data.shape[0], data.shape[1]
    t2_mono = np.full((nx, ny), np.nan)
    t2_off  = np.full((nx, ny), np.nan)
    for x in range(nx):
        for y in range(ny):
            if not mask[x, y, z]:
                continue
            signal = data[x, y, z, :]
            p_mono, _, _ = fit_mono(te, signal)
            p_off,  _, _ = fit_mono_offset(te, signal)
            if p_mono is not None: t2_mono[x, y] = p_mono["T2"]
            if p_off  is not None: t2_off[x, y]  = p_off["T2"]
    return t2_mono, t2_off


def compute_bi_maps(data, te, mask, z):
    nx, ny = data.shape[0], data.shape[1]
    t2c_bi    = np.full((nx, ny), np.nan)
    t2l_bi    = np.full((nx, ny), np.nan)
    t2c_bioff = np.full((nx, ny), np.nan)
    t2l_bioff = np.full((nx, ny), np.nan)
    for x in range(nx):
        for y in range(ny):
            if not mask[x, y, z]:
                continue
            signal = data[x, y, z, :]
            p_bi,    _, _ = fit_bi(te, signal)
            p_bioff, _, _ = fit_bi_offset(te, signal)
            if p_bi is not None:
                t2c_bi[x, y] = p_bi["T2c"]
                t2l_bi[x, y] = p_bi["T2l"]
            if p_bioff is not None:
                t2c_bioff[x, y] = p_bioff["T2c"]
                t2l_bioff[x, y] = p_bioff["T2l"]
    return t2c_bi, t2l_bi, t2c_bioff, t2l_bioff


def compute_utils_maps(data, te, mask, z):
    nx, ny = data.shape[0], data.shape[1]
    aic_map      = np.full((nx, ny), np.nan)
    i0_voxelwise = np.full((nx, ny), np.nan)
    r2_map       = np.full((nx, ny), np.nan)
    rmse_map     = np.full((nx, ny), np.nan)
    model_labels = ["mono", "mono+offset", "bi", "bi+offset"]
    model_counts = {k: 0 for k in model_labels}
    all_params   = {}
    for x in range(nx):
        for y in range(ny):
            if not mask[x, y, z]:
                continue
            signal = data[x, y, z, :]
            p_mono,  f_mono,  _ = fit_mono(te, signal)
            p_off,   f_off,   _ = fit_mono_offset(te, signal)
            p_bi,    f_bi,    _ = fit_bi(te, signal)
            p_bioff, f_bioff, _ = fit_bi_offset(te, signal)
            aic_dict = {
                "mono":        compute_aic(signal, f_mono,  2),
                "mono+offset": compute_aic(signal, f_off,   3),
                "bi":          compute_aic(signal, f_bi,    4),
                "bi+offset":   compute_aic(signal, f_bioff, 5),
            }
            best = min(aic_dict, key=aic_dict.get)
            model_counts[best] += 1
            aic_map[x, y] = model_labels.index(best)
            params_map = {"mono": p_mono, "mono+offset": p_off,
                          "bi": p_bi, "bi+offset": p_bioff}
            fitted_map = {"mono": f_mono, "mono+offset": f_off,
                          "bi": f_bi, "bi+offset": f_bioff}
            best_p = params_map[best]
            best_f = fitted_map[best]
            if best_p is not None:
                i0_voxelwise[x, y] = best_p["I0"]
                r2_map[x, y]       = compute_r2(signal, best_f)
                rmse_map[x, y]     = compute_rmse(signal, best_f)
            all_params[(x, y)] = params_map
    global_best = max(model_counts, key=model_counts.get)
    i0_global   = np.full((nx, ny), np.nan)
    for (x, y), pmap in all_params.items():
        p = pmap[global_best]
        if p is not None:
            i0_global[x, y] = p["I0"]
    return aic_map, i0_voxelwise, i0_global, r2_map, rmse_map, global_best, model_counts


def compute_noise_maps(data, te, mask, z):
    nx, ny = data.shape[0], data.shape[1]
    c_mono_off = np.full((nx, ny), np.nan)
    c_bi_off   = np.full((nx, ny), np.nan)
    for x in range(nx):
        for y in range(ny):
            if not mask[x, y, z]:
                continue
            signal = data[x, y, z, :]
            p_off,   _, _ = fit_mono_offset(te, signal)
            p_bioff, _, _ = fit_bi_offset(te, signal)
            if p_off   is not None: c_mono_off[x, y] = p_off["C"]
            if p_bioff is not None: c_bi_off[x, y]   = p_bioff["C"]
    return c_mono_off, c_bi_off

def _fit_voxel_mono_cfix(x, y, signal, te, c_fixed):
    """Fit mono+offset avec C fixe — C n'est pas optimisé."""
    te  = np.asarray(te,  dtype=float)
    sig = np.asarray(signal, dtype=float)

    # Modèle avec C fixe — seuls I0 et T2 sont optimisés
    def _mono_c_fixed(t, I0, T2):
        return I0 * np.exp(-t / T2) + c_fixed

    I0_est, T2_est = _estimate_p0(te, sig)
    p0     = [I0_est, T2_est]
    bounds = ([0, 0.1], [np.inf, 5000.0])

    try:
        popt, _ = curve_fit(
            _mono_c_fixed, te, sig,
            p0=p0, bounds=bounds,
            method="trf", maxfev=10000,
        )
        return x, y, popt[1]  # T2
    except Exception:
        return x, y, np.nan
    
def _fit_global(x, y, signal, te, global_best, k):
    """Fit avec le modèle global pour calcul R²/RMSE global."""
    func_map = {
        "mono":        fit_mono,
        "mono+offset": fit_mono_offset,
        "bi":          fit_bi,
        "bi+offset":   fit_bi_offset,
    }
    _, fitted, _ = func_map[global_best](te, signal)
    r2   = compute_r2(signal, fitted)   if fitted is not None else np.nan
    rmse = compute_rmse(signal, fitted) if fitted is not None else np.nan
    return x, y, r2, rmse
