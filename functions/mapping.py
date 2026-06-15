"""
Voxel-wise fitting functions and parallel execution for T2 parametric mapping.

This module provides:
- Per-voxel fitting workers (called by joblib in parallel)
- ``run_parallel`` : unified parallel execution wrapper with tqdm progress bar
- ``plot_histogram`` : background noise histogram utility

All ``_fit_voxel_*`` functions share the same signature expected by
``run_parallel`` and return tuples whose first two elements are always
the voxel coordinates ``(x, y)``.
"""

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from tqdm import tqdm

from .model import (
    fit_mono, fit_mono_offset, fit_bi, fit_bi_offset, _estimate_p0,
)
from .utils import compute_aic, compute_r2, compute_rmse, tqdm_joblib


# ── Histogram utility ─────────────────────────────────────────────────────────

def plot_histogram(data, ax=None, bins=100):
    """Plot the intensity histogram of background corner voxels.

    Extracts voxels from the 8 corners of the volume (assumed to contain
    only background noise) and displays their intensity distribution.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data. If 4-D, the maximum projection along the echo axis
        is used.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None``, a new figure is created and displayed.
    bins : int, optional
        Number of histogram bins. Default is 100.

    Examples
    --------
    >>> plot_histogram(data)                     # standalone figure
    >>> plot_histogram(data, ax=axes[0, 0])      # embed in existing layout
    """
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    nx, ny, nz = vol.shape
    cx, cy, cz = max(1, nx // 20), max(1, ny // 20), max(1, nz // 20)
    corners = np.concatenate([
        vol[:cx,  :cy,  :cz ].flatten(), vol[-cx:, :cy,  :cz ].flatten(),
        vol[:cx,  -cy:, :cz ].flatten(), vol[-cx:, -cy:, :cz ].flatten(),
        vol[:cx,  :cy,  -cz:].flatten(), vol[-cx:, :cy,  -cz:].flatten(),
        vol[:cx,  -cy:, -cz:].flatten(), vol[-cx:, -cy:, -cz:].flatten(),
    ])
    corners = corners[corners > 0]
    mu = np.mean(corners)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots()

    ax.hist(corners, bins=bins, density=True, color="steelblue")
    ax.axvline(mu, linestyle="--", color="cyan", label=f"µ={mu:.1f}")
    ax.set_title("Background noise histogram")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend()

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Parallel execution wrapper ────────────────────────────────────────────────

def run_parallel(func, voxels, data, te, z, n_jobs=-1, **kwargs):
    """Run a voxel-wise fitting function in parallel using joblib.

    Dispatches ``func(x, y, signal, te, **kwargs)`` for every ``(x, y)``
    pair in ``voxels``, using all available CPU cores by default.

    Parameters
    ----------
    func : callable
        A ``_fit_voxel_*`` function from this module. Must accept
        ``(x, y, signal, te, **kwargs)`` and return a tuple starting
        with ``(x, y, ...)``.
    voxels : list of tuple of int
        List of ``(x, y)`` coordinate pairs to process.
    data : np.ndarray of shape (nx, ny, nz, n_te)
        Full MRI data array.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    z : int
        Slice index along the z axis.
    n_jobs : int, optional
        Number of parallel workers. ``-1`` uses all available cores.
        Default is ``-1``.
    **kwargs
        Additional keyword arguments forwarded to ``func``
        (e.g. ``c_fixed=12.5`` for :func:`_fit_voxel_mono_cfix`).

    Returns
    -------
    results : list
        List of tuples returned by ``func``, one per voxel.
        Order is not guaranteed to match input order.

    Examples
    --------
    >>> voxels = [(x, y) for x in range(nx) for y in range(ny) if mask[x, y, z]]
    >>> results = run_parallel(_fit_voxel_mono, voxels, data, te, z=5)
    """
    with tqdm_joblib(tqdm(total=len(voxels), desc="Computing")):
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(func)(x, y, data[x, y, z, :], te, **kwargs)
            for x, y in voxels
        )


# ── Per-voxel workers ─────────────────────────────────────────────────────────

def _fit_voxel_mono(x, y, signal, te):
    """Fit mono-exponential models (with and without offset) on a single voxel.

    Parameters
    ----------
    x, y : int
        Voxel coordinates in the 2-D slice.
    signal : np.ndarray of shape (n_te,)
        Signal decay curve for this voxel.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.

    Returns
    -------
    x, y : int
        Input coordinates (passed through for result reassembly).
    t2_mono : float
        T2 from the mono-exponential fit (ms). ``np.nan`` on failure.
    t2_off : float
        T2 from the mono-exponential + offset fit (ms). ``np.nan`` on failure.
    """
    p_mono, _, _ = fit_mono(te, signal)
    p_off,  _, _ = fit_mono_offset(te, signal)
    t2_mono = p_mono["T2"] if p_mono is not None else np.nan
    t2_off  = p_off["T2"]  if p_off  is not None else np.nan
    return x, y, t2_mono, t2_off


def _fit_voxel_bi(x, y, signal, te):
    """Fit bi-exponential models (with and without offset) on a single voxel.

    Parameters
    ----------
    x, y : int
        Voxel coordinates in the 2-D slice.
    signal : np.ndarray of shape (n_te,)
        Signal decay curve for this voxel.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.

    Returns
    -------
    x, y : int
        Input coordinates.
    t2c_bi : float
        Short T2 component from bi-exponential fit (ms).
    t2l_bi : float
        Long T2 component from bi-exponential fit (ms).
    t2c_bioff : float
        Short T2 component from bi-exponential + offset fit (ms).
    t2l_bioff : float
        Long T2 component from bi-exponential + offset fit (ms).
    t2_eff_bi : float
        Effective T2 = f·T2c + (1-f)·T2l for bi model (ms).
    t2_eff_bioff : float
        Effective T2 for bi + offset model (ms).
    """
    p_bi,    _, _ = fit_bi(te, signal)
    p_bioff, _, _ = fit_bi_offset(te, signal)

    t2c_bi    = p_bi["T2c"]    if p_bi    is not None else np.nan
    t2l_bi    = p_bi["T2l"]    if p_bi    is not None else np.nan
    t2c_bioff = p_bioff["T2c"] if p_bioff is not None else np.nan
    t2l_bioff = p_bioff["T2l"] if p_bioff is not None else np.nan

    t2_eff_bi = (
        p_bi["f"] * p_bi["T2c"] + (1 - p_bi["f"]) * p_bi["T2l"]
        if p_bi is not None else np.nan
    )
    t2_eff_bioff = (
        p_bioff["f"] * p_bioff["T2c"] + (1 - p_bioff["f"]) * p_bioff["T2l"]
        if p_bioff is not None else np.nan
    )
    return x, y, t2c_bi, t2l_bi, t2c_bioff, t2l_bioff, t2_eff_bi, t2_eff_bioff


def _fit_voxel_utils(x, y, signal, te):
    """Run all four models and compute AIC-based model selection metrics.

    Fits all four models, selects the best by AIC, and computes goodness-of-fit
    metrics (R², RMSE) as well as per-model I0 values for global I0 mapping.

    Parameters
    ----------
    x, y : int
        Voxel coordinates.
    signal : np.ndarray of shape (n_te,)
        Signal decay curve.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.

    Returns
    -------
    x, y : int
        Input coordinates.
    best_idx : int
        Index of the best model (0=mono, 1=mono+offset, 2=bi, 3=bi+offset).
    best : str
        Name of the best model.
    i0_vw : float
        I0 from the best model (filtered for outliers > 1e7).
    r2 : float
        R² of the best fit.
    rmse : float
        RMSE of the best fit.
    i0_per_model : dict
        I0 values keyed by model name, for global I0 map computation.
    f_bi_val : float
        Water fraction f from bi-exponential fit.
    f_bioff_val : float
        Water fraction f from bi-exponential + offset fit.
    t2_eff : float
        Effective T2 from bi+offset model (ms).
    all_fitted : dict
        Fitted signals keyed by model name, for downstream R²/RMSE computation.
    """
    model_labels = ["mono", "mono+offset", "bi", "bi+offset"]

    p_mono,  f_mono,  _ = fit_mono(te, signal)
    p_off,   f_off,   _ = fit_mono_offset(te, signal)
    p_bi,    f_bi_fit, _ = fit_bi(te, signal)
    p_bioff, f_bioff, _ = fit_bi_offset(te, signal)

    aic_dict = {
        "mono":        compute_aic(signal, f_mono,   2),
        "mono+offset": compute_aic(signal, f_off,    3),
        "bi":          compute_aic(signal, f_bi_fit, 4),
        "bi+offset":   compute_aic(signal, f_bioff,  5),
    }
    best     = min(aic_dict, key=aic_dict.get)
    best_idx = model_labels.index(best)

    params_map = {
        "mono": p_mono, "mono+offset": p_off,
        "bi": p_bi,     "bi+offset":   p_bioff,
    }
    fitted_map = {
        "mono": f_mono, "mono+offset": f_off,
        "bi": f_bi_fit, "bi+offset":   f_bioff,
    }

    best_p = params_map[best]
    best_f = fitted_map[best]

    i0_raw = best_p["I0"] if best_p is not None else np.nan
    i0_vw  = i0_raw if (best_p is not None and 0 < i0_raw < 1e7) else np.nan

    r2   = compute_r2(signal, best_f)   if best_f is not None else np.nan
    rmse = compute_rmse(signal, best_f) if best_f is not None else np.nan

    i0_per_model = {
        k: (params_map[k]["I0"] if params_map[k] is not None else np.nan)
        for k in model_labels
    }

    f_bi_val    = p_bi["f"]    if p_bi    is not None else np.nan
    f_bioff_val = p_bioff["f"] if p_bioff is not None else np.nan

    t2_eff = (
        p_bioff["f"] * p_bioff["T2c"] + (1 - p_bioff["f"]) * p_bioff["T2l"]
        if p_bioff is not None else np.nan
    )

    return (
        x, y, best_idx, best, i0_vw, r2, rmse,
        i0_per_model, f_bi_val, f_bioff_val, t2_eff, fitted_map,
    )


def _fit_voxel_error(x, y, signal, te):
    """Compute AIC-based model selection and voxel-wise goodness-of-fit metrics.

    Similar to :func:`_fit_voxel_utils` but returns all fitted signals for
    a subsequent global-model R²/RMSE pass in :func:`display.display_slice`.

    Parameters
    ----------
    x, y : int
        Voxel coordinates.
    signal : np.ndarray of shape (n_te,)
        Signal decay curve.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.

    Returns
    -------
    x, y : int
        Input coordinates.
    best : str
        Name of the AIC-selected best model.
    r2_vw : float
        R² of the best fit.
    rmse_vw : float
        RMSE of the best fit.
    fitted_map : dict
        Fitted signals for all four models, keyed by model name.
    """
    p_mono,  f_mono,  _ = fit_mono(te, signal)
    p_off,   f_off,   _ = fit_mono_offset(te, signal)
    p_bi,    f_bi_fit, _ = fit_bi(te, signal)
    p_bioff, f_bioff, _ = fit_bi_offset(te, signal)

    aic_dict = {
        "mono":        compute_aic(signal, f_mono,   2),
        "mono+offset": compute_aic(signal, f_off,    3),
        "bi":          compute_aic(signal, f_bi_fit, 4),
        "bi+offset":   compute_aic(signal, f_bioff,  5),
    }
    best = min(aic_dict, key=aic_dict.get)
    fitted_map = {
        "mono": f_mono, "mono+offset": f_off,
        "bi": f_bi_fit, "bi+offset":   f_bioff,
    }
    best_f  = fitted_map[best]
    r2_vw   = compute_r2(signal, best_f)   if best_f is not None else np.nan
    rmse_vw = compute_rmse(signal, best_f) if best_f is not None else np.nan

    return x, y, best, r2_vw, rmse_vw, fitted_map


def _fit_voxel_noise(x, y, signal, te):
    """Extract the offset parameter C from models with an additive constant.

    Used to build spatial maps of the noise floor and to estimate a global
    median C for the fixed-offset mono-exponential fit.

    Parameters
    ----------
    x, y : int
        Voxel coordinates.
    signal : np.ndarray of shape (n_te,)
        Signal decay curve.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.

    Returns
    -------
    x, y : int
        Input coordinates.
    c_mono : float
        Offset C from mono-exponential + offset fit. ``np.nan`` on failure.
    c_bi : float
        Offset C from bi-exponential + offset fit. ``np.nan`` on failure.
    """
    p_off,   _, _ = fit_mono_offset(te, signal)
    p_bioff, _, _ = fit_bi_offset(te, signal)
    c_mono = p_off["C"]   if p_off   is not None else np.nan
    c_bi   = p_bioff["C"] if p_bioff is not None else np.nan
    return x, y, c_mono, c_bi


def _fit_voxel_mono_cfix(x, y, signal, te, c_fixed):
    """Fit a mono-exponential model with a fixed (non-optimised) offset C.

    Useful when a global noise floor estimate is available and should be
    applied uniformly across all voxels, reducing the degrees of freedom.

    Parameters
    ----------
    x, y : int
        Voxel coordinates.
    signal : np.ndarray of shape (n_te,)
        Signal decay curve.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    c_fixed : float
        Fixed offset value (typically the median C from
        :func:`_fit_voxel_noise`).

    Returns
    -------
    x, y : int
        Input coordinates.
    t2 : float
        Fitted T2 in milliseconds. ``np.nan`` on failure.
    """
    te  = np.asarray(te,     dtype=float)
    sig = np.asarray(signal, dtype=float)

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
        return x, y, popt[1]
    except Exception:
        return x, y, np.nan


def _fit_global(x, y, signal, te, global_best, k):
    """Fit a single voxel with the globally selected best model.

    Used in a second parallel pass (after model selection) to compute
    R²/RMSE maps using a single model applied uniformly across the slice.

    Parameters
    ----------
    x, y : int
        Voxel coordinates.
    signal : np.ndarray of shape (n_te,)
        Signal decay curve.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    global_best : {"mono", "mono+offset", "bi", "bi+offset"}
        Name of the model to apply.
    k : int
        Number of free parameters (used externally for AIC; not used here).

    Returns
    -------
    x, y : int
        Input coordinates.
    r2 : float
        R² of the fit with the global model.
    rmse : float
        RMSE of the fit with the global model.
    """
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