"""
Voxel-wise fitting for T2 parametric mapping, run in parallel.

One call per slice, :func:`fit_slice`, fits the two models of
:mod:`functions.model` (mono and bi) on every voxel and keeps everything the
viewer needs: the T2 maps, which model the data justify, and the quality maps.
Every button reads that single result, so nothing is fitted twice.

Also provided:

- :func:`run_parallel` : joblib wrapper with a progress bar
- :func:`slice_summary` : counts and medians of a fitted slice, for the console
- :func:`plot_histogram` : background noise histogram
"""

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .model import MODELS, fit_both, fit_quality, standardized_residuals
from .utils import compute_r2, compute_rmse, tqdm_joblib

# ── Histogram utility ─────────────────────────────────────────────────────────

def plot_histogram(data, ax=None, bins=100, z=None, t=None, bg_mask=None, te_ms=None, sigma=None):
    """Plot the intensity histogram of the background noise.

    Two modes:

    * **One slice, one echo** (``z`` and ``t`` given): histogram of the
      background voxels of slice ``z`` at echo ``t``. With ``sigma`` given, the
      Rayleigh law that pure noise of that level should follow is drawn on top:
      if the two agree, the noise level is trustworthy.
      ``bg_mask`` is the 3-D background mask from
      :func:`functions.utils.estimate_noise_auto` (computed here if omitted).
    * **Whole volume** (``z``/``t`` omitted): the background population of the
      maximum projection over the echoes (Otsu + sigma-clipping). Its mean is
      about 2.2x higher than for a single echo, because the maximum of 32 noise
      draws is larger than one draw.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If ``None``, a new figure is created and displayed.
    bins : int, optional
        Number of histogram bins. Default is 100.
    z, t : int or None, optional
        Slice and echo index for the one-slice/one-echo mode.
    bg_mask : np.ndarray of bool, shape (nx, ny, nz), optional
        Background mask (one-slice/one-echo mode).
    te_ms : float or None, optional
        Echo time shown in the title.
    sigma : float or None, optional
        Noise level (see :func:`functions.utils.estimate_sigma`). If given, in the
        one-slice/one-echo mode the expected Rayleigh curve and the expected mean
        of a pure-noise background, ``sigma * sqrt(pi/2)`` (about 1.25 sigma), are
        drawn.

    Examples
    --------
    >>> plot_histogram(data)                          # whole volume, max projection
    >>> plot_histogram(data, z=7, t=0)                # slice 8 (z counts from 0), first echo
    """
    from .utils import estimate_noise_auto

    if z is not None and t is not None and data.ndim == 4:
        if bg_mask is None:
            _, _, bg_mask, _ = estimate_noise_auto(np.max(data, axis=-1), verbose=False)
        values = data[:, :, z, t][bg_mask[:, :, z]]
        values = values[values > 0]
        mean = float(values.mean()) if values.size else float("nan")
        te_txt = f", TE={te_ms:.1f} ms" if te_ms is not None else ""
        title = f"Background noise, slice {z + 1}, echo {t + 1}{te_txt}"
    else:
        vol = np.max(data, axis=-1) if data.ndim == 4 else data
        mean, _std, _background_mask, values = estimate_noise_auto(vol, verbose=False)
        title = "Background noise histogram (max over echoes, Otsu + sigma-clip)"

    standalone = ax is None
    if standalone:
        _fig, ax = plt.subplots()

    ax.hist(values, bins=bins, density=True, color="steelblue")
    ax.axvline(mean, linestyle="--", color="cyan", label=f"µ={mean:.1f}")
    if sigma is not None and z is not None and t is not None:
        floor = float(sigma) * np.sqrt(np.pi / 2)
        grid = np.linspace(0, float(values.max()) if values.size else 5 * sigma, 300)
        ax.plot(grid, grid / sigma**2 * np.exp(-(grid**2) / (2 * sigma**2)),
                color="orange", linewidth=1.8,
                label=f"Rayleigh law for sigma={sigma:.0f}\n(expected mean {floor:.0f} = 1.25 x sigma)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    if standalone:
        plt.tight_layout()
        plt.show()


# ── Parallel execution wrapper ────────────────────────────────────────────────

def run_parallel(func, voxels, data, te, z, n_jobs=-1, **kwargs):
    """Run a voxel-wise function in parallel using joblib.

    Dispatches ``func(x, y, signal, te, **kwargs)`` for every ``(x, y)``
    pair in ``voxels``, using all available CPU cores by default.

    Parameters
    ----------
    func : callable
        Must accept ``(x, y, signal, te, **kwargs)`` and return a tuple
        starting with ``(x, y, ...)``. It has to live in an importable module
        (not in a script run as ``__main__``) so the workers can load it.
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
    **kwargs
        Additional keyword arguments forwarded to ``func``.

    Returns
    -------
    results : list
        List of tuples returned by ``func``, one per voxel.
    """
    with tqdm_joblib(tqdm(total=len(voxels), desc="Computing")):
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(func)(x, y, data[x, y, z, :], te, **kwargs)
            for x, y in voxels
        )


# ── Per-voxel fit ─────────────────────────────────────────────────────────────

# Values kept for every voxel. "best" is 1 where the data justify bi, 0 for mono.
_RAW_KEYS = (
    "best", "delta_aicc",
    "I0_mono", "T2_mono", "chi2_mono", "r2_mono", "rmse_mono", "limit_mono",
    "I0_bi", "f_bi", "T2c_bi", "T2l_bi", "chi2_bi", "r2_bi", "rmse_bi", "limit_bi",
)


def _fit_voxel(x, y, signal, te, sigma):
    """Fit mono and bi on one voxel and decide which one to keep.

    Returns
    -------
    x, y : int
        Input coordinates.
    rec : dict
        One float per name in ``_RAW_KEYS`` (NaN if a fit failed) and ``resid``:
        the standardized residuals of the kept model (see
        :func:`functions.model.fit_quality`), shape ``(n_te,)``.
    """
    out = fit_both(te, signal, sigma)
    rec = dict.fromkeys(_RAW_KEYS, np.nan)
    rec["best"] = float(out["best"] == "bi")
    rec["delta_aicc"] = out["delta_aicc"]
    rec["resid"] = np.full(len(te), np.nan)

    for name in MODELS:
        res = out[name]
        if res is None:
            continue
        for key, value in res["params"].items():
            rec[f"{key}_{name}"] = value
        rec[f"chi2_{name}"] = fit_quality(res, signal, sigma)
        rec[f"r2_{name}"] = compute_r2(signal, res["expected"])
        rec[f"rmse_{name}"] = compute_rmse(signal, res["expected"])
        rec[f"limit_{name}"] = float(res["at_bound"])
        if name == out["best"]:
            rec["resid"] = standardized_residuals(res, signal, sigma)
    return x, y, rec


def finish_maps(maps):
    """Add the maps derived from the per-voxel results (in place, also returned).

    - ``*_best`` : the value of the model kept in each voxel (``I0``, ``chi2``,
      ``r2``, ``rmse`` and ``limit``, the latter being 1 when a T2 of the kept
      model ended on a limit of its search range, so its value is not reliable);
    - ``T2eff_bi`` : amplitude-weighted mean T2 of the bi model, ``f*T2c + (1-f)*T2l``;
    - the bi maps (``T2c_bi``, ``T2l_bi``, ``f_bi``, ``T2eff_bi``) are emptied
      (NaN) where mono is kept: there the data do not justify two components.
    """
    is_bi = maps["best"] == 1
    for key in ("I0", "chi2", "r2", "rmse", "limit"):
        maps[f"{key}_best"] = np.where(is_bi, maps[f"{key}_bi"], maps[f"{key}_mono"])
    maps["T2eff_bi"] = maps["f_bi"] * maps["T2c_bi"] + (1 - maps["f_bi"]) * maps["T2l_bi"]
    for key in ("T2c_bi", "T2l_bi", "f_bi", "T2eff_bi"):
        maps[key] = np.where(is_bi, maps[key], np.nan)
    return maps


def fit_slice(data, te, z, voxels, sigma, n_jobs=-1):
    """Fit mono and bi on every voxel of a slice.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz, n_te)
        Raw MRI data.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    z : int
        Slice index.
    voxels : list of tuple of int
        ``(x, y)`` of the voxels to fit (the tissue mask of the slice).
    sigma : float
        Noise level (:func:`functions.utils.estimate_sigma`).
    n_jobs : int, optional
        Number of parallel workers (``-1`` = all cores).

    Returns
    -------
    maps : dict of np.ndarray of shape (nx, ny)
        NaN outside ``voxels``. Kept model: ``best`` (1 = bi, 0 = mono),
        ``delta_aicc`` (evidence for bi, see :func:`functions.model.fit_both`).
        Per model (``mono``, ``bi``): ``I0``, ``chi2`` (reduced chi-square),
        ``r2``, ``rmse``, ``limit``; ``T2_mono``; ``T2c_bi``, ``T2l_bi``,
        ``f_bi``, ``T2eff_bi`` (kept voxels only). And the ``*_best`` maps of
        :func:`finish_maps`.
    resid : np.ndarray of shape (n_voxels, n_te)
        Standardized residuals of the kept model, voxel by voxel.
    """
    nx, ny = data.shape[:2]
    results = run_parallel(_fit_voxel, voxels, data, te, z, n_jobs=n_jobs, sigma=sigma)
    maps = {key: np.full((nx, ny), np.nan) for key in _RAW_KEYS}
    resid = np.full((len(results), len(te)), np.nan)
    for i, (x, y, rec) in enumerate(results):
        for key in _RAW_KEYS:
            maps[key][x, y] = rec[key]
        resid[i] = rec["resid"]
    return finish_maps(maps), resid


def slice_summary(maps):
    """Counts and medians of a fitted slice, as a dict.

    Keys: ``n`` (voxels fitted), ``n_bi``, ``n_mono``, ``share_bi``, ``n_limit``
    (voxels whose kept model has a T2 on a search limit), ``chi2_median``
    (median reduced chi-square of the kept model).
    """
    fitted = np.isfinite(maps["best"])
    n = int(fitted.sum())
    n_bi = int(np.sum(maps["best"][fitted] == 1))
    return {
        "n": n, "n_bi": n_bi, "n_mono": n - n_bi,
        "share_bi": n_bi / n if n else float("nan"),
        "n_limit": int(np.nansum(maps["limit_best"])),
        "chi2_median": float(np.nanmedian(maps["chi2_best"])) if n else float("nan"),
    }


# Order of the statistics everywhere they are shown (console of the viewer,
# quickstats.py, CSV): n, mean, SD, median, min, max.
STAT_KEYS = ("mean", "std", "median", "min", "max")
_STAT_TITLES = {"mean": "mean", "std": "SD", "median": "median", "min": "min", "max": "max"}


def map_stats(arr):
    """Descriptive statistics of the finite values of a map, as a dict.

    Keys: ``n``, ``mean``, ``std``, ``median``, ``min``, ``max``. ``std`` is the
    sample standard deviation (divided by ``n - 1``), the usual one in papers and
    the default of R, MATLAB, Excel and pandas (NumPy's own default divides by
    ``n``). Everything but ``n`` is NaN for an empty map, and ``std`` for a single
    value.
    """
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(vals.size), "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
        "median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max()),
    }


def stats_header(title, label_width=14):
    """Header line of a statistics table (same layout as :func:`stats_line`)."""
    return f"  {title:<{label_width}}{'n':>7}" + "".join(f"{_STAT_TITLES[k]:>9}" for k in STAT_KEYS)


def stats_line(label, st, digits=1, label_width=14):
    """One line of a statistics table from a :func:`map_stats` dict."""
    cells = "".join(f"{'-':>9}" if not np.isfinite(st[k]) else f"{st[k]:>9.{digits}f}" for k in STAT_KEYS)
    return f"  {label:<{label_width}}{st['n']:>7d}{cells}"
