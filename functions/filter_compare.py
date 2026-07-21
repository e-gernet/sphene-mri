"""
Comparison of pre-processing filter strategies on fit quality.

This module evaluates whether filtering the raw signal before curve
fitting (spatial or temporal smoothing) actually improves the goodness
of fit, by comparing mean R² and RMSE across all tissue voxels of a
given slice for each filtering strategy.
"""

import numpy as np

from .utils import filter_data, compute_mask
from .mapping import run_parallel, _fit_voxel_error


def compare_filters(data, te, z, methods=("none", "gaussian_spatial", "savgol_temporal"), device="cpu"):
    """Compare filtering strategies on a single slice.

    For each method, filters the volume, recomputes the tissue mask,
    fits the AIC-selected best model on every masked voxel, and reports
    the mean R² and RMSE.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz, n_te)
        Raw MRI data.
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    z : int
        Slice index to evaluate.
    methods : tuple of str, optional
        Filtering strategies to compare. Default compares all three
        methods supported by :func:`functions.utils.filter_data`.
    device : {"cpu", "gpu"}, optional
        Compute device forwarded to :func:`functions.utils.filter_data`
        for the ``gaussian_spatial`` method. Default ``"cpu"``.

    Returns
    -------
    results : dict
        Dictionary keyed by method name, each value a dict with
        ``n_voxels``, ``R2_mean``, and ``RMSE_mean``.

    Examples
    --------
    >>> results = compare_filters(data, te_values, z=data.shape[2] // 2)
    >>> for method, stats in results.items():
    ...     print(method, stats)
    """
    results = {}
    print(f"\n[Filter comparison] slice z={z}")
    print("─" * 58)

    # Fixed mask computed once on the raw (unfiltered) data, so every
    # method is compared on the exact same set of voxels — otherwise
    # spatial smoothing artificially grows the mask and the comparison
    # becomes unfair.
    ref_mask = compute_mask(data, method="rician")
    voxels = [
        (x, y)
        for x in range(ref_mask.shape[0])
        for y in range(ref_mask.shape[1])
        if ref_mask[x, y, z]
    ]
    if not voxels:
        print("[Filter comparison] no voxel in reference mask — aborted")
        return results

    for method in methods:
        data_f = filter_data(data, method=method, sigma=1.0, window=5, poly=2, device=device)

        out = run_parallel(_fit_voxel_error, voxels, data_f, te, z)
        r2_vals = [row[3] for row in out]
        rmse_vals = [row[4] for row in out]

        r2_mean = float(np.nanmean(r2_vals))
        rmse_mean = float(np.nanmean(rmse_vals))
        results[method] = {
            "n_voxels": len(voxels),
            "R2_mean": r2_mean,
            "RMSE_mean": rmse_mean,
        }
        print(
            f"[{method:>18}] n={len(voxels):5d} | "
            f"R²={r2_mean:.3f} | RMSE={rmse_mean:.2f}"
        )

    print("─" * 58)
    return results
