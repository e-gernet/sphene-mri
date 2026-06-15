"""
Utility functions for signal quality metrics, noise estimation, and masking.

This module provides:
- Goodness-of-fit metrics: AIC, R², RMSE, Pearson r
- Noise estimation from background corners
- Brain/tissue mask generation (histogram, Otsu, Rician)
- joblib/tqdm integration helper
"""

import contextlib
import joblib
import numpy as np
from scipy import ndimage


# ── joblib / tqdm ─────────────────────────────────────────────────────────────

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to report progress into a tqdm progress bar.

    Parameters
    ----------
    tqdm_object : tqdm.tqdm
        An already-instantiated tqdm bar (e.g. ``tqdm(total=n)``).

    Yields
    ------
    tqdm_object : tqdm.tqdm
        The same bar, patched in-place.

    Examples
    --------
    >>> with tqdm_joblib(tqdm(total=100, desc="Fitting")):
    ...     results = Parallel(n_jobs=-1)(delayed(f)(i) for i in range(100))
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# ── Goodness-of-fit metrics ───────────────────────────────────────────────────

def compute_aic(signal, fit, k):
    """Compute the Akaike Information Criterion (AIC) for a fitted signal.

    Uses the small-sample corrected form when ``n / k < 40``.
    A lower AIC indicates a better trade-off between fit quality and model
    complexity.

    Parameters
    ----------
    signal : array-like of shape (n_te,)
        Observed signal intensities.
    fit : array-like of shape (n_te,) or None
        Model-predicted signal. If ``None``, returns ``np.inf``.
    k : int
        Number of free parameters in the model
        (e.g. 2 for mono, 3 for mono+offset, 4 for bi, 5 for bi+offset).

    Returns
    -------
    aic : float
        AIC value. Returns ``np.inf`` if ``fit`` is ``None``.

    Examples
    --------
    >>> aic = compute_aic(signal, fitted_signal, k=2)
    """
    if fit is None:
        return np.inf
    n = len(signal)
    residuals = signal - fit
    rss = np.sum(residuals ** 2)
    if rss <= 0:
        rss = 1e-10
    return 2 * k + n * np.log(rss / n)


def compute_r2(signal, fitted):
    """Compute the coefficient of determination R².

    Parameters
    ----------
    signal : array-like of shape (n_te,)
        Observed signal intensities.
    fitted : array-like of shape (n_te,) or None
        Model-predicted signal. If ``None``, returns ``np.nan``.

    Returns
    -------
    r2 : float
        R² value in [0, 1]. Returns ``np.nan`` if ``fitted`` is ``None``
        or if the signal has zero variance.

    Examples
    --------
    >>> r2 = compute_r2(signal, fitted_signal)
    """
    if fitted is None:
        return np.nan
    ss_res = np.sum((signal - fitted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def compute_rmse(signal, fitted):
    """Compute the Root Mean Square Error (RMSE) between signal and fit.

    Parameters
    ----------
    signal : array-like of shape (n_te,)
        Observed signal intensities.
    fitted : array-like of shape (n_te,) or None
        Model-predicted signal. If ``None``, returns ``np.nan``.

    Returns
    -------
    rmse : float
        RMSE in the same units as the signal intensity.

    Examples
    --------
    >>> rmse = compute_rmse(signal, fitted_signal)
    """
    if fitted is None:
        return np.nan
    return np.sqrt(np.mean((signal - fitted) ** 2))


def compute_pearson_r(signal, fitted):
    """Compute the Pearson correlation coefficient between signal and fit.

    Parameters
    ----------
    signal : array-like of shape (n_te,)
        Observed signal intensities.
    fitted : array-like of shape (n_te,) or None
        Model-predicted signal. If ``None``, returns ``np.nan``.

    Returns
    -------
    r : float
        Pearson r in [-1, 1]. Returns ``np.nan`` if either array has zero
        standard deviation or if ``fitted`` is ``None``.

    Examples
    --------
    >>> r = compute_pearson_r(signal, fitted_signal)
    """
    if fitted is None:
        return np.nan
    signal = np.asarray(signal, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    if np.std(signal) == 0 or np.std(fitted) == 0:
        return np.nan
    return np.corrcoef(signal, fitted)[0, 1]


# ── Noise estimation ──────────────────────────────────────────────────────────

def _extract_corners(data, corner_fraction=0.05):
    """Extract voxel intensities from the 8 corners of a 3-D volume.

    Corners are used as background regions to estimate noise statistics,
    assuming the sample does not occupy the edges of the field of view.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data. If 4-D, the maximum projection along the echo axis
        is used.
    corner_fraction : float, optional
        Fraction of each axis used to define corner size. Default is 0.05
        (5 % of each dimension).

    Returns
    -------
    corners : np.ndarray of shape (n_corner_voxels,)
        Strictly positive intensity values extracted from the 8 corners.
    """
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    nx, ny, nz = vol.shape
    cx = max(1, int(nx * corner_fraction))
    cy = max(1, int(ny * corner_fraction))
    cz = max(1, int(nz * corner_fraction))
    corners = np.concatenate([
        vol[:cx,  :cy,  :cz ].flatten(), vol[-cx:, :cy,  :cz ].flatten(),
        vol[:cx,  -cy:, :cz ].flatten(), vol[-cx:, -cy:, :cz ].flatten(),
        vol[:cx,  :cy,  -cz:].flatten(), vol[-cx:, :cy,  -cz:].flatten(),
        vol[:cx,  -cy:, -cz:].flatten(), vol[-cx:, -cy:, -cz:].flatten(),
    ])
    return corners[corners > 0]


def estimate_noise(data, corner_fraction=0.05):
    """Estimate background noise mean and standard deviation from volume corners.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    corner_fraction : float, optional
        Fraction of each axis used to define corner regions. Default is 0.05.

    Returns
    -------
    mean : float
        Mean intensity in background corners.
    std : float
        Standard deviation of intensity in background corners.

    Examples
    --------
    >>> mean, std = estimate_noise(data)
    >>> print(f"SNR estimate: {signal_peak / std:.1f}")
    """
    corners = _extract_corners(data, corner_fraction)
    mean, std = np.mean(corners), np.std(corners)
    print(f"[Noise] Mean: {mean:.4f} | Std: {std:.4f}")
    return mean, std


# ── Masking ───────────────────────────────────────────────────────────────────

def _apply_morphology(mask):
    """Apply binary closing and hole-filling to a 3-D mask.

    Parameters
    ----------
    mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Input binary mask.

    Returns
    -------
    mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Morphologically cleaned mask.
    """
    struct = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=4)
    mask = ndimage.binary_fill_holes(mask)
    return mask


def mask_histogram(data, k=3.5, use_morpho=False):
    """Generate a binary mask by thresholding at µ + k·σ over the full volume.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    k : float, optional
        Number of standard deviations above the mean used as threshold.
        Default is 3.5.
    use_morpho : bool, optional
        If ``True``, apply binary closing and hole-filling after thresholding.
        Default is ``False``.

    Returns
    -------
    mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Binary mask where ``True`` indicates tissue.
    """
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    flat = vol.flatten()
    threshold = np.mean(flat) + k * np.std(flat)
    mask = vol > threshold
    if use_morpho:
        mask = _apply_morphology(mask)
    return mask


def mask_otsu(data, use_morpho=False):
    """Generate a binary mask using Otsu's automatic thresholding.

    References
    ----------
    Otsu, N. (1979). A threshold selection method from gray-level histograms.
    *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    use_morpho : bool, optional
        If ``True``, apply binary closing and hole-filling after thresholding.
        Default is ``False``.

    Returns
    -------
    mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Binary mask where ``True`` indicates tissue.
    """
    from skimage.filters import threshold_otsu
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    mask = vol > threshold_otsu(vol)
    if use_morpho:
        mask = _apply_morphology(mask)
    return mask


def mask_rician(data, corner_fraction=0.05, k=4.0, use_morpho=False):
    """Generate a binary mask using a Rician noise threshold.

    Estimates the Rician noise parameter σ from background corners, then
    thresholds at k·σ_rician. This approach is better suited to magnitude
    MRI data than Gaussian-based methods.

    References
    ----------
    Gudbjartsson, H., & Patz, S. (1995). The Rician distribution of noisy
    MRI data. *Magnetic Resonance in Medicine*, 34(6), 910–914.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    corner_fraction : float, optional
        Fraction of each axis used to define background corners. Default 0.05.
    k : float, optional
        Threshold multiplier applied to σ_rician. Default is 4.0.
    use_morpho : bool, optional
        If ``True``, apply binary closing and hole-filling after thresholding.
        Default is ``False``.

    Returns
    -------
    mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Binary mask where ``True`` indicates tissue.
    """
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    corners = _extract_corners(vol if vol.ndim == 3 else data, corner_fraction)
    sigma_rician = np.mean(corners) / np.sqrt(np.pi / 2)
    mask = vol > k * sigma_rician
    if use_morpho:
        mask = _apply_morphology(mask)
    return mask


def compute_mask(data, method="rician", use_morpho=False, **kwargs):
    """Compute a 3-D binary tissue mask using the specified method.

    This is the main entry point for mask generation. All masking strategies
    operate on the maximum-intensity projection along the echo axis.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    method : {"rician", "otsu", "histogram"}, optional
        Masking strategy. Default is ``"rician"``.

        - ``"rician"`` : Rician noise threshold (recommended for magnitude MRI)
        - ``"otsu"``   : Otsu automatic threshold
        - ``"histogram"`` : µ + k·σ global threshold
    use_morpho : bool, optional
        If ``True``, apply binary closing and hole-filling. Default is ``False``.
    **kwargs
        Additional keyword arguments forwarded to the selected masking function
        (e.g. ``k=4.0`` for Rician, ``k=3.5`` for histogram).

    Returns
    -------
    mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Binary mask where ``True`` indicates tissue.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported strategies.

    Examples
    --------
    >>> mask = compute_mask(data, method="rician", k=4.0, use_morpho=True)
    >>> mask = compute_mask(data, method="otsu")
    """
    methods = {
        "histogram": mask_histogram,
        "otsu":      mask_otsu,
        "rician":    mask_rician,
    }
    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {list(methods.keys())}."
        )
    return methods[method](data, use_morpho=use_morpho, **kwargs)