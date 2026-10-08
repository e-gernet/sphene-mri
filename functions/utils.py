"""
Utility functions for signal quality metrics, noise estimation, and masking.

This module provides:
- Goodness-of-fit metrics: AIC, R², RMSE
- Automatic background noise estimation (Otsu + sigma-clipping)
- Tissue mask generation (histogram, Otsu, Rician)
- joblib/tqdm integration helper
"""

import contextlib

import joblib
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

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


# ── Noise estimation ──────────────────────────────────────────────────────────

def estimate_noise_auto(data, n_iter=10, sigma_clip=3.0, verbose=True):
    """Estimate background noise mean/std from an automatic background segmentation.

    Segments background vs. sample automatically with Otsu's threshold on
    the maximum-intensity projection, then refines the background
    statistics with iterative sigma-clipping - a standard background
    estimation technique (widely used in astronomical image reduction)
    that requires no geometry-specific tuning.

    Why sigma-clipping rather than a fixed dilation margin: a raw Otsu
    split still contains voxels just below threshold, at the sample's
    edge, that are brighter than true background (Gibbs ringing,
    partial-volume, susceptibility halo). Excluding them by dilating the
    foreground by a fixed number of voxels works, but that number is a
    magic constant tied to the halo width of *this* sample at *this*
    resolution - it has no reason to transfer to a different sample
    shape, size, or acquisition. Sigma-clipping instead treats the halo
    as what it statistically is: an outlier population relative to the
    bulk of the background. It is rejected automatically regardless of
    its width or the sample's geometry, with no free parameter to
    re-tune per dataset.

    Chosen over a Gaussian-mixture approach for determinism: Otsu has no
    random initialisation, so the initial split is exactly reproducible.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    n_iter : int, optional
        Maximum number of sigma-clipping iterations. The loop stops early
        as soon as an iteration removes no further voxels (convergence).
        Default is 10.
    sigma_clip : float, optional
        Voxels beyond ``sigma_clip`` standard deviations from the current
        background mean are rejected at each iteration. Default is 3.0.
    verbose : bool, optional
        Print one summary line. Default is True. Callers that only need the
        background mask pass False so the line is not repeated.

    Returns
    -------
    mean : float
        Mean intensity over the clipped background voxels. On 4-D data this is
        the mean of the MAXIMUM over the echoes (about 2.2x the background of
        a single echo), not the noise level: use :func:`estimate_sigma` for that.
    std : float
        Standard deviation over the clipped background voxels (same remark).
    background_mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Voxels classified as background *before* clipping (the Otsu
        split), returned so it can be reused (e.g. plotted) without
        re-running Otsu. Note this mask does not reflect the clipping
        step, which only affects the returned statistics.
    clipped_values : np.ndarray, 1-D
        The exact 1-D array of intensities that produced ``mean``/``std``
        (background, Otsu-split, post sigma-clip). Kept so a histogram of
        "the noise we actually measured" can be drawn.

    Examples
    --------
    >>> mean, std, bg, clipped = estimate_noise_auto(data)
    >>> print(f"[Noise] auto mean={mean:.1f} std={std:.1f}")
    """
    from skimage.filters import threshold_otsu

    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    threshold = threshold_otsu(vol)
    background_mask = vol <= threshold

    bg_values = vol[background_mask]
    bg_values = bg_values[bg_values > 0]  # drop true zero-padding voxels

    clipped = bg_values
    for i in range(n_iter):
        mean, std = np.mean(clipped), np.std(clipped)
        kept = clipped[np.abs(clipped - mean) < sigma_clip * std]
        if kept.size == clipped.size:
            break
        clipped = kept

    mean, std = float(np.mean(clipped)), float(np.std(clipped))
    if verbose:
        what = "max over echoes" if data.ndim == 4 else "image"
        print(
            f"[Noise] Background of the {what} (Otsu + {i + 1}-pass sigma-clip): "
            f"mean of the max = {mean:.1f} | std = {std:.1f} | "
            f"n={clipped.size}/{bg_values.size} background voxels kept"
        )
    return mean, std, background_mask, clipped


def estimate_sigma(data, bg_mask=None):
    """Estimate the Rician noise level σ of the acquisition, per echo then combined.

    ``estimate_noise_auto`` works on the *maximum projection over echoes*,
    which is the right input for segmenting tissue from background but the
    wrong one for measuring noise: the maximum of ``n_te`` independent noise
    samples is systematically larger than a single sample, so the mean/std
    it returns overestimate σ by a factor ≈ 2 on 32-echo data. Here the
    background (Otsu split, reused from ``estimate_noise_auto``) is read
    echo by echo instead.

    For pure Rician noise the median of the magnitude is
    ``σ·sqrt(2·ln 2)``, so ``σ = median / 1.1774``. The median is used
    rather than the mean or second moment because it is insensitive to the
    few tissue-halo / ghosting voxels that survive the Otsu split on the
    first (brightest) echoes. The per-echo estimates should all agree
    (noise does not depend on TE); their median is returned.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz, n_te)
        Raw 4-D MRI data (magnitude).
    bg_mask : np.ndarray of bool, shape (nx, ny, nz), optional
        Background mask from :func:`estimate_noise_auto`. Computed (silently)
        if omitted.

    Returns
    -------
    sigma : float
        Noise standard deviation of the underlying Gaussian channels, in
        the same units as ``data``.

    Examples
    --------
    >>> sigma = estimate_sigma(data)
    >>> snr_first_echo = np.median(data[mask, 0]) / sigma
    """
    if bg_mask is None:
        _, _, bg_mask, _ = estimate_noise_auto(np.max(data, axis=-1), verbose=False)
    per_echo = []
    for t in range(data.shape[-1]):
        bg = data[..., t][bg_mask]
        bg = bg[bg > 0]
        per_echo.append(np.median(bg) / np.sqrt(2 * np.log(2)))
    return float(np.median(per_echo))


# ── Pre-processing filters ────────────────────────────────────────────────────

def _check_gpu_available():
    """Check whether a CUDA GPU and ``cupy`` are usable.

    Returns
    -------
    available : bool
        ``True`` if ``cupy`` is installed and at least one CUDA device is
        detected. ``False`` otherwise (including if ``cupy`` is simply not
        installed — this is not an error, just "no GPU path available").
    """
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability  # raises if no usable device
        return True
    except Exception:
        return False


def filter_data(data, method="none", sigma=1.0, window=5, poly=2, device="cpu"):
    """Filter the raw 4-D MRI volume before masking and fitting.

    Two independent strategies are available:

    - ``"gaussian_spatial"`` : Gaussian smoothing applied slice-by-slice,
      independently for each echo. Reduces spatial noise but blurs tissue
      boundaries. Supports both CPU (scipy) and GPU (cupy) execution.
    - ``"savgol_temporal"`` : Savitzky-Golay smoothing applied along the
      echo-time axis, voxel by voxel. Smooths the decay curve without
      touching spatial resolution. CPU only — no GPU implementation yet.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz, n_te)
        Raw MRI data.
    method : {"none", "gaussian_spatial", "savgol_temporal"}, optional
        Filtering strategy. Default is ``"none"`` (no filtering, returns
        ``data`` unchanged).
    sigma : float, optional
        Standard deviation for the Gaussian kernel (spatial method only).
        Default is 1.0.
    window : int, optional
        Window length for the Savitzky-Golay filter (temporal method only).
        Must be odd and greater than ``poly``. Default is 5.
    poly : int, optional
        Polynomial order for the Savitzky-Golay filter (temporal method
        only). Default is 2.
    device : {"cpu", "gpu"}, optional
        Compute device for ``"gaussian_spatial"``. If ``"gpu"`` is
        requested but no CUDA device / ``cupy`` install is found, silently
        falls back to CPU with an explanatory message (never raises).
        Ignored for ``"savgol_temporal"`` and ``"none"``. Default ``"cpu"``.

    Returns
    -------
    filtered : np.ndarray of shape (nx, ny, nz, n_te)
        Filtered data, same shape and dtype as the input, always a plain
        numpy array regardless of which device performed the computation.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported strategies.

    Examples
    --------
    >>> data_f = filter_data(data, method="gaussian_spatial", sigma=1.0, device="gpu")
    >>> data_f = filter_data(data, method="savgol_temporal", window=5, poly=2)
    """
    if method == "none":
        return data

    if method == "gaussian_spatial":
        use_gpu = device == "gpu"
        if use_gpu and not _check_gpu_available():
            print(
                "[Filter] --device gpu requested but no usable CUDA device "
                "(or cupy is not installed) — falling back to CPU."
            )
            use_gpu = False

        n_echos = data.shape[3]

        if use_gpu:
            import cupy as cp
            from cupyx.scipy import ndimage as cndimage
            print("[Filter] Running gaussian_spatial on GPU (cupy).")
            data_gpu = cp.asarray(data)
            filtered_gpu = cp.empty_like(data_gpu)
            for t in range(n_echos):
                filtered_gpu[..., t] = cndimage.gaussian_filter(
                    data_gpu[..., t], sigma=sigma
                )
            return cp.asnumpy(filtered_gpu)

        filtered = np.empty_like(data)
        for t in range(n_echos):
            filtered[..., t] = ndimage.gaussian_filter(data[..., t], sigma=sigma)
        return filtered

    if method == "savgol_temporal":
        n_echos = data.shape[3]
        win = min(window, n_echos if n_echos % 2 == 1 else n_echos - 1)
        if win <= poly:
            print(
                f"[Filter] Not enough echoes ({n_echos}) for Savitzky-Golay "
                f"(window={win}, poly={poly}) — skipping temporal filter."
            )
            return data
        return savgol_filter(data, window_length=win, polyorder=poly, axis=-1)

    raise ValueError(
        f"Unknown method '{method}'. Choose from "
        f"['none', 'gaussian_spatial', 'savgol_temporal']."
    )


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


def mask_rician(data, k=None, p_false=1e-6, use_morpho=False):
    """Binary tissue mask: threshold on the max projection, set from the noise level.

    The threshold is ``k * sigma``, with ``sigma`` the noise level of one echo
    (:func:`estimate_sigma`; for 3-D input, the Rician floor
    ``mean(background) / sqrt(pi/2)``).

    How ``k`` is chosen (``k=None``): the mask is applied to the MAXIMUM over
    the ``n`` echoes, so the threshold must clear the largest of ``n`` noise
    draws. For pure noise (Rayleigh law) one draw exceeds ``T`` with
    probability ``q = exp(-T^2 / 2 sigma^2)``, and the maximum of ``n`` draws
    exceeds it with probability ``p = 1 - (1 - q)^n``. Setting ``p`` to
    ``p_false`` and solving for ``T`` gives ``k = sqrt(-2 ln q)``, with
    ``q = 1 - (1 - p_false)^(1/n)``. With the default ``p_false = 1e-6`` and 32
    echoes, ``k`` is about 5.9: on average less than one background voxel per
    million is wrongly labelled tissue. ``k`` fixes the threshold directly if given.

    References
    ----------
    Gudbjartsson, H., & Patz, S. (1995). The Rician distribution of noisy
    MRI data. *Magnetic Resonance in Medicine*, 34(6), 910-914.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz) or (nx, ny, nz, n_te)
        Raw MRI data.
    k : float or None, optional
        Threshold in units of sigma. If None (default), derived from ``p_false``.
    p_false : float, optional
        Accepted probability that a pure-noise voxel passes the threshold.
        Default 1e-6.
    use_morpho : bool, optional
        If ``True``, apply binary closing and hole-filling after thresholding.

    Returns
    -------
    mask : np.ndarray of shape (nx, ny, nz), dtype bool
        Binary mask where ``True`` indicates tissue.

    Examples
    --------
    >>> mask = mask_rician(data)
    """
    vol = np.max(data, axis=-1) if data.ndim == 4 else data

    _, _, bg_mask, _ = estimate_noise_auto(data)
    if data.ndim == 4:
        sigma = estimate_sigma(data, bg_mask=bg_mask)
        n_te = data.shape[-1]
    else:
        bg_values = vol[bg_mask]
        bg_values = bg_values[bg_values > 0]
        sigma = np.mean(bg_values) / np.sqrt(np.pi / 2)
        n_te = 1

    if k is None:
        q = -np.expm1(np.log1p(-p_false) / n_te)   # per-echo exceedance probability
        k = float(np.sqrt(-2.0 * np.log(q)))
    threshold = k * sigma
    print(f"[Mask] threshold = {k:.2f} x sigma = {threshold:.0f}  "
          f"(sigma = {sigma:.0f}, {n_te} echo(es))")

    mask = vol > threshold
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
        (e.g. ``p_false=1e-6`` or ``k=5.9`` for Rician, ``k=3.5`` for histogram).

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
    >>> mask = compute_mask(data, method="rician", p_false=1e-6, use_morpho=True)
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