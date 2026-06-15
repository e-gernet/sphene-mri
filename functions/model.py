"""
Exponential decay models for MRI T2 relaxometry.

This module provides four analytical signal models and the fitting functions
that wrap ``scipy.optimize.curve_fit``. All models describe the spin-echo
signal decay as a function of echo time (TE).

Models
------
- Mono-exponential : ``S(t) = I0 · exp(-t / T2)``
- Mono-exponential with offset : ``S(t) = I0 · exp(-t / T2) + C``
- Bi-exponential : ``S(t) = I0 · [f · exp(-t / T2c) + (1-f) · exp(-t / T2l)]``
- Bi-exponential with offset : bi model + constant offset C

Notes
-----
All T2 values are in milliseconds. Initial parameter estimates are derived
from a log-linear regression on the signal to improve convergence speed and
robustness.
"""

import numpy as np
from scipy.optimize import curve_fit


# ── Analytical models ─────────────────────────────────────────────────────────

def _mono(t, I0, T2):
    return I0 * np.exp(-t / T2)


def _mono_offset(t, I0, T2, C):
    return I0 * np.exp(-t / T2) + C


def _bi(t, I0, f, T2c, T2l):
    return I0 * (f * np.exp(-t / T2c) + (1 - f) * np.exp(-t / T2l))


def _bi_offset(t, I0, f, T2c, T2l, C):
    return I0 * (f * np.exp(-t / T2c) + (1 - f) * np.exp(-t / T2l)) + C


# ── Initial parameter estimation ──────────────────────────────────────────────

def _estimate_p0(te, signal):
    """Estimate initial parameters via log-linear regression.

    Fits ``log(S) = log(I0) - t / T2`` by ordinary least squares on the
    positive signal values, providing robust starting points for non-linear
    optimisation.

    Parameters
    ----------
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    signal : array-like of shape (n_te,)
        Observed signal intensities (must contain at least 2 positive values).

    Returns
    -------
    I0_est : float
        Estimated signal amplitude at TE = 0.
    T2_est : float
        Estimated T2 relaxation time in milliseconds (clamped to ≥ 1 ms).
    """
    s = np.array(signal, dtype=float)
    mask = s > 0
    if mask.sum() < 2:
        return float(s[0]) if s[0] > 0 else 1.0, 50.0

    log_s = np.log(s[mask])
    t_m = np.array(te)[mask]
    coeffs = np.polyfit(t_m, log_s, 1)
    T2_est = max(-1.0 / coeffs[0], 1.0)
    I0_est = np.exp(coeffs[1])
    return I0_est, T2_est


# ── Mono-exponential fit ──────────────────────────────────────────────────────

def fit_mono(te, signal):
    """Fit a mono-exponential decay model to a T2 signal.

    Model: ``S(t) = I0 · exp(-t / T2)``

    Parameters
    ----------
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    signal : array-like of shape (n_te,)
        Observed signal intensities.

    Returns
    -------
    params : dict or None
        Fitted parameters ``{"I0": float, "T2": float}``.
        ``None`` if optimisation failed.
    fitted : np.ndarray of shape (n_te,) or None
        Model-predicted signal at the fitted parameters.
        ``None`` if optimisation failed.
    covariance : np.ndarray of shape (2, 2) or None
        Covariance matrix of the fitted parameters.
        ``None`` if optimisation failed.

    Examples
    --------
    >>> params, fitted, cov = fit_mono(te, signal)
    >>> if params is not None:
    ...     print(f"T2 = {params['T2']:.1f} ms")
    """
    te = np.asarray(te, dtype=float)
    sig = np.asarray(signal, dtype=float)

    I0_est, T2_est = _estimate_p0(te, sig)
    p0 = [I0_est, T2_est]
    bounds = ([0, 0.1], [np.inf, 5000.0])

    try:
        popt, pcov = curve_fit(
            _mono, te, sig,
            p0=p0, bounds=bounds,
            method="trf", maxfev=10000,
        )
        return {"I0": popt[0], "T2": popt[1]}, _mono(te, *popt), pcov
    except Exception:
        return None, None, None


# ── Mono-exponential with offset ──────────────────────────────────────────────

def fit_mono_offset(te, signal):
    """Fit a mono-exponential decay model with a constant offset.

    Model: ``S(t) = I0 · exp(-t / T2) + C``

    The offset C accounts for Rician noise floor or residual signal from
    long-T2 components not captured by a single exponential.

    Parameters
    ----------
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    signal : array-like of shape (n_te,)
        Observed signal intensities.

    Returns
    -------
    params : dict or None
        Fitted parameters ``{"I0": float, "T2": float, "C": float}``.
        ``None`` if optimisation failed.
    fitted : np.ndarray of shape (n_te,) or None
        Model-predicted signal. ``None`` if optimisation failed.
    covariance : np.ndarray of shape (3, 3) or None
        Covariance matrix. ``None`` if optimisation failed.

    Examples
    --------
    >>> params, fitted, cov = fit_mono_offset(te, signal)
    >>> if params is not None:
    ...     print(f"T2 = {params['T2']:.1f} ms, offset C = {params['C']:.1f}")
    """
    te = np.asarray(te, dtype=float)
    sig = np.asarray(signal, dtype=float)

    I0_est, T2_est = _estimate_p0(te, sig)
    C_est = float(np.percentile(sig, 5))
    p0 = [I0_est, T2_est, C_est]
    bounds = ([0, 0.1, 0], [np.inf, 5000.0, np.inf])

    try:
        popt, pcov = curve_fit(
            _mono_offset, te, sig,
            p0=p0, bounds=bounds,
            method="trf", maxfev=10000,
        )
        params = {"I0": popt[0], "T2": popt[1], "C": popt[2]}
        return params, _mono_offset(te, *popt), pcov
    except Exception:
        return None, None, None


# ── Bi-exponential fit ────────────────────────────────────────────────────────

def fit_bi(te, signal):
    """Fit a bi-exponential decay model to a T2 signal.

    Model: ``S(t) = I0 · [f · exp(-t / T2c) + (1-f) · exp(-t / T2l)]``

    where T2c is the short (constrained) component and T2l is the long
    (free) component. Initial parameters are anchored on the mono-exponential
    T2 estimate to improve convergence.

    A post-fit consistency check discards solutions whose effective T2
    (``f · T2c + (1-f) · T2l``) deviates by more than 200 % from the
    mono-exponential estimate, which typically indicates a degenerate fit.

    Parameters
    ----------
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    signal : array-like of shape (n_te,)
        Observed signal intensities.

    Returns
    -------
    params : dict or None
        Fitted parameters ``{"I0": float, "f": float, "T2c": float, "T2l": float}``.
        ``None`` if optimisation failed or consistency check failed.
    fitted : np.ndarray of shape (n_te,) or None
        Model-predicted signal. ``None`` on failure.
    covariance : np.ndarray of shape (4, 4) or None
        Covariance matrix. ``None`` on failure.

    Notes
    -----
    ``f`` is the fraction of the short-T2 (constrained water) component,
    bounded to [0.05, 0.95] to avoid degenerate single-component solutions.

    Examples
    --------
    >>> params, fitted, cov = fit_bi(te, signal)
    >>> if params is not None:
    ...     print(f"T2c = {params['T2c']:.1f} ms, T2l = {params['T2l']:.1f} ms")
    """
    te = np.asarray(te, dtype=float)
    sig = np.asarray(signal, dtype=float)

    I0_est, T2_mono = _estimate_p0(te, sig)

    T2c_est = max(T2_mono / 4.0, te[0])
    T2l_est = min(T2_mono * 2.5, te[-1] * 0.8)
    T2l_est = max(T2l_est, T2c_est * 2.5)
    if T2l_est <= T2c_est:
        T2l_est = T2c_est * 3.0

    p0 = [I0_est, 0.5, T2c_est, T2l_est]

    T2l_max = min(T2c_est * 10.0, T2_mono * 5.0, te[-1] * 3.0)
    T2c_max = min(T2l_est * 0.45, T2_mono * 1.5)
    bounds = (
        [0,      0.05, te[0] * 0.5, T2c_est * 2.0],
        [np.inf, 0.95, T2c_max,     T2l_max       ],
    )

    if bounds[0][2] >= bounds[1][2] or bounds[0][3] >= bounds[1][3]:
        return None, None, None

    try:
        popt, pcov = curve_fit(
            _bi, te, sig,
            p0=p0, bounds=bounds,
            method="trf", maxfev=20000,
        )
        f_val, T2c_val, T2l_val = popt[1], popt[2], popt[3]
        T2_eff = f_val * T2c_val + (1 - f_val) * T2l_val
        if abs(T2_eff - T2_mono) / (T2_mono + 1e-10) > 2.0:
            return None, None, None

        params = {"I0": popt[0], "f": popt[1], "T2c": popt[2], "T2l": popt[3]}
        return params, _bi(te, *popt), pcov
    except Exception:
        return None, None, None


# ── Bi-exponential with offset ────────────────────────────────────────────────

def fit_bi_offset(te, signal):
    """Fit a bi-exponential decay model with a constant offset.

    Model: ``S(t) = I0 · [f · exp(-t/T2c) + (1-f) · exp(-t/T2l)] + C``

    Uses the same anchoring strategy as :func:`fit_bi` with an additional
    offset parameter C estimated from the 5th percentile of the signal.

    Parameters
    ----------
    te : array-like of shape (n_te,)
        Echo times in milliseconds.
    signal : array-like of shape (n_te,)
        Observed signal intensities.

    Returns
    -------
    params : dict or None
        Fitted parameters
        ``{"I0": float, "f": float, "T2c": float, "T2l": float, "C": float}``.
        ``None`` if optimisation failed or consistency check failed.
    fitted : np.ndarray of shape (n_te,) or None
        Model-predicted signal. ``None`` on failure.
    covariance : np.ndarray of shape (5, 5) or None
        Covariance matrix. ``None`` on failure.

    Examples
    --------
    >>> params, fitted, cov = fit_bi_offset(te, signal)
    >>> if params is not None:
    ...     print(f"f = {params['f']:.2f}, C = {params['C']:.1f}")
    """
    te = np.asarray(te, dtype=float)
    sig = np.asarray(signal, dtype=float)

    I0_est, T2_mono = _estimate_p0(te, sig)

    T2c_est = max(T2_mono / 4.0, te[0])
    T2l_est = min(T2_mono * 2.5, te[-1] * 0.8)
    T2l_est = max(T2l_est, T2c_est * 2.5)
    if T2l_est <= T2c_est:
        T2l_est = T2c_est * 3.0

    C_est = float(np.percentile(sig, 5))
    p0 = [I0_est, 0.5, T2c_est, T2l_est, C_est]

    T2l_max = min(T2c_est * 10.0, T2_mono * 5.0, te[-1] * 3.0)
    T2c_max = min(T2l_est * 0.45, T2_mono * 1.5)
    bounds = (
        [0,      0.05, te[0] * 0.5, T2c_est * 2.0, 0     ],
        [np.inf, 0.95, T2c_max,     T2l_max,        np.inf],
    )

    if bounds[0][2] >= bounds[1][2] or bounds[0][3] >= bounds[1][3]:
        return None, None, None

    try:
        popt, pcov = curve_fit(
            _bi_offset, te, sig,
            p0=p0, bounds=bounds,
            method="trf", maxfev=20000,
        )
        f_val, T2c_val, T2l_val = popt[1], popt[2], popt[3]
        T2_eff = f_val * T2c_val + (1 - f_val) * T2l_val
        if abs(T2_eff - T2_mono) / (T2_mono + 1e-10) > 2.0:
            return None, None, None

        params = {
            "I0": popt[0], "f": popt[1],
            "T2c": popt[2], "T2l": popt[3], "C": popt[4],
        }
        return params, _bi_offset(te, *popt), pcov
    except Exception:
        return None, None, None