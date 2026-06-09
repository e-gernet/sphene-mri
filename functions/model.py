import numpy as np
from scipy.optimize import curve_fit


# ── Modèles analytiques ──────────────────────────────────────────────────────

def _mono(t, I0, T2):
    return I0 * np.exp(-t / T2)

def _mono_offset(t, I0, T2, C):
    return I0 * np.exp(-t / T2) + C

def _bi(t, I0, f, T2c, T2l):
    return I0 * (f * np.exp(-t / T2c) + (1 - f) * np.exp(-t / T2l))

def _bi_offset(t, I0, f, T2c, T2l, C):
    return I0 * (f * np.exp(-t / T2c) + (1 - f) * np.exp(-t / T2l)) + C


# ── Estimation p0 ────────────────────────────────────────────────────────────

def _estimate_p0(te, signal):
    """
    Estimation robuste des paramètres initiaux par régression log-linéaire.
    Retourne (I0_est, T2_est).
    """
    s = np.array(signal, dtype=float)
    mask = s > 0
    if mask.sum() < 2:
        return float(s[0]) if s[0] > 0 else 1.0, 50.0

    log_s = np.log(s[mask])
    t_m   = np.array(te)[mask]

    # Régression linéaire sur log(S) = log(I0) - t/T2
    coeffs = np.polyfit(t_m, log_s, 1)      # coeffs[0]=slope, coeffs[1]=intercept
    T2_est = max(-1.0 / coeffs[0], 1.0)     # -1/slope, clampé > 1 ms
    I0_est = np.exp(coeffs[1])

    return I0_est, T2_est


# ── Fit mono ─────────────────────────────────────────────────────────────────

def fit_mono(te, signal):
    """
    Fit mono-exponentiel : S(t) = I0 · exp(-t/T2)
    Retourne : (params_dict, fitted_signal, covariance) ou (None, None, None)
    params_dict : {"I0": ..., "T2": ...}
    """
    te  = np.asarray(te,     dtype=float)
    sig = np.asarray(signal, dtype=float)

    I0_est, T2_est = _estimate_p0(te, sig)
    p0     = [I0_est, T2_est]
    bounds = ([0, 0.1], [np.inf, 5000.0])

    try:
        popt, pcov = curve_fit(
            _mono, te, sig,
            p0=p0, bounds=bounds,
            method="trf", maxfev=10000,
        )
        fitted  = _mono(te, *popt)
        params  = {"I0": popt[0], "T2": popt[1]}
        return params, fitted, pcov

    except Exception:
        return None, None, None


# ── Fit mono + offset ────────────────────────────────────────────────────────

def fit_mono_offset(te, signal):
    """
    Fit mono-exponentiel avec offset : S(t) = I0 · exp(-t/T2) + C
    Retourne : (params_dict, fitted_signal, covariance) ou (None, None, None)
    params_dict : {"I0": ..., "T2": ..., "C": ...}
    """
    te  = np.asarray(te,     dtype=float)
    sig = np.asarray(signal, dtype=float)

    I0_est, T2_est = _estimate_p0(te, sig)
    C_est  = float(np.percentile(sig, 5))   # estimé bas du signal = offset
    p0     = [I0_est, T2_est, C_est]
    bounds = ([0, 0.1, 0], [np.inf, 5000.0, np.inf])

    try:
        popt, pcov = curve_fit(
            _mono_offset, te, sig,
            p0=p0, bounds=bounds,
            method="trf", maxfev=10000,
        )
        fitted = _mono_offset(te, *popt)
        params = {"I0": popt[0], "T2": popt[1], "C": popt[2]}
        return params, fitted, pcov

    except Exception:
        return None, None, None


# ── Fit bi-exponentiel ───────────────────────────────────────────────────────

def fit_bi(te, signal):
    """
    Fit bi-exponentiel : S(t) = I0·[f·exp(-t/T2c) + (1-f)·exp(-t/T2l)]
    Stratégie : ancrage sur T2_mono + ratio physiologique + filtre T2_eff.
    """
    te  = np.asarray(te, dtype=float)
    sig = np.asarray(signal, dtype=float)

    # Étape 1 — T2_mono comme référence physique
    I0_est, T2_mono = _estimate_p0(te, sig)

    # Étape 2 — initialisation ancrée sur T2_mono
    T2c_est = max(T2_mono / 4.0, te[0])
    T2l_est = min(T2_mono * 2.5, te[-1] * 0.8)
    T2l_est = max(T2l_est, T2c_est * 2.5)
    if T2l_est <= T2c_est:
        T2l_est = T2c_est * 3.0

    p0 = [I0_est, 0.5, T2c_est, T2l_est]

    # Bornes — ratio T2l/T2c max=10, T2l max=T2_mono*5
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

        # Étape 3 — filtre T2_eff : doit rester dans 2× T2_mono
        T2_eff = f_val * T2c_val + (1 - f_val) * T2l_val
        if abs(T2_eff - T2_mono) / (T2_mono + 1e-10) > 2.0:
            return None, None, None

        fitted = _bi(te, *popt)
        params = {"I0": popt[0], "f": popt[1],
                  "T2c": popt[2], "T2l": popt[3]}
        return params, fitted, pcov

    except Exception:
        return None, None, None


def fit_bi_offset(te, signal):
    """
    Fit bi-exponentiel avec offset.
    Même stratégie que fit_bi.
    """
    te  = np.asarray(te, dtype=float)
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

        fitted = _bi_offset(te, *popt)
        params = {"I0": popt[0], "f": popt[1], "T2c": popt[2],
                  "T2l": popt[3], "C": popt[4]}
        return params, fitted, pcov

    except Exception:
        return None, None, None