"""
Simulation tools to test the fits against a known truth.

- :func:`rician`, :func:`true_curve` : simulated noisy decays, with the same noise
  law as a real magnitude MRI image
- :func:`crlb`, :func:`snr_needed` : the best precision ANY method can reach with
  that noise (Cramer-Rao bound, computed with the Rice law)
- :func:`monte_carlo` and :func:`trial` : many independent noisy trials, run in
  parallel (``trial`` lives here so the joblib workers can import it)
- :func:`ls_mono`, :func:`ls_mono_offset` : plain least squares (the former way of
  fitting), kept only to show in ``validation.py`` why it was replaced

Truth description
-----------------
A truth is a dict with the key ``"snr"`` (I0 / sigma) and either
``{"kind": "mono", "T2": ...}`` or ``{"kind": "bi", "f": ..., "T2c": ..., "T2l": ...}``.
"""

import math

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from scipy.special import i0e, i1e

from .model import fit_both, signal_bi, signal_mono

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ── Noisy decays ──────────────────────────────────────────────────────────────

def rician(nu, sigma, rng):
    """Magnitude of the signal ``nu`` plus complex Gaussian noise of level ``sigma``.

    This is what a magnitude MRI image measures: noise on both the real and the
    imaginary channel, then the modulus. It never goes below zero.
    """
    nu = np.asarray(nu, dtype=float)
    return np.hypot(nu + sigma * rng.standard_normal(nu.shape), sigma * rng.standard_normal(nu.shape))


def true_curve(te, truth, sigma):
    """Noise-free decay of a ``truth`` (see the module docstring), amplitude ``snr * sigma``."""
    i0 = truth["snr"] * sigma
    if truth["kind"] == "mono":
        return signal_mono(te, i0, truth["T2"])
    return signal_bi(te, i0, truth["f"], truth["T2c"], truth["T2l"])


# ── Cramer-Rao bound with Rician noise ───────────────────────────────────────

def rician_fisher_info(amplitude, sigma, n_grid=2000):
    """Fisher information carried by one Rician measurement about its true amplitude.

    Computed by numerical integration of the squared score of the Rice density,
    so there is no closed formula to trust. It tends to ``1/sigma^2`` at high SNR
    (Gaussian limit) and to 0 when the amplitude goes to 0: a value near the noise
    floor says almost nothing about the amplitude.

    Parameters
    ----------
    amplitude : array-like of shape (n,)
        True (noise-free) amplitudes.
    sigma : float
        Noise level (standard deviation of each Gaussian channel).

    Returns
    -------
    np.ndarray of shape (n,)
    """
    a = np.atleast_1d(np.asarray(amplitude, dtype=float))
    # Integrate only where the density lives: [A - 12 sigma, A + 12 sigma] (clipped at 0)
    lo = np.maximum(0.0, a[:, None] - 12.0 * sigma)
    m = lo + np.linspace(0.0, 1.0, n_grid)[None, :] * (a[:, None] + 12.0 * sigma - lo)
    z = m * a[:, None] / sigma**2
    pdf = m / sigma**2 * np.exp(-((m - a[:, None]) ** 2) / (2 * sigma**2)) * i0e(z)
    ratio = np.divide(i1e(z), i0e(z))                       # I1/I0, numerically stable
    score = -a[:, None] / sigma**2 + m / sigma**2 * ratio
    return _trapezoid(pdf * score**2, m, axis=1)


def crlb(te, truth, sigma, n_vox=1):
    """Smallest standard deviation any unbiased method can reach on each parameter.

    Parameters
    ----------
    te : array-like
        Echo times (ms).
    truth : dict
        See the module docstring.
    sigma : float
        Noise level.
    n_vox : int, optional
        Number of voxels of one homogeneous zone fitted together (the bound
        then shrinks by ``sqrt(n_vox)``).

    Returns
    -------
    dict
        Mono: ``T2`` (relative). Bi: ``T2c`` and ``T2l`` (relative), ``f`` (absolute).
        ``inf`` where the parameters cannot be told apart.
    """
    te = np.asarray(te, dtype=float)
    i0 = truth["snr"] * sigma
    if truth["kind"] == "mono":
        t2 = truth["T2"]
        e = np.exp(-te / t2)
        amp = i0 * e
        jac = np.stack([e, i0 * e * te / t2**2], axis=1)
        names, refs = ["I0", "T2"], [i0, t2]
    else:
        f, t2c, t2l = truth["f"], truth["T2c"], truth["T2l"]
        ea, eb = np.exp(-te / t2c), np.exp(-te / t2l)
        amp = i0 * (f * ea + (1 - f) * eb)
        jac = np.stack([f * ea + (1 - f) * eb, i0 * (ea - eb),
                        i0 * f * ea * te / t2c**2, i0 * (1 - f) * eb * te / t2l**2], axis=1)
        names, refs = ["I0", "f", "T2c", "T2l"], [i0, f, t2c, t2l]
    fim = (jac.T * rician_fisher_info(amp, sigma)) @ jac
    try:
        with np.errstate(invalid="ignore"):
            sd = np.sqrt(np.diag(np.linalg.inv(fim))) / math.sqrt(n_vox)
        sd = np.where(np.isfinite(sd), sd, np.inf)
    except np.linalg.LinAlgError:
        sd = np.full(len(names), np.inf)
    out = {}
    for name, s, ref in zip(names, sd, refs):
        if name == "f":
            out["f"] = float(s)
        elif name != "I0":
            out[name] = float(s / ref)
    return out


def snr_needed(te, truth, sigma, target_cv, n_vox=1):
    """Smallest SNR (I0 / sigma) at which every T2 of ``truth`` reaches the relative
    precision ``target_cv``. ``inf`` if no SNR up to 1e6 is enough."""
    def worst(snr):
        r = crlb(te, {**truth, "snr": snr}, sigma, n_vox)
        return max(v for k, v in r.items() if k != "f")

    lo, hi = 1.0, 1e6
    if worst(hi) > target_cv:
        return float("inf")
    for _ in range(50):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if worst(mid) > target_cv else (lo, mid)
    return hi


# ── Plain least squares (the former way of fitting, for comparison only) ──────

def _log_start(te, y):
    """Starting amplitude and T2 from a straight line through the log of the first echoes."""
    slope, intercept = np.polyfit(te[:5], np.log(np.maximum(y[:5], 1.0)), 1)
    return math.exp(intercept), (-1.0 / slope if slope < 0 else 100.0)


def ls_mono(te, y):
    """Plain least-squares mono-exponential fit (no noise model).

    Returns ``[I0, T2]`` (NaN when the fit fails).
    """
    i0, t2 = _log_start(te, y)
    try:
        popt, _ = curve_fit(signal_mono, te, y, p0=[i0, t2], bounds=([0, 0.1], [np.inf, 5000.0]))
    except (RuntimeError, ValueError):
        return np.full(2, np.nan)
    return popt


def ls_mono_offset(te, y):
    """Plain least-squares mono-exponential fit plus a free constant.

    Returns ``[I0, T2, C]`` (NaN when the fit fails).
    """
    i0, t2 = _log_start(te, y)
    try:
        popt, _ = curve_fit(lambda t, a, b, c: signal_mono(t, a, b) + c, te, y,
                            p0=[i0, t2, 0.0], bounds=([0, 0.1, 0], [np.inf, 5000.0, np.inf]))
    except (RuntimeError, ValueError):
        return np.full(3, np.nan)
    return popt


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def trial(seed, te, truth, sigma, n_vox=1, het=0.0, sigma_factor=1.0, compare=False, echo_gain=None, skip=0):
    """One noisy trial: simulate, fit mono and bi, decide, return plain numbers.

    Parameters
    ----------
    seed : int
        Seed of this trial (each trial uses its own).
    te : array-like
        Echo times (ms).
    truth : dict
        See the module docstring.
    sigma : float
        True noise level used to simulate.
    n_vox : int, optional
        Number of voxels of the zone. 1 = one voxel; more = the voxels are
        fitted together (joint likelihood), as ``Zone fit`` does.
    het : float, optional
        Relative spread of the amplitude from voxel to voxel (0.25 = +-25 %).
    sigma_factor : float, optional
        The fit is given ``sigma * sigma_factor`` (to test a wrong noise level).
    compare : bool, optional
        Also fit the former way: plain least squares (``ls``, ``ls_off``) and, for a
        zone, plain least squares on the voxel-averaged magnitudes (``ls_mean``).
    echo_gain : array-like of shape (n_te,), optional
        Factor applied to the noise-free signal at each echo (1.03 = 3 % too high),
        to mimic echoes that do not follow the model.
    skip : int, optional
        Number of first echoes left out of the fit (``SKIP_FIRST_ECHOES`` in ``main.py``).

    Returns
    -------
    dict of floats
        ``bi_kept`` (1/0), ``delta_aicc``, ``mono_T2``, ``mono_limit``, ``bi_f``,
        ``bi_T2c``, ``bi_T2l``, ``bi_limit`` (NaN if a fit failed), and the
        least-squares values if ``compare``.
    """
    rng = np.random.default_rng(seed)
    te = np.asarray(te, dtype=float)
    amp = true_curve(te, truth, sigma)
    if echo_gain is not None:
        amp = amp * np.asarray(echo_gain, dtype=float)
    if n_vox > 1:
        gains = np.maximum(1.0 + het * rng.standard_normal((n_vox, 1)), 0.1)
        y = rician(gains * amp, sigma, rng)
    else:
        y = rician(amp, sigma, rng)
    out = fit_both(te[skip:], y[..., skip:], sigma * sigma_factor)
    nan = float("nan")
    mono, bi = out["mono"], out["bi"]
    row = {
        "bi_kept": float(out["best"] == "bi"), "delta_aicc": out["delta_aicc"],
        "mono_T2": mono["params"]["T2"] if mono else nan,
        "mono_limit": float(mono["at_bound"]) if mono else nan,
        "bi_f": bi["params"]["f"] if bi else nan,
        "bi_T2c": bi["params"]["T2c"] if bi else nan,
        "bi_T2l": bi["params"]["T2l"] if bi else nan,
        "bi_limit": float(bi["at_bound"]) if bi else nan,
    }
    if compare:
        first = y if n_vox == 1 else y[0]
        row["ls"] = float(ls_mono(te, first)[1])
        row["ls_off"] = float(ls_mono_offset(te, first)[1])
        if n_vox > 1:
            row["ls_mean"] = float(ls_mono(te, y.mean(axis=0))[1])
    return row


def monte_carlo(worker, n_trials, seed=0, n_jobs=-1, **kwargs):
    """Run ``worker(seed=..., **kwargs)`` ``n_trials`` times in parallel.

    The worker returns a dict of floats; the result is one array per key.
    ``worker`` must live in an importable module (like :func:`trial`).
    """
    results = Parallel(n_jobs=n_jobs)(delayed(worker)(seed=seed + i, **kwargs) for i in range(n_trials))
    return {key: np.array([r[key] for r in results], dtype=float) for key in results[0]}
