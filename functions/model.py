"""
T2 decay models and their fit under Rician noise.

Two models describe the spin-echo signal ``S(t)`` as a function of the echo
time ``t`` (ms):

- **mono** : ``S(t) = I0 * exp(-t / T2)``
- **bi**   : ``S(t) = I0 * [f * exp(-t / T2c) + (1 - f) * exp(-t / T2l)]``
  (``T2c`` = short component, ``T2l`` = long component, ``f`` = share of the
  short one)

How the fit works
-----------------
An MRI magnitude image is never negative, so its noise is not symmetric around
the true signal: when the true signal is small, the measured value stays above
zero, at about ``1.25 * sigma`` (the "noise floor"). A plain least-squares fit
reads that floor as real signal and over-estimates the long T2.

Here the noise level ``sigma`` is measured once in the background of the image
(:func:`functions.utils.estimate_sigma`), then the fit looks for the parameters
that make the measured values the **most probable** under the exact law of the
noise (the Rice law): this is the *maximum likelihood* method. It needs no extra
"offset" parameter, the noise floor is already part of the model.

Which model to trust
--------------------
``bi`` always fits at least as well as ``mono`` since it has more freedom, so a
better fit is not enough. :func:`fit_both` keeps ``bi`` only if it improves the
likelihood by more than its two extra parameters cost (AICc criterion).

Several voxels
--------------
Giving a 2-D array (voxels x echoes) to :func:`fit_model` fits all the voxels
together, with the same likelihood: nothing is averaged before the fit, so the
noise floor is not distorted. The voxels share the **shape** of the decay (T2,
or T2c, T2l and f) but each keeps its **own amplitude**, because two voxels of
the same tissue rarely have the same signal level. This is what "Zone fit"
uses.

Notes
-----
All T2 values are in milliseconds. The search bounds depend only on the
acquisition (first and last echo times), never on the voxel being fitted.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e, i1e

MODELS = ("mono", "bi")

# bi is kept only if AICc(mono) - AICc(bi) is larger than this. 0 would be the plain
# "lowest AICc wins" rule (about 3 % of pure-mono voxels called bi); 2 brings that
# false-alarm rate to about 1 %. Measured by ``python validation.py`` (section 4).
DELTA_AICC = 2.0

# Starting points tried for the bi fit, as (f, short T2, long T2), the T2 being
# multiples of the mono T2 found first. The best result of the starts is kept.
_BI_STARTS = ((0.5, 0.4, 2.5), (0.3, 0.25, 1.6), (0.7, 0.6, 4.0))


# ── Signal models ─────────────────────────────────────────────────────────────

def signal_mono(t, I0, T2):
    """Mono-exponential decay."""
    return I0 * np.exp(-t / T2)


def signal_bi(t, I0, f, T2c, T2l):
    """Bi-exponential decay (``f`` = share of the short component ``T2c``)."""
    return I0 * (f * np.exp(-t / T2c) + (1 - f) * np.exp(-t / T2l))


def t2_limits(te):
    """Fixed physical search range for every T2, from the acquisition only.

    Below half the first echo time a decay is over before it is measured; above
    five times the last echo time the signal is nearly flat over the whole
    acquisition, so T2 cannot be read from it.
    """
    return 0.5 * float(te[0]), 5.0 * float(te[-1])


# ── Rice law ──────────────────────────────────────────────────────────────────

def rice_mean(nu, sigma):
    """Mean of a Rician magnitude whose noise-free value is ``nu``.

    Equals ``sigma * sqrt(pi/2)`` (the noise floor) when ``nu = 0`` and tends to
    ``nu`` when ``nu`` is much larger than ``sigma``.
    """
    b = np.asarray(nu, dtype=float) ** 2 / (4.0 * sigma**2)
    return sigma * np.sqrt(np.pi / 2.0) * ((1 + 2 * b) * i0e(b) + 2 * b * i1e(b))


def rice_var(nu, sigma):
    """Variance of a Rician magnitude whose noise-free value is ``nu``."""
    nu = np.asarray(nu, dtype=float)
    return np.maximum(2.0 * sigma**2 + nu**2 - rice_mean(nu, sigma) ** 2, 1e-12 * sigma**2)


def rice_loglik(M, nu, sigma):
    """Log-likelihood of measured magnitudes ``M`` for noise-free values ``nu``.

    Terms that do not depend on ``nu`` are left out: they are identical for every
    model fitted on the same data, so they cancel in any model comparison.
    """
    x = M * nu / sigma**2
    return float(np.sum(-((M - nu) ** 2) / (2 * sigma**2) + np.log(i0e(x))))


# ── Maximum-likelihood fit ────────────────────────────────────────────────────

def _shape(theta, t, model):
    """Unit-amplitude decay ``s(t)`` and its derivatives for the internal shape parameters.

    Internal shape parameters: ``log(T2)`` for ``mono``; the share ``f`` then
    ``log(T2)`` of each component for ``bi`` (the log keeps every T2 positive and
    treats 10 -> 20 ms like 100 -> 200 ms). Returns ``s`` of shape ``(n_te,)`` and
    ``ds`` of shape ``(n_te, n_shape_params)``.
    """
    if model == "mono":
        e = np.exp(-t / np.exp(theta[0]))
        return e, (e * t / np.exp(theta[0]))[:, None]
    f, lc, ll = theta
    ec, el = np.exp(-t / np.exp(lc)), np.exp(-t / np.exp(ll))
    ds = np.stack([ec - el, f * ec * t / np.exp(lc), (1 - f) * el * t / np.exp(ll)], axis=1)
    return f * ec + (1 - f) * el, ds


def _neg_loglik(theta, t, y, sigma, model):
    """Negative log-likelihood of all the voxels in ``y`` (n_voxels x n_te) and its gradient.

    ``theta`` = one amplitude per voxel, then the shared shape parameters.
    """
    m = y.shape[0]
    amp = theta[:m]
    s, ds = _shape(theta[m:], t, model)
    nu = amp[:, None] * s[None, :]
    x = y * nu / sigma**2
    e0, e1 = i0e(x), i1e(x)
    nll = np.sum((y - nu) ** 2 / (2 * sigma**2) - np.log(e0))
    # d(nll)/d(nu) = (nu - y * I1(x)/I0(x)) / sigma^2 for every measurement
    g_nu = (nu - y * e1 / e0) / sigma**2
    return float(nll), np.concatenate([g_nu @ s, (amp @ g_nu) @ ds])


def _start_values(te, mean_curve, sigma):
    """Amplitude and T2 from a straight-line fit of log(signal), on the echoes
    clearly above the noise floor (at least the first three)."""
    n_ok = min(max(3, int(np.sum(mean_curve > 2.0 * sigma))), len(te))
    slope, intercept = np.polyfit(te[:n_ok], np.log(np.maximum(mean_curve[:n_ok], 1e-12)), 1)
    return float(np.exp(intercept)), (-1.0 / slope if slope < 0 else np.inf)


def fit_model(model, te, signal, sigma):
    """Fit one model by maximum likelihood under Rician noise.

    Parameters
    ----------
    model : {"mono", "bi"}
    te : array-like of shape (n_te,)
        Echo times in ms.
    signal : array-like of shape (n_te,) or (n_voxels, n_te)
        Measured magnitudes. A 2-D array is treated as several voxels of one
        homogeneous zone: they share the shape of the decay but each has its own
        amplitude (see the module docstring).
    sigma : float
        Noise level of the acquisition, in the units of ``signal``.

    Returns
    -------
    result : dict or None
        ``params`` (``I0``, ``T2`` for mono; ``I0``, ``f``, ``T2c``, ``T2l`` for
        bi; for a zone ``I0`` is the mean amplitude of its voxels), ``fitted``
        (noise-free signal at ``te``; for a zone, of the average voxel),
        ``expected`` (what the model predicts you will *measure*: the fitted
        signal plus the noise floor), ``amp`` (amplitude of each voxel) and
        ``shape`` (the unit decay), ``loglik``, ``aicc``, ``n_obs``, ``n_params``
        (all the free parameters, the amplitudes included) and ``at_bound`` (a T2
        ended on a limit of the search range, so its value is not reliable).
        ``None`` if the data cannot be fitted.
    """
    te = np.asarray(te, dtype=float)
    y = np.atleast_2d(np.asarray(signal, dtype=float))
    m = y.shape[0]
    mean_curve = y.mean(axis=0)
    scale = float(mean_curve.max())
    if not (np.all(np.isfinite(y)) and scale > 0 and sigma > 0):
        return None

    yn, sn = y / scale, sigma / scale
    t_lo, t_hi = t2_limits(te)
    lo, hi = np.log(t_lo), np.log(t_hi)
    i0_start, t2_start = _start_values(te, mean_curve / scale, sn)
    t2_start = float(np.clip(t2_start, t_lo * 1.5, t_hi / 1.5))
    # each voxel starts at the amplitude of the mean curve, scaled by its own first echoes
    gain = np.clip(y[:, :3].mean(axis=1) / max(mean_curve[:3].mean(), 1e-12), 0.05, 20.0)
    amp0 = i0_start * gain

    if model == "mono":
        shape_starts = [[np.log(t2_start)]]
        shape_bounds = [(lo, hi)]
    else:
        shape_starts = [[f0, np.log(t2_start * a), np.log(min(t2_start * b, t_hi / 1.5))] for f0, a, b in _BI_STARTS]
        shape_bounds = [(0, 1), (lo, hi), (lo, hi)]
    bounds = [(0, 50)] * m + shape_bounds
    lower, upper = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

    best = None
    for sh in shape_starts:
        x0 = np.clip(np.concatenate([amp0, sh]), lower, upper)
        res = minimize(_neg_loglik, x0, args=(te, yn, sn, model), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None

    amp, th = best.x[:m] * scale, best.x[m:]
    if model == "mono":
        params = {"I0": float(amp.mean()), "T2": float(np.exp(th[0]))}
        t2_checked = [params["T2"]]
        shape = signal_mono(te, 1.0, params["T2"])
    else:
        f, t2c, t2l = float(th[0]), float(np.exp(th[1])), float(np.exp(th[2]))
        if t2c > t2l:                       # the two labels are interchangeable:
            f, t2c, t2l = 1 - f, t2l, t2c   # always call the short one T2c
        params = {"I0": float(amp.mean()), "f": f, "T2c": t2c, "T2l": t2l}
        shape = signal_bi(te, 1.0, f, t2c, t2l)
        # a component that carries (almost) no signal has a meaningless T2
        t2_checked = [t for t, share in ((t2c, f), (t2l, 1 - f)) if share > 0.01]

    nu = amp[:, None] * shape[None, :]
    n_obs, k = y.size, m + len(th)
    loglik = rice_loglik(y, nu, sigma)
    penalty = 2 * k + (2 * k * (k + 1) / (n_obs - k - 1) if n_obs - k - 1 > 0 else np.inf)
    return {
        "model": model, "params": params, "fitted": nu.mean(axis=0),
        "expected": rice_mean(nu, sigma).mean(axis=0), "amp": amp, "shape": shape,
        "loglik": loglik, "aicc": -2.0 * loglik + penalty, "n_obs": n_obs, "n_params": k,
        "at_bound": any(t <= t_lo * 1.001 or t >= t_hi * 0.999 for t in t2_checked),
    }


def fit_both(te, signal, sigma, delta=DELTA_AICC):
    """Fit mono and bi, then keep the one the data justify.

    Returns
    -------
    out : dict
        ``mono`` and ``bi`` (results of :func:`fit_model`, or ``None``), ``best``
        (``"mono"`` or ``"bi"``) and ``delta_aicc`` (``AICc(mono) - AICc(bi)``;
        positive = the data favour bi). ``bi`` wins only if ``delta_aicc`` is
        larger than ``delta``.
    """
    mono = fit_model("mono", te, signal, sigma)
    bi = fit_model("bi", te, signal, sigma)
    d_aicc = mono["aicc"] - bi["aicc"] if (mono and bi) else np.nan
    best = "bi" if (bi is not None and (mono is None or d_aicc > delta)) else "mono"
    return {"mono": mono, "bi": bi, "best": best, "delta_aicc": d_aicc}


def standardized_residuals(result, signal, sigma):
    """Measured minus expected, divided by the spread the Rice law predicts there.

    Same shape as ``signal``. If the model is right they scatter around 0 with a
    spread of 1, whatever the echo (near the noise floor the spread is smaller
    than ``sigma`` and the Rice law accounts for it). For a zone, each voxel is
    compared with the decay it was fitted with (its own amplitude).
    """
    y = np.atleast_2d(np.asarray(signal, dtype=float))
    nu = result["amp"][:, None] * result["shape"][None, :]
    resid = (y - rice_mean(nu, sigma)) / np.sqrt(rice_var(nu, sigma))
    return resid.reshape(np.shape(signal))


def fit_quality(result, signal, sigma):
    """Reduced chi-square of a fit, using the known noise level.

    Sum of the squared :func:`standardized_residuals`, divided by the number of
    measurements minus the number of free parameters. If the model explains the
    data down to the noise, the result is close to 1; much larger than 1 means
    something is missing from the model (or the zone is not homogeneous); much
    smaller than 1 means a ``sigma`` that is too high.
    """
    resid = standardized_residuals(result, signal, sigma)
    return float(np.sum(resid**2) / (resid.size - result["n_params"]))
