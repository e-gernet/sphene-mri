"""
validation.py -- Ce fichier d'IRM permet-il de retrouver ce que j'attends ?

Le script demande un NIfTI + un ACQP (fenetres de choix de fichiers) puis les
valeurs que tu ATTENDS : un couple T2c / T2l (avec f) et/ou un T2 mono.
Il affiche ensuite dans la console :

1. Les prerequis du fichier : temps d'echo, bruit (sigma), SNR par coupe.
2. La faisabilite (borne de Cramer-Rao, bruit de Rice) : la meilleure precision
   qu'une methode PEUT atteindre sur T2c, T2l et f avec ce SNR, pour 1 voxel ou
   pour une ROI de N voxels, et le SNR / nombre de voxels necessaires.
3. Un test par simulation avec les 4 modeles du pipeline (mono, mono+offset, bi,
   bi+offset) : on simule ton tissu attendu avec le bruit de TON fichier, on
   lance les memes fits et le meme choix par AIC que « Average + Fit », et on
   regarde si on retrouve les valeurs de depart.
4. (optionnel, ``--couples``) Meme test sur une liste de couples T2c/T2l
   (par ex. 12/36, 12/60, 12/120 ms) : lesquels sait-on retrouver ici ?

Pour tester le programme sans vraies donnees, genere des jeux simules avec
``python make_simulated.py`` : chaque jeu a une image .png du resultat attendu.

Usage
-----
    pixi run validate
    python validation.py --nifti a.nii --acqp a.acqp --t2c 12 --t2l 60 --f 0.5 --t2-mono 30
    python validation.py --nifti a.nii --acqp a.acqp --couples

Limites (a lire avant de citer les chiffres)
--------------------------------------------
- I0 (signal a TE=0) vient des donnees : signal median du tissu au 1er echo,
  ramene a TE=0 avec la decroissance attendue.
- Pour une ROI de N voxels, la borne de Cramer-Rao suppose qu'on moyenne AVANT de
  prendre le module (bruit divise par racine de N). « Average + Fit » moyenne
  APRES, ce qui garde le plancher de bruit : la simulation (partie 3) est donc
  le chiffre le plus honnete pour le viewer.
- Cramer-Rao est une borne basse de l'erreur : une vraie methode fait au mieux aussi bien.
"""

import argparse
import math

import numpy as np
from joblib import Parallel, delayed
from scipy.special import i0e, i1e

from functions.io import choose_acqp, choose_nifti, load_acqp, load_nifti
from functions.model import fit_bi, fit_bi_offset, fit_mono, fit_mono_offset
from functions.utils import compute_aic, compute_mask, estimate_sigma

MODELS = ("mono", "mono+offset", "bi", "bi+offset")
N_PARAMS = {"mono": 2, "mono+offset": 3, "bi": 4, "bi+offset": 5}
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

DEFAULT_COUPLES = [(12, 36), (12, 60), (12, 120), (25, 50), (8, 40), (20, 100)]  # f = 0.5


# ── Cramer-Rao bound with Rician noise ───────────────────────────────────────

def rician_fisher_info(amplitude, sigma, n_grid=2000):
    """Fisher information on the true amplitude A of a Rician measurement.

    Computed by direct numerical integration of the squared score of the
    Rice density, ``E[(d log p / dA)^2]``, so there is no closed-form
    formula to trust. Tends to ``1/sigma^2`` at high SNR (Gaussian limit) and
    to 0 as ``A -> 0`` (a magnitude near the noise floor says almost nothing
    about A).

    Parameters
    ----------
    amplitude : array-like of shape (n,)
        True (noise-free) amplitudes.
    sigma : float
        Noise standard deviation of the underlying Gaussian channels.

    Returns
    -------
    np.ndarray of shape (n,)
    """
    a = np.atleast_1d(np.asarray(amplitude, dtype=float))
    # Integrate only where the density lives: [A - 12 sigma, A + 12 sigma]
    # (clipped at 0). A grid spanning [0, A + 12 sigma] would not resolve the
    # narrow peak once A >> sigma.
    lo = np.maximum(0.0, a[:, None] - 12.0 * sigma)
    m = lo + np.linspace(0.0, 1.0, n_grid)[None, :] * (a[:, None] + 12.0 * sigma - lo)
    z = m * a[:, None] / sigma**2
    pdf = m / sigma**2 * np.exp(-((m - a[:, None]) ** 2) / (2 * sigma**2)) * i0e(z)
    ratio = np.divide(i1e(z), i0e(z))                       # I1/I0, stable
    score = -a[:, None] / sigma**2 + m / sigma**2 * ratio
    return _trapezoid(pdf * score**2, m, axis=1)


def crlb_bi(te, i0, f, t2c, t2l, sigma, n_avg=1):
    """Cramer-Rao standard deviations of (T2c, T2l, f) for a bi-exponential.

    Returns
    -------
    dict with ``t2c_cv``, ``t2l_cv`` (relative) and ``f_sd`` (absolute);
    ``inf`` where the information matrix is singular.
    """
    s = sigma / math.sqrt(n_avg)
    ea, eb = np.exp(-te / t2c), np.exp(-te / t2l)
    amp = i0 * (f * ea + (1 - f) * eb)
    jac = np.stack([f * ea + (1 - f) * eb, i0 * (ea - eb),
                    i0 * f * ea * te / t2c**2, i0 * (1 - f) * eb * te / t2l**2], axis=1)
    fim = (jac.T * rician_fisher_info(amp, s)) @ jac
    try:
        sd = np.sqrt(np.diag(np.linalg.inv(fim)))
    except np.linalg.LinAlgError:
        return {"t2c_cv": np.inf, "t2l_cv": np.inf, "f_sd": np.inf}
    return {"t2c_cv": sd[2] / t2c, "t2l_cv": sd[3] / t2l, "f_sd": sd[1]}


def snr_needed(te, f, t2c, t2l, target_cv):
    """Smallest I0/sigma for which both T2c and T2l reach ``target_cv`` (1 voxel)."""
    def worst(snr):
        r = crlb_bi(te, snr, f, t2c, t2l, 1.0)
        return max(r["t2c_cv"], r["t2l_cv"])
    lo, hi = 1.0, 1e6
    if worst(hi) > target_cv:
        return np.inf
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if worst(mid) > target_cv else (lo, mid)
    return hi


# ── Simulation with the pipeline's own models ───────────────────────────────

def fit_four(te, signal):
    """The viewer's 4 fits + AIC selection. Returns (best_name, params_by_model)."""
    fits = {"mono": fit_mono(te, signal), "mono+offset": fit_mono_offset(te, signal),
            "bi": fit_bi(te, signal), "bi+offset": fit_bi_offset(te, signal)}
    aic = {m: compute_aic(signal, fits[m][1], N_PARAMS[m]) for m in MODELS}
    return min(aic, key=aic.get), {m: fits[m][0] for m in MODELS}


def _rician(amp, sigma, rng):
    return np.hypot(amp + rng.normal(0, sigma, amp.shape), rng.normal(0, sigma, amp.shape))


def _trial(te, amp, sigma, n_vox, seed):
    rng = np.random.default_rng(seed)
    roi = _rician(np.tile(amp, (n_vox, 1)), sigma, rng).mean(axis=0)
    best, params = fit_four(te, roi)
    return best, params[best]


def monte_carlo(te, amp, sigma, n_vox, trials, seed0=0):
    """Fraction of trials won by each model, and the winning parameters."""
    res = Parallel(n_jobs=-1)(
        delayed(_trial)(te, amp, sigma, n_vox, seed0 + 7919 * n_vox + i) for i in range(trials)
    )
    return [r[0] for r in res], [r[1] for r in res]


def _q(values):
    v = np.array(values, dtype=float)
    return f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}-{np.percentile(v, 75):.1f}]"


# ── Rapports (console) ──────────────────────────────────────────────────────

def report_prerequisites(data, te, mask, sigma, voxel_dims):
    dte = np.diff(te)
    print("\n── 1. Prerequis du fichier " + "─" * 48)
    print(f"  dimensions            : {data.shape}  (x, y, coupe, echo)")
    print(f"  taille de voxel       : {voxel_dims[0]:.3f} x {voxel_dims[1]:.3f} x {voxel_dims[2]:.3f} mm")
    print(f"  echos                 : {len(te)}, TE de {te[0]:.2f} a {te[-1]:.2f} ms, "
          f"pas {dte.mean():.3f} ms{'' if np.allclose(dte, dte.mean(), rtol=1e-3) else '  (PAS regulier !)'}")
    print(f"  bruit (sigma)         : {sigma:.0f}  (mesure dans le fond, meme valeur pour tous les echos)")
    for z in range(data.shape[2]):
        m = mask[:, :, z]
        if m.any():
            first = np.median(data[:, :, z, 0][m]) / sigma
            last = np.median(data[:, :, z, -1][m]) / sigma
            print(f"  coupe z={z:<3} tissu n={int(m.sum()):<6}: SNR = {first:6.1f} au 1er echo, {last:5.1f} au dernier")
        else:
            print(f"  coupe z={z:<3} aucun voxel de tissu dans le masque")
    print("  (SNR = signal du tissu / bruit. Plus il est grand, mieux on separe deux T2.)")


def report_feasibility(te, sigma, snr_meas, f, t2c, t2l):
    print("\n── 2. Faisabilite (borne de Cramer-Rao, bruit de Rice) " + "─" * 20)
    print(f"  tissu attendu : T2c={t2c:g} ms, T2l={t2l:g} ms (rapport {t2l / t2c:.1f}), f={f:g}")
    print(f"  SNR mesure a TE=0 pour ce tissu : {snr_meas:.1f}")
    print("  meilleure precision POSSIBLE (erreur relative sur T2c et T2l) selon la taille de ROI :")
    print(f"    {'N voxels':>8} | {'T2c':>8} | {'T2l':>8} | {'f (abs.)':>8}")
    i0 = snr_meas * sigma
    for n in (1, 4, 16, 64, 256):
        r = crlb_bi(te, i0, f, t2c, t2l, sigma, n)
        print(f"    {n:>8} | {100 * r['t2c_cv']:>6.0f} % | {100 * r['t2l_cv']:>6.0f} % | {r['f_sd']:>8.3f}")
    for tgt in (0.30, 0.10):
        need = snr_needed(te, f, t2c, t2l, tgt)
        n_need = math.ceil((need / snr_meas) ** 2) if np.isfinite(need) else None
        txt = "impossible" if n_need is None else f"environ {n_need} voxel(s) au SNR mesure"
        print(f"  T2c et T2l a +/-{tgt:.0%} : SNR >= {need:.0f} par voxel  ->  {txt}")
    print("  ATTENTION : ce calcul suppose qu'on moyenne avant le module. « Average + Fit » moyenne apres :\n"
          "  il reste un plancher de bruit qui biaise T2l et f (voir partie 3).")


def report_montecarlo(te, sigma, snr_meas, kind, p, trials, n_list):
    if kind == "bi":
        f, t2c, t2l = p
        amp = snr_meas * sigma * (f * np.exp(-te / t2c) + (1 - f) * np.exp(-te / t2l))
        print(f"\n  tissu BI attendu : T2c={t2c:g}  T2l={t2l:g}  f={f:g}  (SNR {snr_meas:.0f})")
    else:
        amp = snr_meas * sigma * np.exp(-te / p)
        print(f"\n  tissu MONO attendu : T2={p:g}  (SNR {snr_meas:.0f})")
    print(f"    {'N':>4} | modele choisi par l'AIC (%): {'mono':>5} {'mono+C':>6} {'bi':>5} {'bi+C':>5} | valeurs retrouvees")
    for n in n_list:
        names, params = monte_carlo(te, amp, sigma, n, trials)
        pct = {m: 100 * names.count(m) / len(names) for m in MODELS}
        line = (f"    {n:>4} |                              {pct['mono']:>4.0f}% {pct['mono+offset']:>5.0f}% "
                f"{pct['bi']:>4.0f}% {pct['bi+offset']:>4.0f}% | ")
        if kind == "bi":
            bp = [pp for nm, pp in zip(names, params) if nm in ("bi", "bi+offset")]
            if len(bp) >= 5:
                line += (f"T2c {_q([x['T2c'] for x in bp])}, T2l {_q([x['T2l'] for x in bp])}, "
                         f"f {np.median([x['f'] for x in bp]):.2f}  (mediane [quartiles])")
            else:
                line += "le bi n'est presque jamais choisi"
        else:
            mp = [pp for nm, pp in zip(names, params) if nm in ("mono", "mono+offset")]
            line += f"T2 {_q([x['T2'] for x in mp])}" if mp else "le mono n'est jamais choisi"
        print(line)


def report_couples(te, sigma, snr_ref, trials, n_vox, f=0.5):
    """Plusieurs couples T2c/T2l : lesquels retrouve-t-on avec les modeles du pipeline ?"""
    print("\n── 4. Plusieurs couples T2c / T2l : lesquels retrouve-t-on ? " + "─" * 14)
    print(f"  f={f:g}, SNR={snr_ref:.0f}, ROI de {n_vox} voxels, {trials} essais par couple")
    print(f"    {'attendu':>10} | bi choisi | retrouve (mediane)")
    for t2c, t2l in DEFAULT_COUPLES:
        amp = snr_ref * sigma * (f * np.exp(-te / t2c) + (1 - f) * np.exp(-te / t2l))
        names, params = monte_carlo(te, amp, sigma, n_vox, trials)
        bp = [pp for nm, pp in zip(names, params) if nm in ("bi", "bi+offset")]
        share = 100 * len(bp) / len(names)
        res = (f"T2c {np.median([x['T2c'] for x in bp]):5.1f}  T2l {np.median([x['T2l'] for x in bp]):5.1f}  "
               f"f {np.median([x['f'] for x in bp]):.2f}") if len(bp) >= 5 else "non retrouve (bi presque jamais choisi)"
        print(f"    {t2c:>4g}/{t2l:<5g} | {share:>7.0f} % | {res}")


# ── Point d'entree ──────────────────────────────────────────────────────────

def _ask_float(prompt):
    raw = input(f"{prompt} (Entree pour passer) : ").strip().replace(",", ".")
    return float(raw) if raw else None


def main():
    ap = argparse.ArgumentParser(description="Ce fichier permet-il de retrouver ce que j'attends ?")
    ap.add_argument("--nifti"), ap.add_argument("--acqp")
    ap.add_argument("--t2c", type=float), ap.add_argument("--t2l", type=float)
    ap.add_argument("--f", type=float), ap.add_argument("--t2-mono", type=float)
    ap.add_argument("--trials", type=int, default=100, help="essais par taille de ROI (0 = sauter la partie 3)")
    ap.add_argument("--n-voxels", default="1,16,64", help="tailles de ROI pour la simulation")
    ap.add_argument("--couples", action="store_true", help="ajoute la partie 4 (plusieurs couples T2c/T2l)")
    a = ap.parse_args()

    nifti = a.nifti or choose_nifti()
    data, img = load_nifti(nifti)
    acqp = a.acqp or choose_acqp()
    te = load_acqp(acqp)
    voxel_dims = img.header.get_zooms()[:3]
    mask = compute_mask(data, method="rician")
    sigma = estimate_sigma(data)
    report_prerequisites(data, te, mask, sigma, voxel_dims)

    t2c, t2l, f, t2m = a.t2c, a.t2l, a.f, a.t2_mono
    if all(v is None for v in (t2c, t2l, f, t2m)):
        print("\nValeurs attendues (utilisees pour les parties 2 et 3) :")
        t2c, t2l, f = _ask_float("  T2c (ms)"), _ask_float("  T2l (ms)"), _ask_float("  f (0 a 1)")
        t2m = _ask_float("  T2 mono (ms)")

    zs = max(range(data.shape[2]), key=lambda z: int(mask[:, :, z].sum()))
    if not mask[:, :, zs].any():
        raise SystemExit("Aucun voxel de tissu dans aucune coupe.")
    first = np.median(data[:, :, zs, 0][mask[:, :, zs]])
    n_list = [int(v) for v in a.n_voxels.split(",")]

    have_bi = None not in (t2c, t2l, f)
    snr_bi = None
    if have_bi:
        decay0 = f * np.exp(-te[0] / t2c) + (1 - f) * np.exp(-te[0] / t2l)
        snr_bi = first / decay0 / sigma
        report_feasibility(te, sigma, snr_bi, f, t2c, t2l)
    if a.trials > 0 and (have_bi or t2m is not None):
        print("\n── 3. Simulation avec les 4 modeles du pipeline + AIC " + "─" * 21)
        print(f"  {a.trials} essais par taille de ROI, bruit de Rice, sigma={sigma:.0f}. ROI = moyenne de N voxels "
              "(comme « Average + Fit »).")
        if have_bi:
            report_montecarlo(te, sigma, snr_bi, "bi", (f, t2c, t2l), a.trials, n_list)
        if t2m is not None:
            snr_m = first / np.exp(-te[0] / t2m) / sigma
            report_montecarlo(te, sigma, snr_m, "mono", t2m, a.trials, n_list)
    if a.couples:
        snr_ref = snr_bi if snr_bi is not None else first / np.exp(-te[0] / 30.0) / sigma
        report_couples(te, sigma, snr_ref, max(a.trials, 20), n_list[-1])
    if not (have_bi or t2m is not None or a.couples):
        print("\nAucune valeur attendue : seuls les prerequis ont ete verifies.")


if __name__ == "__main__":
    main()
