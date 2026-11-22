"""
validation.py -- Les calculs de Sphene-MRI sont-ils justes ? Rapport complet par simulation.

Le script fabrique des mesures dont on CONNAIT la vérité, avec le même bruit que de
vraies images IRM, lance exactement les mêmes calculs que le viewer, et regarde si
on retrouve la vérité. Il écrit un dossier ``validation_report/`` :

- ``rapport.html``   : le rapport (un seul fichier, figures incluses, à ouvrir dans un navigateur)
- ``figures/*.png``  : les mêmes figures, une par fichier (pour un diaporama)
- ``controles.csv``  : chaque contrôle, sa valeur mesurée, son critère, RÉUSSI / ÉCHEC

Sections du rapport
-------------------
 1. Le bruit d'une image IRM (le « plancher » de bruit)
 2. T2 mono : biais et dispersion, comparés aux moindres carrés classiques
 3. T2 bi : précision de T2c, T2l et f
 4. Choix mono / bi : fausses alarmes et détection
 5. « Zone fit » : voxels ajustés ensemble
 6. Sensibilité à une erreur sur le niveau de bruit
 7. Premiers échos perturbés : quel effet, et faut-il les ignorer ?
 8. Cartes complètes sur jeux simulés (masque, bruit mesuré, ajustement voxel par voxel)
 9. Zone hétérogène : une limite à connaître
10. (si ``--nifti``) Vos données : le modèle de bruit tient-il ?
11. (si ``--t2c`` ... ``--t2-mono``) Votre tissu attendu : qu'espérer de votre acquisition ?

Usage
-----
    pixi run validate-quick        # 1 à 2 minutes : tendances et vérifications (peu de tirages)
    pixi run validate              # environ 8 minutes (2 coeurs) : tirages complets, à citer dans un document
    python validation.py --nifti a.nii --acqp a.acqp          # ajoute vos données (section 10)
    python validation.py --acqp a.acqp --t2c 12 --t2l 60 --f 0.5 --snr 30   # section 11
    python validation.py --choose-files                       # fenêtres de choix de fichiers
    python validation.py --presentation                       # 2 figures anglaises de la présentation (dossier validation_report/presentation/)

Le niveau de bruit et les temps d'écho par défaut sont ceux du grain de blé (sigma = 513,
32 échos de 2,9172 ms). Avec ``--acqp`` / ``--nifti``, les simulations utilisent VOS temps
d'écho et VOTRE niveau de bruit. Les tirages sont reproductibles (``--seed``).

Code de sortie : 0 si aucun contrôle n'échoue, 1 sinon.
"""

import argparse
import contextlib
import csv
import io
import itertools
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.stats import binom  # noqa: E402

import make_simulated  # noqa: E402
from functions import simulation as sim  # noqa: E402
from functions.mapping import fit_slice, slice_summary  # noqa: E402
from functions.model import DELTA_AICC, fit_both, rice_loglik, rice_mean, rice_var  # noqa: E402
from functions.report import FAIL, WARN, Report  # noqa: E402
from functions.utils import check_rice_noise, compute_mask, estimate_sigma, sigma_per_echo  # noqa: E402

# ── Réglages ─────────────────────────────────────────────────────────────────

SIGMA_DEFAULT = 513.0                          # bruit du grain de blé
TE_DEFAULT = 2.9172 * np.arange(1, 33)         # 32 échos, comme le grain de blé
COUPLES = [(12, 36), (12, 60), (12, 120), (25, 50), (8, 40), (20, 100)]   # (T2c, T2l), f = 0,5
COUPLES_F = [(12, 60, 0.3), (12, 60, 0.7)]                                 # (T2c, T2l, f)
SEPARABLE_CV = 0.25     # un couple est « séparable » si la borne de Cramér-Rao donne moins de 25 % sur T2c et T2l

MODES = {
    "full": dict(t2s=(12, 30, 60, 120), snrs=(10, 20, 30, 50, 100, 150), n_mono=300,
                 bi_snrs=(150, 30), n_bi=200,
                 heat_ratios=(1.5, 2, 3, 4, 6, 8), heat_snrs=(10, 15, 20, 30, 50, 80, 150), n_heat=80,
                 zone_sizes=(1, 4, 16, 64, 256), n_zone=100,
                 sigma_factors=(0.8, 0.9, 1.0, 1.1, 1.2), n_sigma=250, n_echo=200, n_echo_zone=100,
                 sample_per_disk=None, real_voxels=None),
    "quick": dict(t2s=(12, 30, 120), snrs=(20, 30, 100, 150), n_mono=60,
                  bi_snrs=(150, 30), n_bi=40,
                  heat_ratios=(2, 4, 8), heat_snrs=(15, 30, 80), n_heat=30,
                  zone_sizes=(1, 16, 64), n_zone=30,
                  sigma_factors=(0.8, 0.9, 1.0, 1.1, 1.2), n_sigma=60, n_echo=60, n_echo_zone=30,
                  sample_per_disk=80, real_voxels=600),
}

C_SPHENE, C_LS, C_OFF, C_TRUE = "#4C72B0", "#DD8452", "#55A868", "black"


# ── Petits outils ────────────────────────────────────────────────────────────

def fr(x, nd=1, sign=False):
    """Nombre à la française : virgule décimale, vrai signe moins, pas de « -0,0 »."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("+-") if not sign else "+" + s.lstrip("+-")
    return s.replace(".", ",").replace("-", "−")


def pc(x, nd=1, sign=False):
    """Pourcentage à la française (``x`` déjà en %)."""
    return fr(x, nd, sign) + " %"


def big(x):
    """Entier avec espace comme séparateur de milliers (58305 -> « 58 305 »)."""
    return f"{x:,.0f}".replace(",", " ").replace("-", "−")


def rate_ok(k, n, p_max, alpha=1e-3):
    """Un nombre ``k`` d'événements sur ``n`` essais est-il compatible avec un taux vrai d'au plus ``p_max`` ?

    Test binomial exact, unilatéral : ne répond « non » que si voir ``k`` événements ou plus aurait moins d'une chance
    sur mille (``alpha``) d'arriver quand le taux vrai vaut ``p_max``. Le contrôle reste juste quel que soit le nombre
    d'essais : peu d'essais (mode rapide) donnent un test indulgent, beaucoup d'essais un test sévère. Sans cela,
    un contrôle pourrait échouer par simple malchance du tirage au sort.
    """
    return bool(binom.sf(k - 1, n, p_max) >= alpha)


def rate_txt(k, n):
    """« 1,2 % (9 sur 720) »."""
    return f"{pc(100 * k / n, 1)} ({big(k)} sur {big(n)})"


def fr_label(text):
    """Étiquette de jeu simulé à la française (``f=0.5`` -> ``f=0,5``, ``20%`` -> ``20 %``)."""
    return re.sub(r"(\d)%", r"\1 %", re.sub(r"(\d)\.(\d)", r"\1,\2", text))


def plain_log_ticks(ax, axis="y"):
    """Axe logarithmique avec des nombres simples (1, 2, 5, 10...) et la virgule décimale."""
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    ticks = [m * 10.0**k for k in range(-3, 5) for m in (1, 2, 5) if lo <= m * 10.0**k <= hi]
    if len(ticks) < 3:
        ticks = [m * 10.0**k for k in range(-3, 5) for m in (1, 2, 3, 4, 5, 7) if lo <= m * 10.0**k <= hi]
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    target.set_minor_formatter(NullFormatter())


def rel_stats(est, truth):
    """Biais moyen, dispersion et erreur-type du biais, en % de la vérité."""
    est = est[np.isfinite(est)]
    sd = 100 * est.std(ddof=1) / truth
    return {"n": est.size, "bias": 100 * (est.mean() / truth - 1), "sd": sd, "se": sd / math.sqrt(est.size)}


def robust(est, truth):
    """Médiane, quartiles et dispersion robuste (intervalle interquartile), en % de la vérité."""
    est = est[np.isfinite(est)]
    q1, med, q3 = np.percentile(est, [25, 50, 75])
    return {"med": med, "q1": q1, "q3": q3, "err": 100 * (med / truth - 1),
            "sd": 100 * (q3 - q1) / 1.349 / truth}


def silent(func, *args, **kwargs):
    """Appelle ``func`` sans afficher ce qu'elle imprime."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def snr_note():
    return ("SNR = signal au temps d'écho 0 divisé par le bruit. Le chiffre « SNR » affiché par le viewer "
            "(signal du 1er écho / bruit) est un peu plus bas ; pour le grain de blé on est autour de 25 à 30.")


# ── 1. Le bruit ──────────────────────────────────────────────────────────────

def section_noise(rep, ctx):
    sec = rep.section(
        "Le bruit d'une image IRM",
        "Pourquoi ne peut-on pas simplement ajuster une exponentielle sur les mesures ?")
    te, sigma, rng = ctx.te, ctx.sigma, np.random.default_rng(ctx.seed)
    sec.text("Une image IRM est une image de <b>modules</b> : le programme voit la longueur d'un vecteur, "
             "jamais un nombre avec un signe. Le bruit ne peut donc pas faire descendre une mesure sous zéro. "
             "Quand le vrai signal devient petit (échos tardifs), les mesures ne tombent pas à zéro : elles "
             "restent autour d'un <b>plancher</b> positif, que l'on peut prédire (loi de Rice).")

    n = 1_000_000
    noise = sim.rician(np.zeros(n), sigma, rng)
    floor, med = noise.mean() / sigma, np.median(noise) / sigma
    rows, err_mean, err_var, err_mle = [], 0.0, 0.0, 0.0
    for r in (0.0, 0.5, 1, 2, 4, 10):
        draws = sim.rician(np.full(1_000_000, r * sigma), sigma, rng)
        m_sim, v_sim = draws.mean(), draws.var()
        m_th, v_th = float(rice_mean(r * sigma, sigma)), float(rice_var(r * sigma, sigma))
        err_mean = max(err_mean, abs(m_th / m_sim - 1))
        err_var = max(err_var, abs(v_th / v_sim - 1))
        rows.append([fr(r, 1), fr(m_sim / sigma, 4), fr(m_th / sigma, 4), pc(100 * (m_th / m_sim - 1), 2, True),
                     fr(v_sim / sigma**2, 4), fr(v_th / sigma**2, 4), pc(100 * (v_th / v_sim - 1), 2, True)])
    for r in (0.5, 1, 3):
        draws = sim.rician(np.full(100_000, r * sigma), sigma, rng)
        best = minimize_scalar(lambda nu: -rice_loglik(draws, nu, sigma), bounds=(0, 8 * sigma),
                               method="bounded", options={"xatol": 1e-4 * sigma})
        err_mle = max(err_mle, abs(best.x - r * sigma) / sigma)

    sec.text(f"<b>Test.</b> On simule {big(n)} mesures sans aucun signal, avec le niveau de bruit σ = {sigma:.0f} "
             f"de vos images. La moyenne mesurée vaut <b>{fr(floor, 4)} σ</b> (la théorie donne σ√(π/2) = 1,2533 σ) "
             f"et la médiane {fr(med, 4)} σ (théorie 1,1774 σ). Le tableau compare ensuite les formules utilisées par "
             "le programme (moyenne et variance d'une mesure en module) à des simulations, pour un signal vrai ν "
             "de plus en plus grand devant σ.")
    sec.table(["ν / σ", "moyenne simulée / σ", "formule / σ", "écart", "variance simulée / σ²", "formule / σ²", "écart"],
              rows)

    # Ce qu'un ajustement classique trouve, même sans aucun bruit aléatoire
    t2_demo, snr_demo = 12.0, 30
    truth = {"kind": "mono", "T2": t2_demo, "snr": snr_demo}
    exact = sim.true_curve(te, truth, sigma)
    expected = rice_mean(exact, sigma)
    i0_ls, t2_ls = sim.ls_mono(te, expected)
    ctx.R["ls_asym_bias"] = 100 * (t2_ls / t2_demo - 1)
    ctx.R["ls_asym_t2"] = t2_demo
    sec.text(f"<b>Conséquence.</b> Prenons un tissu T2 = {t2_demo:.0f} ms au SNR {snr_demo}. Sa courbe <i>moyenne</i> "
             "mesurée (sans aucun bruit aléatoire, juste l'effet du plancher) ne descend jamais sous le plancher : elle se "
             "redresse aux échos tardifs. "
             f"Un ajustement classique (moindres carrés) en tire T2 = <b>{fr(t2_ls, 2)} ms</b> au lieu de "
             f"{t2_demo:.0f} ms : un biais de {pc(ctx.R['ls_asym_bias'], 1, True)} qu'aucune répétition ne fera disparaître. "
             "C'est ce que corrige la vraisemblance de Rice (le modèle sait que le plancher existe).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.hist(noise[noise < 5 * sigma], bins=120, density=True, color="steelblue", label="mesures simulées (aucun signal)")
    grid = np.linspace(0, 5 * sigma, 300)
    ax1.plot(grid, grid / sigma**2 * np.exp(-grid**2 / (2 * sigma**2)), color="darkorange", lw=2, label="loi de Rice (théorie)")
    ax1.axvline(noise.mean(), color="k", ls="--", label=f"moyenne = {fr(floor, 3)} σ")
    ax1.set_xlabel("valeur mesurée"), ax1.set_ylabel("densité")
    ax1.set_title("Sans aucun signal, la mesure n'est pas nulle")
    ax1.legend(fontsize=8)
    noisy = sim.rician(exact, sigma, rng)
    ax2.semilogy(te, noisy, "o", ms=3.5, color="0.5", label="une mesure simulée")
    ax2.semilogy(te, exact, "--", color=C_TRUE, label=f"signal vrai (T2 = {t2_demo:.0f} ms)")
    ax2.semilogy(te, expected, color=C_SPHENE, lw=2, label="ce qu'on mesure en moyenne")
    ax2.semilogy(te, sim.signal_mono(te, i0_ls, t2_ls), color=C_LS, lw=1.6,
                 label=f"ajustement classique (T2 = {fr(t2_ls, 1)} ms)")
    ax2.axhline(sigma * math.sqrt(math.pi / 2), color="darkorange", ls=":", label="plancher 1,25 σ")
    ax2.set_ylim(0.2 * sigma, 1.5 * snr_demo * sigma)
    ax2.set_xlabel("temps d'écho (ms)"), ax2.set_ylabel("signal (échelle logarithmique)")
    ax2.set_title("Le signal mesuré s'arrête sur le plancher")
    ax2.legend(fontsize=8)
    plain_log_ticks(ax2, "y")
    sec.figure(fig, "01_bruit_plancher.png", "À gauche : le bruit pur n'est pas centré sur zéro. À droite (échelle "
               "logarithmique) : le signal vrai (tirets) continue de descendre, la mesure moyenne (bleu) s'arrête sur le "
               "plancher ; une droite ajustée par moindres carrés (orange) est tirée vers le haut.")

    sec.check("Plancher du bruit pur (moyenne / σ)", fr(floor, 4), "1,2533 à 0,5 % près", abs(floor / 1.2533141 - 1) < 0.005)
    sec.check("Moyenne d'une mesure en module : formule du programme contre simulation", pc(100 * err_mean, 2),
              "écart maximal 1 %", err_mean < 0.01)
    sec.check("Variance d'une mesure en module : formule contre simulation", pc(100 * err_var, 2),
              "écart maximal 3 %", err_var < 0.03)
    sec.check("La vraisemblance de Rice retrouve le signal vrai (ν = 0,5 σ, 1 σ, 3 σ)", f"{fr(err_mle, 3)} σ au plus",
              "écart maximal 0,02 σ", err_mle < 0.02)
    sec.info(f"Biais d'un ajustement classique sans aucun bruit aléatoire (T2 = {t2_demo:.0f} ms, SNR {snr_demo})",
             pc(ctx.R["ls_asym_bias"], 1, True))
    sec.conclusion("Le plancher de bruit existe et vaut bien 1,25 σ ; les formules du programme sont justes. "
                   "Un ajustement classique est biaisé par construction quand le signal approche du plancher.")


# ── 2. T2 mono ───────────────────────────────────────────────────────────────

def run_mono_trials(ctx):
    out = {}
    for i, (t2, snr) in enumerate(itertools.product(ctx.cfg["t2s"], ctx.cfg["snrs"])):
        truth = {"kind": "mono", "T2": float(t2), "snr": snr}
        out[(t2, snr)] = sim.monte_carlo(sim.trial, ctx.cfg["n_mono"], seed=ctx.seed + 1000 * i,
                                         te=ctx.te, truth=truth, sigma=ctx.sigma, compare=True)
    return out


def section_mono(rep, ctx):
    cfg = ctx.cfg
    sec = rep.section(
        "T2 mono : biais et dispersion",
        "Quand le tissu n'a qu'un seul T2, retrouve-t-on la bonne valeur ? Et que gagne-t-on par rapport "
        "aux moindres carrés classiques ?")
    n = cfg["n_mono"]
    sec.text(f"<b>Test.</b> Pour chaque T2 vrai et chaque SNR, on simule {n} voxels et on ajuste un T2 par trois "
             "méthodes : <b>(A) la méthode de Sphene</b> (vraisemblance de Rice), (B) les moindres carrés classiques, "
             "(C) les moindres carrés avec une constante libre (l'ancienne option « offset »). "
             "L'<b>erreur typique</b> est la médiane des erreurs, en % de la vraie valeur : elle devrait être nulle, et quelques "
             "résultats aberrants ne la faussent pas. Le <b>biais moyen</b> est la moyenne des erreurs. La <b>dispersion</b> est "
             "l'écart-type des résultats : plus elle est petite, plus la mesure est précise. La <b>limite théorique</b> "
             "(borne de Cramér-Rao) est la plus petite dispersion qu'AUCUNE méthode ne peut battre avec ce bruit. " + snr_note())

    rows, res = [], {}
    worst = {"med": 0.0, "med_ok": True, "bias": 0.0, "bias_ok": True, "ratio": 0.0, "ratio_ok": True,
             "ls": -1e9, "off": 1e9}
    for (t2, snr), tr in ctx.mono.items():
        a = rel_stats(tr["mono_T2"], t2)
        ma, mb, mc = (robust(tr[key], t2) for key in ("mono_T2", "ls", "ls_off"))
        bound = 100 * sim.crlb(ctx.te, {"kind": "mono", "T2": float(t2), "snr": snr}, ctx.sigma)["T2"]
        ratio = a["sd"] / bound
        res[(t2, snr)] = (ma, mb, mc, a, bound)
        tol_ratio = 1 + 4 / math.sqrt(2 * a["n"]) + 0.05
        # Erreur typique : erreur-type d'une médiane = 1,25 sigma / racine(n). Biais moyen : on admet l'effet attendu
        # « inverse d'une vitesse » (dispersion relative au carré), voir le texte sous le tableau.
        tol_med = max(1.0, 4 * 1.2533 * ma["sd"] / math.sqrt(a["n"]))
        tol_bias = max(1.0, a["sd"] ** 2 / 100 + 4 * a["se"])
        worst["med_ok"] &= abs(ma["err"]) <= tol_med
        worst["bias_ok"] &= abs(a["bias"]) <= tol_bias
        worst["ratio_ok"] &= ratio <= tol_ratio
        worst["med"] = max(worst["med"], abs(ma["err"]))
        worst["bias"] = max(worst["bias"], abs(a["bias"]))
        worst["ratio"] = max(worst["ratio"], ratio)
        worst["ls"] = max(worst["ls"], mb["err"])
        worst["off"] = min(worst["off"], mc["err"])
        rows.append([f"{t2:g}", f"{snr:g}", pc(ma["err"], 2, True), pc(mb["err"], 2, True), pc(mc["err"], 2, True),
                     f"{pc(a['bias'], 2, True)} ± {fr(2 * a['se'], 2)}", pc(a["sd"], 2), pc(bound, 2), fr(ratio, 2)])
    sec.table(["T2 vrai (ms)", "SNR", "erreur typique A (Sphene)", "erreur typique B (moindres carrés)",
               "erreur typique C (+ constante)", "biais moyen de A ± 2 erreurs-types", "dispersion A", "limite théorique",
               "A / limite"], rows)
    snr_low = min(s for (_, s) in res)
    low = [v for (_, s), v in res.items() if s == snr_low]
    b_lo, b_hi = min(v[3]["bias"] for v in low), max(v[3]["bias"] for v in low)
    m_lo, m_hi = min(v[0]["err"] for v in low), max(v[0]["err"] for v in low)
    sec.text("Le « ± » est l'incertitude due au nombre fini de tirages : un biais plus petit que ce « ± » est compatible avec "
             "zéro. Aux SNR bas, les erreurs de B et C sont, elles, bien au-delà de toute incertitude. "
             f"<b>Un détail honnête :</b> au SNR le plus bas de la table (SNR {snr_low:g}), le biais moyen de A vaut de "
             f"{pc(b_lo, 1, True)} à {pc(b_hi, 1, True)} selon le T2, et son erreur typique de {pc(m_lo, 1, True)} à "
             f"{pc(m_hi, 1, True)}. Quand ces petites valeurs positives apparaissent, c'est une propriété de toute mesure de T2 "
             "bruitée, pas un défaut de la méthode : un T2 est l'inverse d'une vitesse de décroissance, et quand cette vitesse est "
             "connue avec une erreur relative de 14 %, son inverse est en moyenne trop grand d'environ 0,14 × 0,14 ≈ 2 % (le "
             "carré de l'erreur relative). L'effet disparaît quand le SNR monte ; les contrôles ci-dessous l'autorisent "
             "explicitement.")
    ctx.R.update(mono_med=worst["med"], mono_bias=worst["bias"], mono_ratio=worst["ratio"], ls_bias=worst["ls"],
                 off_bias=worst["off"])

    t2_show = [cfg["t2s"][0], cfg["t2s"][len(cfg["t2s"]) // 2], cfg["t2s"][-1]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for j, t2 in enumerate(t2_show):
        snrs = [s for (t, s) in res if t == t2]
        for k, (label, color) in enumerate([("A  Sphene (Rice)", C_SPHENE), ("B  moindres carrés", C_LS), ("C  + constante libre", C_OFF)]):
            axes[0, j].plot(snrs, [res[(t2, s)][k]["err"] for s in snrs], "o-", color=color, label=label)
        axes[0, j].axhline(0, color="k", lw=0.8)
        axes[0, j].set_title(f"T2 vrai = {t2:g} ms")
        axes[0, j].set_ylabel("erreur typique (%)")
        axes[1, j].plot(snrs, [res[(t2, s)][3]["sd"] for s in snrs], "o-", color=C_SPHENE, label="dispersion de Sphene")
        axes[1, j].plot(snrs, [res[(t2, s)][4] for s in snrs], "k--", label="limite théorique")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel("SNR")
        axes[1, j].set_ylabel("dispersion (%)")
        plain_log_ticks(axes[1, j], "x")
        plain_log_ticks(axes[1, j], "y")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("T2 mono : la méthode de Sphene est juste et atteint la limite théorique", fontsize=12)
    fig.tight_layout()
    sec.figure(fig, "02_mono_biais_dispersion.png", "En haut : erreur typique (médiane) selon le SNR, le but est 0. En bas : "
               "dispersion de Sphene (bleu) contre la limite théorique (pointillés) : elles se confondent.")

    sec.check("Erreur typique (médiane) de la méthode de Sphene", f"{pc(worst['med'], 2)} au plus",
              "moins de 1 % (ou 4 erreurs-types de la médiane quand la dispersion est grande)", worst["med_ok"])
    sec.check("Biais moyen de la méthode de Sphene", f"{pc(worst['bias'], 2)} au plus",
              "moins de 1 % (élargi à « dispersion au carré » + 4 erreurs-types quand la dispersion est grande)", worst["bias_ok"])
    sec.check("Dispersion de Sphene / limite théorique", f"{fr(worst['ratio'], 2)} au plus",
              "proche de 1 (marge statistique : 4 erreurs-types + 5 %)", worst["ratio_ok"])
    sec.info("Plus forte erreur typique positive des moindres carrés classiques", pc(worst["ls"], 1, True))
    sec.info("Plus forte erreur typique négative des moindres carrés avec constante libre", pc(worst["off"], 1, True))
    sec.conclusion(f"La méthode de Sphene retrouve le bon T2 : erreur typique {pc(worst['med'], 1)} au plus, biais moyen "
                   f"{pc(worst['bias'], 1)} au plus (aux SNR les plus bas, un léger biais positif est attendu, voir plus haut), et une dispersion "
                   f"aussi petite que la physique le permet. Les moindres carrés classiques surestiment T2 jusqu'à "
                   f"{pc(worst['ls'], 0, True)} quand le signal est faible ; avec une constante libre ils le sous-estiment "
                   f"jusqu'à {pc(worst['off'], 0, True)} : la constante se compense avec T2 (surtout quand T2 est long devant "
                   "la durée de la série d'échos) et décrit mal un plancher qui s'efface quand le signal est fort. C'est "
                   "pourquoi ces deux variantes (et l'ancien « mono avec C fixé ») ont été supprimées : une seule méthode, "
                   "juste, les remplace.")


# ── 3. T2 bi ─────────────────────────────────────────────────────────────────

def bi_truths():
    return [(c, l, 0.5) for c, l in COUPLES] + COUPLES_F


def run_bi_trials(ctx):
    out = {}
    for s, snr in enumerate(ctx.cfg["bi_snrs"]):
        for i, (t2c, t2l, f) in enumerate(bi_truths()):
            truth = {"kind": "bi", "f": f, "T2c": float(t2c), "T2l": float(t2l), "snr": snr}
            out[(t2c, t2l, f, snr)] = sim.monte_carlo(
                sim.trial, ctx.cfg["n_bi"], seed=ctx.seed + 50_000 + 1000 * (10 * s + i),
                te=ctx.te, truth=truth, sigma=ctx.sigma)
    return out


def section_bi(rep, ctx):
    cfg = ctx.cfg
    sec = rep.section(
        "T2 bi : précision de T2c, T2l et f",
        "Quand le tissu a deux composantes (une courte T2c, une longue T2l, dans la proportion f), "
        "retrouve-t-on les trois valeurs ? Jusqu'à quel point ?")
    sec.text(f"<b>Test.</b> Huit tissus à deux composantes, chacun simulé {cfg['n_bi']} fois à deux niveaux de bruit "
             f"(SNR {cfg['bi_snrs'][0]} : peu de bruit ; SNR {cfg['bi_snrs'][1]} : proche du grain de blé). "
             "On ajuste à chaque fois le modèle bi. Le tableau donne la valeur <b>médiane</b> retrouvée et l'intervalle qui contient "
             "la moitié des résultats [Q1 – Q3], à comparer à la vérité. La dernière colonne est la précision que "
             "la physique autorise au mieux (limite de Cramér-Rao). Un couple est dit <b>séparable</b> quand cette limite "
             f"donne moins de {pc(100 * SEPARABLE_CV, 0)} d'erreur sur T2c et sur T2l. " + snr_note())

    # Tolérances : valeur fixe, élargie si les tirages sont peu nombreux (la médiane de n tirages a une
    # erreur-type d'environ 1,25 sigma / racine(n) ; la dispersion robuste, d'environ 1,4 / racine(n) en relatif).
    rows, bad_err, bad_ratio, n_sep = [], {"T2c": 0.0, "T2l": 0.0, "f": 0.0}, 0.0, 0
    ok = {"T2c": True, "T2l": True, "f": True, "ratio": True}
    for (t2c, t2l, f, snr), tr in ctx.bi.items():
        bound = sim.crlb(ctx.te, {"kind": "bi", "f": f, "T2c": float(t2c), "T2l": float(t2l), "snr": snr}, ctx.sigma)
        sep = max(bound["T2c"], bound["T2l"]) <= SEPARABLE_CV
        rc, rl = robust(tr["bi_T2c"], t2c), robust(tr["bi_T2l"], t2l)
        ff = tr["bi_f"][np.isfinite(tr["bi_f"])]
        n_ok = ff.size
        f_med, f_q = np.median(ff), np.percentile(ff, [25, 75])
        rows.append([f"{t2c:g} / {t2l:g}  (f = {fr(f, 1)})", f"{snr:g}",
                     f"{fr(rc['med'], 1)} [{fr(rc['q1'], 1)} – {fr(rc['q3'], 1)}]",
                     f"{fr(rl['med'], 1)} [{fr(rl['q1'], 1)} – {fr(rl['q3'], 1)}]",
                     f"{fr(f_med, 2)} [{fr(f_q[0], 2)} – {fr(f_q[1], 2)}]",
                     f"{pc(100 * bound['T2c'], 0)} / {pc(100 * bound['T2l'], 0)}"
                     + ("" if sep else "  <i>(non séparable)</i>")])
        if snr == cfg["bi_snrs"][0] and sep:
            n_sep += 1
            for key, r in (("T2c", rc), ("T2l", rl)):
                bad_err[key] = max(bad_err[key], abs(r["err"]))
                ok[key] &= abs(r["err"]) <= max(5.0, 4 * 1.25 * r["sd"] / math.sqrt(n_ok))
            f_sd = (f_q[1] - f_q[0]) / 1.349
            bad_err["f"] = max(bad_err["f"], abs(f_med - f))
            ok["f"] &= abs(f_med - f) <= max(0.05, 4 * 1.25 * f_sd / math.sqrt(n_ok))
            ratio = max(rc["sd"] / (100 * bound["T2c"]), rl["sd"] / (100 * bound["T2l"]))
            bad_ratio = max(bad_ratio, ratio)
            tol_ratio = 1.15 + 4 * 1.4 / math.sqrt(n_ok)
            ok["ratio"] &= ratio <= tol_ratio
    sec.table(["vérité T2c / T2l (ms)", "SNR", "T2c retrouvé (ms)", "T2l retrouvé (ms)", "f retrouvé",
               "précision possible (T2c / T2l)"], rows)
    ctx.R.update(bi_err=bad_err, bi_ratio=bad_ratio)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    for ax, (t2c, t2l) in zip(axes.ravel(), COUPLES):
        for snr, color in zip(cfg["bi_snrs"], (C_SPHENE, C_LS)):
            tr = ctx.bi[(t2c, t2l, 0.5, snr)]
            ax.plot(tr["bi_T2c"], tr["bi_T2l"], ".", ms=5, alpha=0.5, color=color, label=f"SNR {snr:g}")
        ax.plot([t2c], [t2l], "*", ms=16, color="k", label="vérité")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel("T2c retrouvé (ms)"), ax.set_ylabel("T2l retrouvé (ms)")
        ax.set_title(f"vérité {t2c:g} / {t2l:g} ms")
        ax.legend(fontsize=8)
    for ax in axes.ravel():
        plain_log_ticks(ax, "x")
        plain_log_ticks(ax, "y")
    fig.suptitle("T2 bi : chaque point est un voxel simulé. Les nuages se resserrent autour de l'étoile quand le bruit baisse",
                 fontsize=12)
    fig.tight_layout()
    sec.figure(fig, "03_bi_nuages.png", "Chaque point = un voxel simulé. Bleu : peu de bruit ; orange : bruit du grain. "
               "L'étendue du nuage orange est imposée par la physique, pas par le programme. Les points alignés en haut sont "
               "des voxels où T2l a atteint la limite de la plage de recherche.")

    margin = "ou 4 erreurs-types de la médiane si les tirages sont peu nombreux"
    sec.check(f"Écart médian sur T2c (SNR {cfg['bi_snrs'][0]}, couples séparables, {n_sep} cas)", pc(bad_err["T2c"], 1) + " au plus",
              f"moins de 5 % ({margin})", ok["T2c"])
    sec.check(f"Écart médian sur T2l (SNR {cfg['bi_snrs'][0]}, couples séparables)", pc(bad_err["T2l"], 1) + " au plus",
              f"moins de 5 % ({margin})", ok["T2l"])
    sec.check(f"Écart médian sur f (SNR {cfg['bi_snrs'][0]}, couples séparables)", fr(bad_err["f"], 3) + " au plus",
              f"moins de 0,05 ({margin})", ok["f"])
    sec.check("Dispersion robuste / limite théorique (T2c et T2l, SNR élevé)", fr(bad_ratio, 2) + " au plus",
              "moins de 1,15 + marge statistique (4 erreurs-types)", ok["ratio"])
    sec.conclusion("Avec peu de bruit, T2c, T2l et f sont retrouvés à quelques % près pour tous les couples séparables. "
                   "Au SNR du grain, la dispersion est grande, et le tableau montre qu'elle est imposée par la physique "
                   "(colonne de droite) : un couple peu séparé (par exemple 25 / 50 ms) ne peut pas être résolu voxel par voxel. "
                   "Les zones regroupant beaucoup de voxels (section 5) lèvent en partie ce problème.")


# ── 4. Choix mono / bi ───────────────────────────────────────────────────────

def run_heatmap_trials(ctx):
    cfg, out = ctx.cfg, {}
    for i, (ratio, snr) in enumerate(itertools.product(cfg["heat_ratios"], cfg["heat_snrs"])):
        truth = {"kind": "bi", "f": 0.5, "T2c": 12.0, "T2l": 12.0 * ratio, "snr": snr}
        tr = sim.monte_carlo(sim.trial, cfg["n_heat"], seed=ctx.seed + 200_000 + 1000 * i,
                             te=ctx.te, truth=truth, sigma=ctx.sigma)
        out[(ratio, snr)] = float(np.mean(tr["bi_kept"]))
    return out


def section_choice(rep, ctx):
    cfg = ctx.cfg
    sec = rep.section(
        "Choix mono / bi",
        "Le programme n'affiche le bi que là où les données le justifient. Se trompe-t-il quand le tissu est mono ? "
        "Voit-il vraiment deux composantes quand elles existent ?")
    sec.text("Le modèle bi a deux paramètres de plus : il colle toujours mieux aux données, même quand il n'y a rien de plus "
             "à expliquer que du bruit. Le programme compare donc les deux modèles avec le critère <b>AICc</b> : le bi est retenu "
             f"seulement si son gain de vraisemblance dépasse le « prix » de ses paramètres en plus, avec une marge de {DELTA_AICC:g} "
             "(ΔAICc &gt; 2). Deux questions : quand le tissu est mono, combien de <b>fausses alarmes</b> (bi retenu à tort) ? "
             "Quand le tissu est bi, combien de <b>détections</b> ?")

    # (a) fausses alarmes, sur les mêmes essais que la section 2
    fa = {k: 100 * float(np.mean(tr["bi_kept"])) for k, tr in ctx.mono.items()}
    n_tot = sum(len(tr["bi_kept"]) for tr in ctx.mono.values())
    pooled = 100 * sum(float(np.sum(tr["bi_kept"])) for tr in ctx.mono.values()) / n_tot
    worst_fa = max(fa.values())
    snrs, t2s = cfg["snrs"], cfg["t2s"]
    sec.text(f"<b>Fausses alarmes</b> (tissu simulé mono, {big(n_tot)} voxels au total) : en moyenne <b>{pc(pooled, 2)}</b> "
             f"des voxels sont classés bi à tort ; le pire des cas vaut {pc(worst_fa, 1)}.")
    sec.table(["T2 vrai (ms) \\ SNR"] + [f"{s:g}" for s in snrs],
              [[f"{t:g}"] + [pc(fa[(t, s)], 1) for s in snrs] for t in t2s])

    # (b) détection, sur les essais de la section 3
    rows, min_power = [], 100.0
    for (t2c, t2l, f, snr), tr in ctx.bi.items():
        power = 100 * float(np.mean(tr["bi_kept"]))
        ratio = t2l / t2c
        if snr == cfg["bi_snrs"][0] and ratio >= 3:
            min_power = min(min_power, power)
        rows.append([f"{t2c:g} / {t2l:g}  (f = {fr(f, 1)})", fr(ratio, 1), f"{snr:g}", pc(power, 0)])
    sec.text("<b>Détection</b> (tissu simulé bi) : part des voxels où le bi est retenu.")
    sec.table(["vérité T2c / T2l (ms)", "rapport T2l / T2c", "SNR", "bi retenu"], rows)

    # (c) carte de détection
    heat = ctx.heat
    ratios, hsnrs = cfg["heat_ratios"], cfg["heat_snrs"]
    mat = np.array([[100 * heat[(r, s)] for r in ratios] for s in hsnrs])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    all_dm = np.concatenate([tr["delta_aicc"][np.isfinite(tr["delta_aicc"])]
                             for k, tr in ctx.mono.items() if k[1] == 30] or [np.array([0.0])])
    bi_dm = ctx.bi.get((12, 60, 0.5, 30), {}).get("delta_aicc", np.array([0.0]))
    top = float(np.percentile(np.concatenate([all_dm, bi_dm[np.isfinite(bi_dm)]]), 97))
    edges = np.linspace(-6, max(top, 12), 50)
    ax1.hist(np.clip(all_dm, -6, edges[-1]), bins=edges, alpha=0.6, color=C_SPHENE, density=True,
             label="tissu mono (SNR 30)")
    ax1.hist(np.clip(bi_dm[np.isfinite(bi_dm)], -6, edges[-1]), bins=edges, alpha=0.6, color=C_LS, density=True,
             label="tissu bi 12 / 60 ms (SNR 30)")
    ax1.axvline(DELTA_AICC, color="k", ls="--", label=f"seuil ΔAICc = {DELTA_AICC:g}")
    ax1.set_xlabel("ΔAICc = AICc(mono) − AICc(bi)   (grand = le bi explique mieux)")
    ax1.set_ylabel("densité")
    ax1.set_title("Le critère sépare les deux situations")
    ax1.legend(fontsize=8)
    im = ax2.imshow(mat, origin="lower", cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax2.set_xticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    ax2.set_yticks(range(len(hsnrs)), [f"{s:g}" for s in hsnrs])
    for i, j in itertools.product(range(len(hsnrs)), range(len(ratios))):
        ax2.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                 color="white" if mat[i, j] < 25 or mat[i, j] > 85 else "black")
    if 4 in ratios:
        ax2.axvline(list(ratios).index(4) + 0.5, color="k", ls="--", lw=1)
        ax2.text(list(ratios).index(4) + 0.55, len(hsnrs) - 0.6, "règle « 4 × T2c < T2l »", fontsize=8, va="top")
    ax2.set_xlabel("rapport T2l / T2c   (T2c = 12 ms, f = 0,5)")
    ax2.set_ylabel("SNR")
    ax2.set_title("Part des voxels où le bi est retenu (%)")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    sec.figure(fig, "04_choix_mono_bi.png", "À gauche : le critère ΔAICc pour un tissu mono (bleu) et un tissu bi (orange). "
               "À droite : probabilité de détecter le bi selon le rapport T2l/T2c et le SNR.")

    # seuil de rapport à 50 % pour chaque SNR de la carte
    thr = []
    for s in hsnrs:
        ok = [r for r in ratios if 100 * heat[(r, s)] >= 50]
        thr.append((s, min(ok) if ok else None))
    thr_txt = ", ".join(f"SNR {s:g} : {('rapport ≥ ' + f'{r:g}'.replace('.', ',')) if r else 'jamais dans la grille testée'}"
                        for s, r in thr)
    ctx.R["choice_thr"] = thr
    sec.text("<b>Lire la figure.</b> À gauche, le pic autour de −5 (bleu) regroupe les voxels où le modèle bi n'a rien trouvé de "
             "plus que le mono : il retombe sur lui et « paie » ses deux paramètres inutiles ; le seuil ΔAICc = 2 est loin à "
             "droite de ce pic, d'où peu de fausses alarmes. Pour un vrai tissu bi (orange), le gain est presque toujours bien "
             "au-dessus du seuil. À droite, la couleur donne la part de voxels où le bi est retenu : vert = détecté, "
             "rouge = le programme répond « mono ».")
    sec.text(f"<b>À partir de quel rapport T2l / T2c détecte-t-on le bi une fois sur deux ?</b> {thr_txt}. "
             "Il n'existe donc <b>pas de rapport universel</b> (comme la règle « 4 × T2c &lt; T2l » parfois citée) : tout dépend du bruit.")

    # (d) pourquoi le seuil 2 ? Mêmes essais, seuil déplacé : on ne refait aucun ajustement
    thresholds = np.array([-6, -4, -2, 0, 1, 2, 3, 4, 6, 8, 10, 15, 20], dtype=float)
    d_mono = np.concatenate([tr["delta_aicc"][np.isfinite(tr["delta_aicc"])] for tr in ctx.mono.values()])
    fa_curve = np.array([100 * float(np.mean(d_mono > t)) for t in thresholds])
    snr_hi, snr_lo = cfg["bi_snrs"][0], cfg["bi_snrs"][-1]
    cases = [((12, 60, 0.5, snr_hi), C_SPHENE, f"bi 12 / 60 ms, SNR {snr_hi:g}"),
             ((12, 60, 0.5, snr_lo), C_LS, f"bi 12 / 60 ms, SNR {snr_lo:g}"),
             ((12, 36, 0.5, snr_lo), C_OFF, f"bi 12 / 36 ms, SNR {snr_lo:g}, cas limite")]
    power = {}
    for key, _, _ in cases:
        d_bi = ctx.bi[key]["delta_aicc"]
        d_bi = d_bi[np.isfinite(d_bi)]
        power[key] = np.array([100 * float(np.mean(d_bi > t)) for t in thresholds])
    key_edge = cases[-1][0]
    i_sel = int(np.argmin(np.abs(thresholds - DELTA_AICC)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.plot(thresholds, fa_curve, "o-", color="firebrick", label="fausses alarmes (tissu mono)")
    for key, color, label in cases:
        ax1.plot(thresholds, power[key], "o-", color=color, label=f"détection ({label})")
    ax1.axvline(DELTA_AICC, color="k", ls="--", label=f"seuil choisi ({DELTA_AICC:g})")
    ax1.set_xlabel("seuil sur ΔAICc"), ax1.set_ylabel("% de voxels classés bi")
    ax1.set_title("Monter le seuil supprime les fausses alarmes… et la détection")
    ax1.legend(fontsize=8)
    ax2.plot(fa_curve, power[key_edge], "o-", color=C_OFF)
    for t, x, y in zip(thresholds, fa_curve, power[key_edge]):
        if t in (0, 2, 6, 10):
            ax2.annotate(f"seuil {t:g}", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax2.plot(fa_curve[i_sel], power[key_edge][i_sel], "*", ms=16, color="k", label="seuil choisi")
    ax2.set_xlim(-0.3, 8)
    ax2.set_xlabel("fausses alarmes (% de voxels mono classés bi ; axe limité à 8 %)")
    ax2.set_ylabel(f"détection, bi 12 / 36 ms, SNR {snr_lo:g} (%)")
    ax2.set_title("Compromis fausses alarmes / détection (cas limite)")
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    sec.figure(fig, "04b_seuil_aicc.png", "Le seuil ΔAICc = 2. À gauche : part de voxels classés bi selon le seuil (rouge : tissu "
               "mono, c'est une erreur ; autres couleurs : tissu bi, c'est une réussite). À droite : le même compromis en un seul "
               "trait pour le cas limite. En bas à gauche on ne se trompe jamais mais on ne détecte rien ; en haut à droite on "
               "détecte tout mais on se trompe souvent.")
    shown = [t for t in (0, 1, 2, 4, 6, 10)]
    sec.table(["seuil sur ΔAICc", "fausses alarmes (tissu mono)"] + [f"détection ({label})" for _, _, label in cases],
              [[f"{t:g}", pc(fa_curve[k], 2)] + [pc(power[key][k], 0) for key, _, _ in cases]
               for k, t in enumerate(thresholds) if t in shown])
    i_hi, i_zero = int(np.argmin(np.abs(thresholds - 10))), int(np.argmin(np.abs(thresholds)))
    sec.text(f"<b>Pourquoi 2 ?</b> C'est le seuil usuel pour dire qu'un modèle est « nettement » meilleur qu'un autre. Ici il laisse "
             f"{pc(fa_curve[i_sel], 1)} de fausses alarmes. Pour le cas limite (12 / 36 ms, SNR {snr_lo:g}), la détection est de "
             f"{pc(power[key_edge][i_sel], 0)} ; avec un seuil de 0 elle monterait à {pc(power[key_edge][i_zero], 0)} mais les fausses "
             f"alarmes à {pc(fa_curve[i_zero], 1)} ; avec un seuil de {thresholds[i_hi]:g}, les fausses alarmes tomberaient à "
             f"{pc(fa_curve[i_hi], 1)} mais la détection à {pc(power[key_edge][i_hi], 0)}. Le seuil 2 est un compromis entre les deux "
             "erreurs possibles, et ce graphique peut être refait pour votre propre bruit.")
    ctx.R.update(thr_fa=float(fa_curve[i_sel]), thr_power_edge=float(power[key_edge][i_sel]))

    k_tot = int(sum(np.sum(tr["bi_kept"]) for tr in ctx.mono.values()))
    cells = [(int(np.sum(tr["bi_kept"])), len(tr["bi_kept"])) for tr in ctx.mono.values()]
    k_w, n_w = max(cells, key=lambda c: c[0] / c[1])
    sec.check("Fausses alarmes, tous tissus mono confondus", rate_txt(k_tot, n_tot),
              "taux compatible avec « au plus 2,5 % » (test binomial)", rate_ok(k_tot, n_tot, 0.025))
    sec.check("Fausses alarmes, pire cas (un T2 et un SNR)", rate_txt(k_w, n_w),
              "taux compatible avec « au plus 6 % » (test binomial)", all(rate_ok(k, n, 0.06) for k, n in cells))
    sec.check(f"Détection au SNR {cfg['bi_snrs'][0]} pour un rapport T2l/T2c ≥ 3 (pire cas)", pc(min_power, 0),
              "au moins 95 %", min_power >= 95)
    ctx.R.update(false_alarm=pooled, false_alarm_max=worst_fa, power_min=min_power)
    sec.conclusion(f"Sur un tissu mono, le bi est retenu à tort dans {pc(pooled, 1)} des voxels seulement. Sur un tissu bi "
                   "bien séparé et peu bruité il est détecté presque à coup sûr ; quand le bruit monte, il faut un rapport "
                   "T2l/T2c plus grand, et en dessous la physique interdit de le voir : le programme répond alors « mono », "
                   "ce qui est la réponse honnête.")


# ── 5. Zone ──────────────────────────────────────────────────────────────────

def section_zone(rep, ctx):
    cfg = ctx.cfg
    sec = rep.section(
        "« Zone fit »",
        "Dans une zone homogène, que gagne-t-on à ajuster tous les voxels ensemble ? Qu'aurait donné une moyenne ?")
    t2, snr, het = 30.0, 30, 0.25
    sec.text(f"<b>Test.</b> Tissu mono T2 = {t2:.0f} ms au SNR {snr}. On simule des zones de N voxels dont l'amplitude varie de "
             f"±{pc(100 * het, 0)} d'un voxel à l'autre (un tissu n'est jamais parfaitement uniforme). Deux façons d'ajuster : "
             "<b>(A) Sphene</b> : les courbes de tous les voxels sont ajustées <i>ensemble</i>, dans la même vraisemblance, "
             "sans rien moyenner avant ; chaque voxel garde sa propre amplitude, seule la forme de la décroissance (T2) est "
             "commune ; <b>(B)</b> on moyenne les modules des voxels puis on ajuste par moindres carrés "
             "classiques (le plancher de bruit reste dans la moyenne). " + f"{cfg['n_zone']} zones par taille.")

    truth = {"kind": "mono", "T2": t2, "snr": snr}
    stats, rows = {}, []
    for i, n in enumerate(cfg["zone_sizes"]):
        tr = sim.monte_carlo(sim.trial, cfg["n_zone"], seed=ctx.seed + 300_000 + 1000 * i, te=ctx.te, truth=truth,
                             sigma=ctx.sigma, n_vox=n, het=het, compare=True)
        a = rel_stats(tr["mono_T2"], t2)
        b = rel_stats(tr["ls_mean" if n > 1 else "ls"], t2)
        bound = 100 * sim.crlb(ctx.te, truth, ctx.sigma, n_vox=n)["T2"]
        stats[n] = (a, b, bound, 100 * float(np.mean(tr["bi_kept"])), len(tr["bi_kept"]))
        rows.append([f"{n}", pc(a["bias"], 2, True), pc(b["bias"], 2, True), pc(a["sd"], 2), pc(bound, 2),
                     pc(stats[n][3], 1)])
    sec.table(["voxels dans la zone", "biais A (Sphene)", "biais B (moyenne + moindres carrés)", "dispersion A",
               "limite théorique", "bi retenu à tort"], rows)

    sizes = list(cfg["zone_sizes"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].errorbar(sizes, [stats[n][0]["bias"] for n in sizes], yerr=[2 * stats[n][0]["se"] for n in sizes], fmt="o-",
                     color=C_SPHENE, capsize=3, label="A  Sphene (ajustement joint)")
    axes[0].errorbar(sizes, [stats[n][1]["bias"] for n in sizes], yerr=[2 * stats[n][1]["se"] for n in sizes], fmt="s-",
                     color=C_LS, capsize=3, label="B  moyenne + moindres carrés")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("voxels dans la zone"), axes[0].set_ylabel("biais sur T2 (%)")
    axes[0].set_title("Moyenner garde un biais qui ne baisse pas")
    axes[0].legend(fontsize=8)
    axes[1].plot(sizes, [stats[n][0]["sd"] for n in sizes], "o-", color=C_SPHENE, label="dispersion de Sphene")
    axes[1].plot(sizes, [stats[n][2] for n in sizes], "k--", label="limite théorique")
    axes[1].set_xscale("log"), axes[1].set_yscale("log")
    axes[1].set_xlabel("voxels dans la zone"), axes[1].set_ylabel("dispersion (%)")
    axes[1].set_title("La précision suit la limite théorique")
    axes[1].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks(sizes)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    plain_log_ticks(axes[1], "y")
    axes[2].bar(range(len(sizes)), [stats[n][3] for n in sizes], color=C_SPHENE)
    axes[2].axhline(3, color="k", ls="--", label="limite du contrôle (3 %)")
    axes[2].set_xticks(range(len(sizes)), [str(n) for n in sizes])
    axes[2].set_xlabel("voxels dans la zone"), axes[2].set_ylabel("bi retenu à tort (%)")
    axes[2].set_title("Fausses alarmes (zone mono)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    sec.figure(fig, "05_zone.png", "Ajuster les voxels ensemble (bleu) supprime le biais que garde la moyenne des modules "
               "(orange), et la précision s'améliore comme 1/√N.")

    worst_bias_ok = all(abs(a["bias"]) <= max(1.0, 4 * a["se"]) for a, _, _, _, _ in stats.values())
    worst_bias = max(abs(a["bias"]) for a, _, _, _, _ in stats.values())
    worst_ratio = max(a["sd"] / bound for a, _, bound, _, _ in stats.values())
    ratio_ok = all(a["sd"] / bound <= 1 + 4 / math.sqrt(2 * a["n"]) + 0.05 for a, _, bound, _, _ in stats.values())
    multi = [v for n, v in stats.items() if n >= 4]
    n_multi = sum(v[4] for v in multi)
    k_multi = int(round(sum(v[3] / 100 * v[4] for v in multi)))
    pooled = 100 * k_multi / max(n_multi, 1)
    biggest = stats[sizes[-1]]
    sec.check("Biais de l'ajustement joint, toutes tailles de zone", pc(worst_bias, 2) + " au plus",
              "moins de 1 % (ou 4 erreurs-types)", worst_bias_ok)
    sec.check("Dispersion / limite théorique", fr(worst_ratio, 2) + " au plus",
              "proche de 1 (marge statistique : 4 erreurs-types + 5 %)", ratio_ok)
    sec.check("Bi retenu à tort sur une zone mono (zones de 4 voxels ou plus)", rate_txt(k_multi, n_multi),
              "taux compatible avec « au plus 3 % » (test binomial)", rate_ok(k_multi, n_multi, 0.03))
    sec.info(f"Biais de la moyenne + moindres carrés pour {sizes[-1]} voxels", pc(biggest[1]["bias"], 2, True))
    ctx.R.update(zone_bias=worst_bias, zone_ls_bias=biggest[1]["bias"], zone_fa=pooled)
    sec.conclusion(f"Ajuster les voxels ensemble donne un T2 sans biais (au pire {pc(worst_bias, 1)}) même quand l'amplitude "
                   f"varie de ±{pc(100 * het, 0)}, et la précision gagne comme 1/√N. Moyenner d'abord garde un biais de "
                   f"{pc(biggest[1]['bias'], 1, True)} qui ne diminue pas quand on ajoute des voxels : plus on en met, plus "
                   "on est précis… sur une mauvaise valeur.")


# ── 6. Erreur sur le niveau de bruit ─────────────────────────────────────────

def section_sigma(rep, ctx):
    cfg = ctx.cfg
    sec = rep.section(
        "Sensibilité à une erreur sur le niveau de bruit",
        "Le niveau de bruit σ est mesuré dans le fond de l'image. S'il est un peu faux, les résultats sont-ils faussés ?")
    t2 = 30.0
    sec.text(f"<b>Test.</b> On simule un tissu mono T2 = {t2:.0f} ms avec le vrai σ, mais on donne au programme un σ trop petit "
             "ou trop grand (de −20 % à +20 %). On regarde le biais sur T2 et le nombre de fausses alarmes (bi retenu à tort). "
             f"{cfg['n_sigma']} voxels par cas.")
    factors, snrs = cfg["sigma_factors"], (30, 150)
    res = {}
    for k, snr in enumerate(snrs):
        for i, fct in enumerate(factors):
            tr = sim.monte_carlo(sim.trial, cfg["n_sigma"], seed=ctx.seed + 400_000 + 1000 * (10 * k + i), te=ctx.te,
                                 truth={"kind": "mono", "T2": t2, "snr": snr}, sigma=ctx.sigma, sigma_factor=fct)
            kept = tr["bi_kept"]
            res[(snr, fct)] = (rel_stats(tr["mono_T2"], t2), 100 * float(np.mean(kept)), int(np.sum(kept)), len(kept))
    sec.table(["erreur sur σ"] + [f"biais T2 (SNR {s:g})" for s in snrs] + [f"fausses alarmes (SNR {s:g})" for s in snrs],
              [[pc(100 * (f - 1), 0, True)] + [pc(res[(s, f)][0]["bias"], 2, True) for s in snrs]
               + [pc(res[(s, f)][1], 1) for s in snrs] for f in factors])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = [100 * (f - 1) for f in factors]
    for snr, color in zip(snrs, (C_SPHENE, C_LS)):
        ax1.errorbar(x, [res[(snr, f)][0]["bias"] for f in factors], yerr=[2 * res[(snr, f)][0]["se"] for f in factors],
                     fmt="o-", color=color, capsize=3, label=f"SNR {snr:g}")
        ax2.plot(x, [res[(snr, f)][1] for f in factors], "o-", color=color, label=f"SNR {snr:g}")
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xlabel("erreur sur σ (%)"), ax1.set_ylabel("biais sur T2 (%)")
    ax1.set_title("Biais sur T2")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("erreur sur σ (%)"), ax2.set_ylabel("bi retenu à tort (%)")
    ax2.set_title("Fausses alarmes")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    sec.figure(fig, "06_sensibilite_sigma.png", "Effet d'une erreur sur σ : biais sur T2 (gauche) et fausses alarmes (droite).")

    f_lo, f_hi = factors[1], factors[3]
    cells10 = [res[(s, f)] for s in snrs for f in (f_lo, f_hi)]
    bias10 = max(abs(c[0]["bias"]) for c in cells10)
    bias_ok = all(abs(c[0]["bias"]) <= max(3.0, 4 * c[0]["se"]) for c in cells10)
    k10, n10 = max(((c[2], c[3]) for c in cells10), key=lambda kn: kn[0] / kn[1])
    fa10 = 100 * k10 / n10
    sec.check("Biais sur T2 avec σ faux de ±10 %", pc(bias10, 2) + " au plus",
              "moins de 3 % (ou 4 erreurs-types)", bias_ok)
    sec.check("Fausses alarmes avec σ faux de ±10 % (pire cas)", rate_txt(k10, n10),
              "taux compatible avec « au plus 8 % » (test binomial)", all(rate_ok(c[2], c[3], 0.08) for c in cells10))
    ctx.R.update(sigma_bias10=bias10, sigma_fa10=fa10)
    sec.conclusion(f"Une erreur de 10 % sur σ change T2 de {pc(bias10, 1)} au plus et fait passer les fausses alarmes à "
                   f"{pc(fa10, 1)} au plus. La mesure de σ dans le fond de l'image est largement assez précise, à condition "
                   "que le fond soit du bruit de Rice pur (images non filtrées : voir la section sur vos données).")


# ── 7. Premiers échos perturbés ──────────────────────────────────────────────

ECHO_PATTERN = np.array([-1.0, 3.0, -1.0]) / 100        # motif observé : 1er écho trop bas, 2e trop haut, 3e trop bas
ECHO_SHIFTS = {"aucun": 0, "3 %": 1, "6 %": 2, "12 %": 4}  # facteur sur le motif ; le nom = décalage du 2e écho
ACANTHE_SHIFT = np.array([-8.7, 10.5, -0.1]) / 100       # décalage mesuré sur les 3 premiers échos du jeu de test de l'acanthe


def section_echoes(rep, ctx):
    cfg = ctx.cfg
    sec = rep.section(
        "Premiers échos perturbés",
        "Sur de vraies images, les tout premiers échos s'écartent parfois de la décroissance attendue (échos stimulés, "
        "refocalisation imparfaite). Quel effet sur T2 et sur le choix mono / bi ? Vaut-il la peine d'ignorer les premiers échos ?")
    t2, n_skip, snrs = 120.0, 3, (30, 150)
    sec.text(f"<b>Test.</b> Tissu mono T2 = {t2:.0f} ms (c'est avec ce T2 que l'effet est le plus visible : essais préliminaires "
             "à 12, 30 et 120 ms). On décale les <b>3 premiers échos</b> selon le motif observé sur des données réelles : le "
             "2<sup>e</sup> trop haut de 3, 6 ou 12 %, le 1<sup>er</sup> et le 3<sup>e</sup> trop bas du tiers de cette valeur. "
             "Chaque cas est ajusté deux fois, sur <b>les mêmes mesures</b> : avec tous les échos, puis sans les "
             f"{n_skip} premiers (option <code>SKIP_FIRST_ECHOES</code> de <code>main.py</code>). {cfg['n_echo']} voxels par cas. "
             "Ce test ne dit pas si VOS échos sont perturbés (c'est la section « Vos données » qui le mesure) : il dit ce que "
             "cela coûte, et ce que coûte d'ignorer des échos.")
    res, rows = {}, []
    for k, snr in enumerate(snrs):
        for i, (name, factor) in enumerate(ECHO_SHIFTS.items()):
            gain = np.ones(len(ctx.te))
            gain[:3] += factor * ECHO_PATTERN
            for skip in (0, n_skip):
                tr = sim.monte_carlo(sim.trial, cfg["n_echo"], seed=ctx.seed + 500_000 + 1000 * (10 * k + i), te=ctx.te,
                                     truth={"kind": "mono", "T2": t2, "snr": snr}, sigma=ctx.sigma, echo_gain=gain, skip=skip)
                st, kept = rel_stats(tr["mono_T2"], t2), tr["bi_kept"]
                res[(snr, name, skip)] = (st, int(np.sum(kept)), len(kept))
                rows.append([f"{snr:g}", name, "tous" if skip == 0 else f"sans les {n_skip} premiers",
                             f"{pc(st['bias'], 2, True)} ± {fr(2 * st['se'], 2)}", pc(st["sd"], 2),
                             rate_txt(int(np.sum(kept)), len(kept))])
    sec.table(["SNR", "décalage du 2e écho", "échos utilisés", "biais sur T2 ± 2 erreurs-types", "dispersion de T2",
               "bi retenu à tort"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(ECHO_SHIFTS)
    for snr, color in zip(snrs, (C_SPHENE, C_LS)):
        for skip, ls, lab in ((0, "-", "tous les échos"), (n_skip, "--", f"sans les {n_skip} premiers")):
            axes[0].errorbar(range(len(names)), [res[(snr, n, skip)][0]["bias"] for n in names],
                             yerr=[2 * res[(snr, n, skip)][0]["se"] for n in names], fmt="o" + ls, color=color, capsize=3,
                             label=f"SNR {snr:g}, {lab}")
            axes[1].plot(range(len(names)), [100 * res[(snr, n, skip)][1] / res[(snr, n, skip)][2] for n in names],
                         "o" + ls, color=color, label=f"SNR {snr:g}, {lab}")
    axes[0].axhline(0, color="k", lw=0.8)
    for ax, ylab, title in ((axes[0], "biais sur T2 (%)", "Biais sur T2"), (axes[1], "bi retenu à tort (%)", "Fausses alarmes")):
        ax.set_xticks(range(len(names)), names)
        ax.set_xlabel("décalage du 2e écho"), ax.set_ylabel(ylab), ax.set_title(title)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    sec.figure(fig, "07_premiers_echos.png", f"Tissu mono T2 = {t2:.0f} ms. Trait plein : tous les échos ; tirets : sans les "
               f"{n_skip} premiers. Un petit décalage change peu de choses ; un grand décalage fausse T2 et fait retenir le bi à "
               "tort quand le bruit est faible ; ignorer les premiers échos écarte le problème.")

    # Ce que coûte l'option quand les échos sont bons : dispersion de T2, avec et sans les premiers échos
    cost = {}
    for i, t2_c in enumerate((12.0, 30.0, 120.0)):
        sd = []
        for skip in (0, n_skip):
            tr = sim.monte_carlo(sim.trial, cfg["n_echo"], seed=ctx.seed + 550_000 + 1000 * i, te=ctx.te,
                                 truth={"kind": "mono", "T2": t2_c, "snr": 30}, sigma=ctx.sigma, skip=skip)
            sd.append(rel_stats(tr["mono_T2"], t2_c)["sd"])
        cost[t2_c] = sd
    sec.text("<b>Le prix de l'option.</b> Ignorer des échos, c'est jeter de l'information. Tissu sans aucune perturbation, SNR 30 :")
    sec.table(["T2 vrai (ms)", "dispersion avec tous les échos", f"dispersion sans les {n_skip} premiers", "rapport"],
              [[f"{t:g}", pc(sd[0], 2), pc(sd[1], 2), "× " + fr(sd[1] / sd[0], 2)] for t, sd in cost.items()])

    # Le motif mesuré sur l'acanthe : ce que donne chaque réglage de l'option, sur un voxel et sur une zone
    t2_a, snr_a, n_vox_a = 30.0, 20, 64
    gain_a = np.ones(len(ctx.te))
    gain_a[:3] += ACANTHE_SHIFT
    sec.text("<b>Le motif réel de l'acanthe.</b> Sur le jeu de test de l'acanthe fourni avec le dépôt (dossier <code>data/test</code> ; "
             "la section « Vos données » le mesure quand on lui donne ces images), les trois "
             f"premiers échos sont décalés de {', '.join(pc(100 * v, 1, True) for v in ACANTHE_SHIFT)} du signal : les deux premiers "
             f"le sont, le 3<sup>e</sup> ne l'est plus. On simule ce motif exact (tissu mono T2 = {t2_a:.0f} ms, SNR {snr_a}, temps d'écho "
             f"et bruit de ce rapport) sur un voxel isolé ({cfg['n_echo']} tirages) et sur une zone de {n_vox_a} voxels "
             f"({cfg['n_echo_zone']} zones) dont l'amplitude varie de ±25 % "
             "(comme « Zone fit »), en ignorant 0, 1, 2 ou 3 échos. Sur une zone l'erreur aléatoire s'efface : le défaut "
             "systématique des échos devient visible.")
    cases = [("aucune perturbation, tous les échos", None, 0)]
    cases += [("motif de l'acanthe, " + ("tous les échos" if k == 0 else f"{k} écho{'s' * (k > 1)} ignoré{'s' * (k > 1)}"), gain_a, k)
              for k in range(4)]
    res_a, rows_a = {}, []
    for label, g, skip in cases:
        cells = []
        for kind, nv, n_trials, off in (("voxel", 1, cfg["n_echo"], 0), ("zone", n_vox_a, cfg["n_echo_zone"], 20_000)):
            tr = sim.monte_carlo(sim.trial, n_trials, seed=ctx.seed + 580_000 + off, te=ctx.te,
                                 truth={"kind": "mono", "T2": t2_a, "snr": snr_a}, sigma=ctx.sigma, n_vox=nv, het=0.25,
                                 echo_gain=g, skip=skip)
            st, kept = rel_stats(tr["mono_T2"], t2_a), tr["bi_kept"]
            res_a[(label, kind)] = (st, int(np.sum(kept)), len(kept))
            cells += [f"{pc(st['bias'], 2, True)} ± {fr(2 * st['se'], 2)}", pc(st["sd"], 2)]
            if kind == "zone":
                cells.append(rate_txt(int(np.sum(kept)), len(kept)))
        rows_a.append([label] + cells)
    sec.table(["cas", "1 voxel : biais ± 2 erreurs-types", "1 voxel : dispersion", f"zone de {n_vox_a} voxels : biais ± 2 erreurs-types",
               "zone : dispersion", "zone : bi retenu à tort"], rows_a)
    zone_a = [res_a[(cases[1 + k][0], "zone")] for k in range(4)]       # motif de l'acanthe, 0 à 3 échos ignorés
    voxel_all = res_a[(cases[1][0], "voxel")][0]

    skipped = [res[(s, n, n_skip)] for s in snrs for n in names]
    bias_ok = all(abs(st["bias"]) <= max(1.0, 4 * st["se"]) for st, _, _ in skipped)
    worst_k, worst_n = max(((k, n) for _, k, n in skipped), key=lambda kn: kn[0] / kn[1])
    fa_ok = all(rate_ok(k, n, 0.04) for _, k, n in skipped)
    sec.check(f"Sans les {n_skip} premiers échos : biais sur T2 (toutes perturbations, SNR 30 et 150)",
              pc(max(abs(st["bias"]) for st, _, _ in skipped), 2) + " au plus", "moins de 1 % (ou 4 erreurs-types)", bias_ok)
    sec.check(f"Sans les {n_skip} premiers échos : bi retenu à tort (pire cas)", rate_txt(worst_k, worst_n),
              "taux compatible avec « au plus 4 % » (test binomial)", fa_ok)
    ok_a = all(abs(zone_a[k][0]["bias"]) <= max(1.0, 4 * zone_a[k][0]["se"]) for k in (2, 3))
    sec.check("Motif de l'acanthe, zone : biais en ignorant 2 ou 3 échos",
              pc(max(abs(zone_a[k][0]["bias"]) for k in (2, 3)), 2) + " au plus", "moins de 1 % (ou 4 erreurs-types)", ok_a)
    sec.info("Motif de l'acanthe, zone : biais avec tous les échos", pc(zone_a[0][0]["bias"], 2, True))
    sec.info("Motif de l'acanthe, zone : biais en ignorant 1 seul écho", pc(zone_a[1][0]["bias"], 2, True))
    sec.info("Motif de l'acanthe, zone : bi retenu à tort en ignorant 1 seul écho", rate_txt(zone_a[1][1], zone_a[1][2]))
    st3, k3, n3 = res[(150, "3 %", 0)]
    st12, k12, n12 = res[(150, "12 %", 0)]
    sec.info("Décalage de 3 %, tous les échos, SNR 150 : biais sur T2", pc(st3["bias"], 2, True))
    sec.info("Décalage de 3 %, tous les échos, SNR 150 : bi retenu à tort", rate_txt(k3, n3))
    sec.info("Décalage de 12 %, tous les échos, SNR 150 : biais sur T2", pc(st12["bias"], 2, True))
    sec.info("Décalage de 12 %, tous les échos, SNR 150 : bi retenu à tort", rate_txt(k12, n12))
    ctx.R.update(echo_bias3=st3["bias"], echo_fa3=100 * k3 / n3, echo_bias12=st12["bias"], echo_fa12=100 * k12 / n12,
                 echo_cost={t: sd[1] / sd[0] for t, sd in cost.items()}, echo_zone_bias=[z[0]["bias"] for z in zone_a],
                 echo_zone_fa=[100 * z[1] / z[2] for z in zone_a])
    sec.conclusion(f"Un décalage de 3 % du 2<sup>e</sup> écho change T2 de {pc(st3['bias'], 1, True)} et fait retenir le bi à tort "
                   f"dans {pc(100 * k3 / n3, 1)} des voxels (SNR 150) : c'est peu. À 12 %, T2 est faussé de "
                   f"{pc(st12['bias'], 1, True)} et le bi est retenu à tort dans {pc(100 * k12 / n12, 0)} des voxels. Ignorer les "
                   f"{n_skip} premiers échos supprime ces effets mais multiplie la dispersion de T2 par "
                   f"{fr(cost[120.0][1] / cost[120.0][0], 1)} (T2 = 120 ms), {fr(cost[30.0][1] / cost[30.0][0], 1)} (30 ms) et "
                   f"{fr(cost[12.0][1] / cost[12.0][0], 1)} (12 ms). L'option est donc à <b>0 par défaut</b> : ne l'activer que si "
                   "la section « Vos données » montre un décalage des premiers échos de plus de 5 % environ. "
                   f"<b>Motif mesuré sur l'acanthe, zone de {n_vox_a} voxels</b> : avec tous les échos le biais sur T2 est de "
                   f"{pc(zone_a[0][0]['bias'], 1, True)} (un voxel isolé : {pc(voxel_all['bias'], 1, True)}, noyé dans une dispersion de "
                   f"{pc(voxel_all['sd'], 1)}) ; <b>ignorer un seul écho est pire</b> ({pc(zone_a[1][0]['bias'], 1, True)}, bi retenu à "
                   f"tort dans {pc(100 * zone_a[1][1] / zone_a[1][2], 0)} des zones, car le 2<sup>e</sup> écho reste décalé) ; en "
                   f"ignorer deux ou trois ramène le biais à {pc(max(abs(zone_a[k][0]['bias']) for k in (2, 3)), 2)} au plus. "
                   "Règle : ignorer <b>tous</b> les échos décalés, ou aucun.")


# ── 8. Cartes complètes sur jeux simulés ─────────────────────────────────────

def _disk_voxels(truth, mask2d, rng, per_disk):
    """Voxels (x, y) du masque, par disque ; sous-échantillonnés à ``per_disk`` si demandé."""
    voxels = []
    for k in range(len(truth["labels"])):
        xs, ys = np.nonzero((truth["disk"] == k) & mask2d)
        idx = np.arange(xs.size)
        if per_disk is not None and xs.size > per_disk:
            idx = rng.choice(xs.size, per_disk, replace=False)
        voxels += [(int(xs[i]), int(ys[i])) for i in idx]
    return voxels


def _run_spatial(ctx, disks, snr, seed):
    """Volume simulé -> masque -> σ mesuré -> ajustement de chaque voxel, comme dans le viewer."""
    data, truth = make_simulated.make_volume(disks, snr, ctx.te, ctx.sigma, seed)
    mask = silent(compute_mask, data, method="rician")
    sigma_est = estimate_sigma(data)
    rng = np.random.default_rng(seed)
    voxels = _disk_voxels(truth, mask[:, :, 0], rng, ctx.cfg["sample_per_disk"])
    maps, _ = fit_slice(data, ctx.te, 0, voxels, sigma_est)
    return data, truth, mask, sigma_est, maps


def section_spatial(rep, ctx):
    sec = rep.section(
        "Cartes complètes sur jeux simulés",
        "Quand on fait tout comme dans le viewer (masque du tissu, niveau de bruit mesuré sur l'image, ajustement de "
        "chaque voxel), retrouve-t-on le contenu de chaque disque ?")
    per = ctx.cfg["sample_per_disk"]
    sec.text("<b>Test.</b> Les jeux simulés de <code>make_simulated.py</code> contiennent 12 disques : 4 de tissu mono "
             "et 8 de tissu bi. Ici le bruit σ n'est <b>pas</b> donné au programme : il le mesure dans le fond de l'image, "
             "comme avec vos données. "
             + (f"Pour aller plus vite, {per} voxels par disque sont ajustés (mode rapide). " if per else
                "Tous les voxels de la première coupe sont ajustés. ")
             + "Pour chaque disque : part des voxels classés bi, et valeurs retrouvées (médiane des voxels) contre vérité.")
    summary = {}
    for snr, seed_off in ((150, 0), (30, 10)):
        t0 = time.time()
        data, truth, mask, sigma_est, maps = _run_spatial(ctx, make_simulated.COUPLES, snr, ctx.seed + seed_off)
        rep_noise = check_rice_noise(data, verbose=False)
        summary[snr] = _spatial_table(sec, ctx, truth, maps, snr, sigma_est, rep_noise, time.time() - t0)
    s150, s30 = summary[150], summary[30]
    ctx.R.update(spatial150=s150, spatial30=s30)

    # contrôles sur le jeu peu bruité (celui qui doit tout retrouver)
    fa_cells = s150["fa_cells"]
    k_w, n_w = max(fa_cells, key=lambda c: c[0] / c[1])
    sec.check("σ mesuré sur l'image / σ vrai (SNR 150)", fr(s150["sigma_ratio"], 3), "entre 0,97 et 1,03",
              0.97 < s150["sigma_ratio"] < 1.03)
    sec.check("Le fond de l'image se comporte comme du bruit de Rice (SNR 150)", pc(100 * s150["spread"], 1),
              "dispersion des quartiles moins de 5 %", s150["noise_ok"])
    sec.check("Disques mono : bi retenu à tort (SNR 150, pire disque)", rate_txt(k_w, n_w),
              "taux compatible avec « au plus 4 % » (test binomial)", all(rate_ok(k, n, 0.04) for k, n in fa_cells))
    sec.check("Disques mono : écart du T2 médian (SNR 150, pire disque)", pc(s150["mono_err"], 1), "moins de 3 %", s150["mono_err"] < 3)
    sec.check("Disques bi séparables : part de voxels classés bi (SNR 150, pire disque)", pc(s150["bi_share_min"], 0),
              "au moins 90 %", s150["bi_share_min"] >= 90)
    sec.check("Disques bi séparables : écart médian sur T2c et T2l (SNR 150, pire disque)", pc(s150["bi_err"], 1),
              "moins de 10 %", s150["bi_err"] < 10)
    sec.check("Disques bi séparables : écart médian sur f (SNR 150, pire disque)", fr(s150["f_err"], 3), "moins de 0,05",
              s150["f_err"] < 0.05)
    k30, n30 = max(s30["fa_cells"], key=lambda c: c[0] / c[1])
    sec.info("SNR 30 (proche du grain) : disques mono, bi retenu à tort (pire disque)", rate_txt(k30, n30))
    sec.info("SNR 30 : disques bi, part de voxels classés bi (disque le moins bien détecté)",
             f"{pc(s30['share_all_min'], 0)} (disque {s30['share_all_arg'] + 1})")
    sec.conclusion("Avec peu de bruit, les 12 disques sont retrouvés : T2 mono, T2c, T2l et f à quelques % près et le bon modèle "
                   "dans presque tous les voxels. Au bruit du grain, les disques mono restent mono (peu de fausses alarmes) ; "
                   "les disques bi dont les composantes sont proches ne sont plus détectés voxel par voxel, comme le prévoit "
                   "la limite de Cramér-Rao (section 3). Le niveau de bruit mesuré sur l'image est correct.")


def _map_figure(sec, truth, maps, name, per_disk, caption):
    """Les mêmes cartes que le viewer (T2 mono, modèle retenu), avec les numéros de disque du tableau."""
    disk = truth["disk"]
    nx, ny = disk.shape
    sparse = per_disk is not None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    def draw(ax, arr, cmap, vmin, vmax, title):
        xs, ys = np.nonzero(np.isfinite(arr))
        sc = ax.scatter(ys, xs, c=arr[xs, ys], s=22 if sparse else 7, marker="s", cmap=cmap, vmin=vmin, vmax=vmax,
                        linewidths=0)
        ax.set_xlim(-0.5, ny - 0.5), ax.set_ylim(nx - 0.5, -0.5)
        ax.set_aspect("equal"), ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        for k in range(len(truth["labels"])):
            cx, cy = np.nonzero(disk == k)
            ax.text(cy.mean(), cx.mean(), str(k + 1), ha="center", va="center", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        return sc

    t2_hi = float(np.nanpercentile(maps["T2_mono"], 99))
    sc = draw(axes[0], maps["T2_mono"], "viridis", 0, t2_hi, "T2 mono retrouvé voxel par voxel (ms)")
    fig.colorbar(sc, ax=axes[0], fraction=0.046)
    sc = draw(axes[1], maps["best"], ListedColormap([C_SPHENE, C_LS]), 0, 1, "Modèle retenu")
    handles = [Patch(color=C_SPHENE, label="mono"), Patch(color=C_LS, label="bi")]
    axes[1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    sec.figure(fig, name, caption + (" Mode rapide : seuls quelques voxels par disque sont ajustés (points)." if sparse else ""))


def _spatial_table(sec, ctx, truth, maps, snr, sigma_est, noise, seconds):
    """Tableau + figures d'un jeu simulé à disques constants ; renvoie les pires écarts."""
    labels, disk = truth["labels"], truth["disk"]
    rows, pts, fa_cells = [], [], []
    mono_err, bi_share_min, bi_err, f_err = 0.0, 100.0, 0.0, 0.0
    share_all_min, share_all_arg = 100.0, -1
    for k, label in enumerate(labels):
        sel = (disk == k) & np.isfinite(maps["best"])
        t2c, t2l, f = (float(np.nanmedian(truth[key][disk == k])) for key in ("t2c", "t2l", "frac"))
        is_bi = abs(t2l - t2c) > 1e-6
        share = 100 * float(np.mean(maps["best"][sel]))
        label = fr_label(label)
        if not is_bi:
            est = np.nanmedian(maps["T2_mono"][sel])
            err = 100 * (est / t2c - 1)
            fa_cells.append((int(np.sum(maps["best"][sel])), int(sel.sum())))
            mono_err = max(mono_err, abs(err))
            rows.append([f"{k + 1}", label, "mono", pc(share, 0), f"T2 = {fr(est, 1)}  ({pc(err, 1, True)})"])
            pts.append(("mono", k, t2c, est, None, None, None, None))
        else:
            kept = sel & (maps["best"] == 1)
            bound = sim.crlb(ctx.te, {"kind": "bi", "f": f, "T2c": t2c, "T2l": t2l, "snr": snr}, ctx.sigma)
            sep = max(bound["T2c"], bound["T2l"]) <= SEPARABLE_CV
            if share < share_all_min:
                share_all_min, share_all_arg = share, k
            if kept.sum() >= 5:
                ec, el, ef = (float(np.nanmedian(maps[key][kept])) for key in ("T2c_bi", "T2l_bi", "f_bi"))
                txt = f"T2c = {fr(ec, 1)}, T2l = {fr(el, 1)}, f = {fr(ef, 2)}"
                pts.append(("bi", k, t2c, ec, t2l, el, f, ef))
            else:
                ec = el = ef = float("nan")
                txt = "bi non retenu"
            if sep:
                bi_share_min = min(bi_share_min, share)
                if np.isfinite(ec):
                    bi_err = max(bi_err, abs(100 * (ec / t2c - 1)), abs(100 * (el / t2l - 1)))
                    f_err = max(f_err, abs(ef - f))
                else:
                    bi_err, f_err = 100.0, 1.0
            rows.append([f"{k + 1}", label + ("" if sep else " <i>(non séparable)</i>"), "bi", pc(share, 0), txt])
    sec.text(f"<b>Jeu « couples », SNR {snr}.</b> σ mesuré = {fr(sigma_est, 0)} (vrai : {ctx.sigma:.0f}). "
             f"Calcul des voxels : {seconds:.0f} s.")
    sec.table(["n°", "disque (vérité)", "type", "classés bi", "valeurs retrouvées (médiane)"], rows)
    _map_figure(sec, truth, maps, f"08_cartes_SNR{snr}_images.png", ctx.cfg["sample_per_disk"],
                f"Jeu « couples », SNR {snr} : les cartes que donnerait le viewer. Les numéros sont ceux du tableau.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    mono = [p for p in pts if p[0] == "mono"]
    bi = [p for p in pts if p[0] == "bi"]
    axes[0].plot([p[2] for p in mono], [p[3] for p in mono], "o", color=C_SPHENE, ms=8)
    lim = [5, 150]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xscale("log"), axes[0].set_yscale("log")
    axes[0].set_xlabel("T2 vrai (ms)"), axes[0].set_ylabel("T2 retrouvé (ms)")
    axes[0].set_title("Disques mono")
    axes[1].plot([p[2] for p in bi], [p[3] for p in bi], "o", color=C_SPHENE, ms=8, label="T2c")
    axes[1].plot([p[4] for p in bi], [p[5] for p in bi], "s", color=C_LS, ms=8, label="T2l")
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xscale("log"), axes[1].set_yscale("log")
    axes[1].set_xlabel("vrai (ms)"), axes[1].set_ylabel("retrouvé (ms)")
    axes[1].set_title("Disques bi : T2c et T2l")
    axes[1].legend(fontsize=8)
    axes[2].plot([p[6] for p in bi], [p[7] for p in bi], "o", color=C_SPHENE, ms=8)
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].set_xlabel("f vrai"), axes[2].set_ylabel("f retrouvé")
    axes[2].set_title("Disques bi : proportion f")
    for ax in axes[:2]:
        plain_log_ticks(ax, "x")
        plain_log_ticks(ax, "y")
    fig.suptitle(f"Jeu « couples » — SNR {snr} : valeurs retrouvées contre vérité (un point par disque, médiane des voxels)",
                 fontsize=11)
    fig.tight_layout()
    sec.figure(fig, f"08_cartes_SNR{snr}.png", "Chaque point est un disque ; sur la diagonale = parfaitement retrouvé.")
    return {"fa_cells": fa_cells, "mono_err": mono_err, "bi_share_min": bi_share_min, "bi_err": bi_err, "f_err": f_err,
            "share_all_min": share_all_min, "share_all_arg": share_all_arg,
            "sigma_ratio": sigma_est / ctx.sigma, "spread": noise["spread"], "noise_ok": noise["spread"] < 0.05}


# ── 9. Zone hétérogène ───────────────────────────────────────────────────────

def section_populations(rep, ctx):
    sec = rep.section(
        "Zone hétérogène : une limite à connaître",
        "Si une zone mélange des voxels de T2 différents, que répond « Zone fit » ?")
    sec.text("<b>Test.</b> Jeu simulé « populations » (SNR 150) : à côté de vrais bi, des disques dont chaque voxel est mono "
             "mais où les voxels sont mélangés (50/50), dispersés ou en dégradé. On ajuste (i) chaque voxel séparément et "
             "(ii) la zone entière d'un coup, comme « Zone fit ». La <b>dispersion</b> est l'écart-type des T2 mono "
             "voxel par voxel, en % de leur médiane.")
    data, truth = make_simulated.make_volume(make_simulated.POPULATIONS, 150, ctx.te, ctx.sigma, ctx.seed + 20)
    mask = silent(compute_mask, data, method="rician")
    sigma_est = estimate_sigma(data)
    rng = np.random.default_rng(ctx.seed)
    voxels = _disk_voxels(truth, mask[:, :, 0], rng, ctx.cfg["sample_per_disk"])
    maps, _ = fit_slice(data, ctx.te, 0, voxels, sigma_est)
    rows, share, cv, zone_bi = [], [], [], []
    for k, label in enumerate(truth["labels"]):
        sel = (truth["disk"] == k) & np.isfinite(maps["best"])
        t2 = maps["T2_mono"][sel]
        share.append(100 * float(np.mean(maps["best"][sel])))
        cv.append(100 * float(np.std(t2) / np.median(t2)))
        xs, ys = np.nonzero(sel)
        zone = fit_both(ctx.te, data[xs, ys, 0, :], sigma_est)
        zone_bi.append(zone["best"] == "bi")
        rows.append([f"{k + 1}", fr_label(label), pc(share[-1], 0), pc(cv[-1], 0), "bi" if zone_bi[-1] else "mono",
                     big(zone["delta_aicc"])])
    sec.table(["n°", "disque (vérité)", "voxels classés bi", "dispersion des T2 mono", "zone entière : modèle retenu",
               "ΔAICc de la zone"], rows)
    _map_figure(sec, truth, maps, "09_zone_heterogene_images.png", ctx.cfg["sample_per_disk"],
                "Jeu « populations » : à gauche, un disque mélangé montre des T2 mono très différents d'un voxel à l'autre ; "
                "un vrai bi, lui, a le même T2 apparent partout.")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(share))
    ax.bar(x - 0.2, share, 0.4, color=C_SPHENE, label="voxels classés bi, un par un (%)")
    ax.bar(x + 0.2, cv, 0.4, color=C_LS, label="dispersion des T2 mono voxel par voxel (%)")
    ax.set_xticks(x, [f"{i + 1}\nzone : {'bi' if b else 'mono'}" for i, b in enumerate(zone_bi)], fontsize=8)
    ax.set_xlabel("disque (voir le tableau)")
    ax.set_ylim(0, max(100, max(cv)) * 1.2)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Zone hétérogène : le bi sur la zone entière ne prouve pas deux composantes DANS chaque voxel")
    fig.tight_layout()
    sec.figure(fig, "09_zone_heterogene.png", "Par disque : part de voxels classés bi, dispersion des T2 voxel par voxel, et "
               "modèle retenu sur la zone entière (sous le numéro).")
    sec.text("<b>Comment lire ce tableau.</b> Quand les voxels sont semblables entre eux (vrai bi : même T2c, T2l et f partout), "
             "la dispersion des T2 mono voxel par voxel reste petite. Quand la zone est un mélange de voxels différents, cette "
             "dispersion est grande, et la zone entière est pourtant classée bi. <b>Un bi sur une grande zone peut donc venir "
             "d'un mélange de voxels mono</b> : il ne prouve pas à lui seul deux composantes à l'intérieur de chaque voxel. "
             "Regarder les voxels un par un (« Show signal », Mono Map) permet de trancher. Ce test n'est pas encore "
             "automatisé dans le viewer : il fait partie des améliorations prévues.")
    sec.info("Disques classés bi sur la zone entière", f"{sum(zone_bi)} sur {len(zone_bi)}")
    sec.conclusion("Un résultat bi sur une zone étendue est à interpréter avec prudence : il peut venir d'un mélange de voxels "
                   "différents. Le programme dit ce que les données de la zone préfèrent ; il ne peut pas dire à lui seul "
                   "si la zone est homogène.")


# ── 10. Vos données ───────────────────────────────────────────────────────────

def section_real(rep, ctx, data, te, mask, voxel_dims, name):
    sec = rep.section(
        "Vos données",
        "Sur votre acquisition, le modèle de bruit tient-il, et le modèle (mono + bi) explique-t-il les mesures ?")
    sigma = estimate_sigma(data)
    noise = check_rice_noise(data, verbose=False)
    per_echo = sigma_per_echo(data)
    dte = np.diff(te)
    sec.text(f"<b>Fichier :</b> {name}. Dimensions {data.shape} (x, y, coupe, écho), voxels de "
             f"{fr(voxel_dims[0], 3)} × {fr(voxel_dims[1], 3)} × {fr(voxel_dims[2], 3)} mm, {len(te)} échos de {fr(te[0], 2)} à "
             f"{fr(te[-1], 2)} ms (pas moyen {fr(dte.mean(), 3)} ms{'' if np.allclose(dte, dte.mean(), rtol=1e-3) else ', PAS régulier'}). "
             f"Niveau de bruit mesuré dans le fond : <b>σ = {sigma:.0f}</b>.")
    rows = []
    for z in range(data.shape[2]):
        m = mask[:, :, z]
        if m.any():
            rows.append([f"{z + 1}", f"{int(m.sum())}", fr(np.median(data[:, :, z, 0][m]) / sigma, 1),
                         fr(np.median(data[:, :, z, -1][m]) / sigma, 1)])
    sec.table(["coupe", "voxels de tissu", "SNR au 1er écho", "SNR au dernier écho"], rows)

    zs = max(range(data.shape[2]), key=lambda z: int(mask[:, :, z].sum()))
    rng = np.random.default_rng(ctx.seed)
    xs, ys = np.nonzero(mask[:, :, zs])
    n_fit = ctx.cfg["real_voxels"]
    idx = np.arange(xs.size) if n_fit is None or xs.size <= n_fit else rng.choice(xs.size, n_fit, replace=False)
    voxels = [(int(xs[i]), int(ys[i])) for i in idx]
    maps, resid = fit_slice(data, te, zs, voxels, sigma)
    s = slice_summary(maps)
    mean_r = np.nanmean(resid, axis=0)
    sem_r = np.nanstd(resid, axis=0) / np.sqrt(np.sum(np.isfinite(resid), axis=0))
    chi2 = maps["chi2_best"][np.isfinite(maps["chi2_best"])]
    sec.text(f"<b>Ajustement de la coupe {zs + 1}</b> ({s['n']} voxels{' tirés au hasard' if n_fit and xs.size > n_fit else ''}) : "
             f"le modèle mono est retenu dans {s['n_mono']} voxels, le bi dans {s['n_bi']} ({pc(100 * s['share_bi'], 0)}). "
             f"Le χ² réduit médian vaut <b>{fr(s['chi2_median'], 2)}</b> (1 = le modèle explique les mesures au niveau du bruit).")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x_te = np.arange(1, len(per_echo) + 1)
    axes[0].plot(x_te, per_echo, "o-", color=C_SPHENE, ms=3.5)
    axes[0].axhline(sigma, color="darkorange", ls="--", label=f"σ utilisé = {sigma:.0f}")
    axes[0].set_xlabel("numéro d'écho"), axes[0].set_ylabel("σ mesuré dans le fond")
    axes[0].set_title(f"Bruit écho par écho (écart {pc(100 * noise['sigma_echo_spread'], 1)})")
    axes[0].legend(fontsize=8)
    axes[1].hist(chi2, bins=np.linspace(0, min(4, np.percentile(chi2, 99.5)), 50), color=C_SPHENE)
    axes[1].axvline(1, color="k", ls="--", label="1 = idéal")
    axes[1].axvline(np.median(chi2), color="darkorange", label=f"médiane = {fr(np.median(chi2), 2)}")
    axes[1].set_xlabel("χ² réduit du modèle retenu"), axes[1].set_ylabel("voxels")
    axes[1].set_title("Le modèle explique-t-il les mesures ?")
    axes[1].legend(fontsize=8)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].errorbar(x_te, mean_r, yerr=2 * sem_r, fmt="o-", color="firebrick", ms=3.5, capsize=2, elinewidth=0.8)
    axes[2].set_xlabel("numéro d'écho"), axes[2].set_ylabel("mesuré − attendu (unités de bruit)")
    axes[2].set_title("Résidu moyen de chaque écho (0 = juste)")
    fig.tight_layout()
    sec.figure(fig, "10_donnees_reelles.png", "Gauche : niveau de bruit mesuré écho par écho. Milieu : qualité de l'ajustement "
               "voxel par voxel. Droite : résidu moyen de chaque écho.")

    # Décalage des 3 premiers échos en % du signal : résidu moyen (en unités de bruit) x bruit / signal du tissu
    mid_max = float(np.max(np.abs(mean_r[3:])))
    mid_txt = (f"Sur tous les autres échos, le résidu moyen reste au plus à {fr(mid_max, 2)} unité de bruit : la forme de la "
               "décroissance est bien décrite." if mid_max <= 0.3 else
               f"Sur les autres échos, le résidu moyen atteint {fr(mid_max, 2)} unité de bruit : la forme de la décroissance "
               "n'est pas parfaitement décrite par le modèle.")
    sig_med = np.array([np.median(data[:, :, zs, i][mask[:, :, zs]]) for i in range(3)])
    shift_pc = 100 * mean_r[:3] * sigma / sig_med
    n_off = max([i + 1 for i, v in enumerate(shift_pc) if abs(v) >= 3], default=0)
    skip_txt = ("" if np.all(np.abs(shift_pc) < 5) else
                " <b>Si vous décidez d'ignorer des échos</b> (option <code>SKIP_FIRST_ECHOES</code>), ignorez-les tous jusqu'au dernier "
                f"qui s'écarte de plus de 3 % : ici <b>{n_off}</b> au minimum. N'en ignorez jamais moins : ignorer un seul écho quand "
                "le suivant est aussi décalé donne un résultat pire que de ne rien ignorer (section 7).")
    sec.text("<b>Les premiers échos.</b> Le résidu moyen de chaque écho (figure de droite) dit si les mesures s'écartent, en moyenne, "
             "de la décroissance ajustée. Pour les 3 premiers échos, cela représente environ "
             + ", ".join(f"<b>{pc(v, 1, True)}</b>" for v in shift_pc)
             + " du signal. La section 7 montre qu'un décalage de 3 % du 2<sup>e</sup> écho est sans conséquence, et qu'à partir de "
             "5 % environ il peut compter, surtout sur une zone de nombreux voxels où l'erreur aléatoire est petite. " + mid_txt + skip_txt)
    sec.check("Le fond de l'image est du bruit de Rice (quartiles de la loi de Rayleigh)", pc(100 * noise["spread"], 1),
              "dispersion moins de 5 %", noise["spread"] < 0.05, "warn")
    sec.check("Niveau de bruit identique d'un écho à l'autre", pc(100 * noise["sigma_echo_spread"], 1), "moins de 5 %",
              noise["sigma_echo_spread"] < 0.05, "warn")
    sec.check("χ² réduit médian du modèle retenu", fr(s["chi2_median"], 2), "entre 0,8 et 1,3", 0.8 <= s["chi2_median"] <= 1.3, "warn")
    sec.check("Décalage moyen des 3 premiers échos, en % du signal", ", ".join(pc(v, 1, True) for v in shift_pc),
              "chacun entre −5 % et +5 % (au-delà : section 7, option SKIP_FIRST_ECHOES)", bool(np.all(np.abs(shift_pc) < 5)),
              "warn")
    sec.check("Résidu moyen des autres échos (du 4e au dernier)", f"{fr(mid_max, 2)} au plus",
              "au plus 0,3 unité de bruit en valeur absolue", mid_max <= 0.3, "warn")
    sec.info("Résidu moyen des 3 premiers échos (unités de bruit)", ", ".join(fr(v, 2, True) for v in mean_r[:3]))
    sec.check("Voxels dont un T2 est sur une limite de la plage de recherche", pc(100 * s["n_limit"] / max(s["n"], 1), 1),
              "moins de 5 %", s["n_limit"] / max(s["n"], 1) < 0.05, "warn")
    sec.conclusion("Ces contrôles décrivent VOS données ; un « ATTENTION » n'est pas un échec du programme mais une consigne "
                   "de prudence. Si les premiers échos s'écartent de plus de 5 % du signal, regarder l'option SKIP_FIRST_ECHOES de "
                   "<code>main.py</code> (échos stimulés / refocalisation imparfaite) : ignorer tous les échos décalés, jamais un "
                   "seul (section 7). Si le fond ne ressemble pas à du bruit "
                   "de Rice (image filtrée, plusieurs antennes de réception), le modèle de bruit ne s'applique pas tel quel.")
    return {"sigma": sigma, "chi2": s["chi2_median"], "bi_share": s["share_bi"], "shift_pc": shift_pc, "noise": noise}


# ── 11. Votre tissu attendu ──────────────────────────────────────────────────

def section_expected(rep, ctx, args, snr_data):
    sec = rep.section(
        "Votre tissu attendu",
        "Si mon tissu a les valeurs que j'attends, que puis-je espérer retrouver avec mon acquisition ?")
    te, sigma = ctx.te, ctx.sigma
    snr = args.snr if args.snr else (snr_data if snr_data else 30.0)
    src = "donné par --snr" if args.snr else ("mesuré sur vos données" if snr_data else "valeur par défaut, à préciser avec --snr")
    have_bi = None not in (args.t2c, args.t2l, args.f)
    sec.text(f"SNR utilisé : <b>{fr(snr, 0)}</b> ({src}) ; σ = {sigma:.0f} ; {len(te)} échos de {fr(te[0], 1)} à {fr(te[-1], 1)} ms. "
             "Les tailles de zone sont des nombres de voxels ajustés ensemble. Un SNR mesuré sur vos données (signal du 1er écho / "
             "bruit) est un peu inférieur au SNR à TE = 0 : c'est une estimation prudente.")
    n_list = (1, 16, 64)
    n_trials = max(ctx.cfg["n_zone"], 20)
    if have_bi:
        truth = {"kind": "bi", "f": args.f, "T2c": args.t2c, "T2l": args.t2l, "snr": snr}
        rows = []
        for n in (1, 4, 16, 64, 256):
            b = sim.crlb(te, truth, sigma, n_vox=n)
            rows.append([f"{n}", pc(100 * b["T2c"], 0), pc(100 * b["T2l"], 0), fr(b["f"], 3)])
        need = {t: sim.snr_needed(te, truth, sigma, t) for t in (0.30, 0.10)}
        sec.text(f"<b>Tissu bi attendu : T2c = {fr(args.t2c, 1)} ms, T2l = {fr(args.t2l, 1)} ms, f = {fr(args.f, 2)}</b> "
                 f"(rapport T2l/T2c = {fr(args.t2l / args.t2c, 1)}). Meilleure précision POSSIBLE selon la taille de la zone :")
        sec.table(["voxels dans la zone", "erreur sur T2c", "erreur sur T2l", "erreur sur f (absolue)"], rows)
        sec.text("SNR nécessaire pour atteindre ±30 % puis ±10 % sur T2c et T2l (un voxel) : "
                 + " ; ".join(f"±{100 * t:.0f} % : {'impossible' if not np.isfinite(v) else fr(v, 0)}" for t, v in need.items()) + ".")
        sim_rows = []
        for i, n in enumerate(n_list):
            tr = sim.monte_carlo(sim.trial, n_trials, seed=ctx.seed + 600_000 + 1000 * i, te=te, truth=truth,
                                 sigma=sigma, n_vox=n)
            kept = tr["bi_kept"] == 1
            if kept.sum() >= 5:
                txt = " ; ".join(f"{k} {fr(np.median(tr[key][kept]), 1 if k != 'f' else 2)} "
                                 f"[{fr(np.percentile(tr[key][kept], 25), 1 if k != 'f' else 2)} – "
                                 f"{fr(np.percentile(tr[key][kept], 75), 1 if k != 'f' else 2)}]"
                                 for k, key in (("T2c", "bi_T2c"), ("T2l", "bi_T2l"), ("f", "bi_f")))
            else:
                txt = "le bi n'est presque jamais retenu"
            sim_rows.append([f"{n}", pc(100 * float(np.mean(tr["bi_kept"])), 0), txt])
        sec.text(f"<b>Simulation avec les calculs du viewer</b> ({n_trials} essais par taille de zone) :")
        sec.table(["voxels dans la zone", "bi retenu", "valeurs retrouvées quand le bi est retenu : médiane [Q1 – Q3]"], sim_rows)
    if args.t2_mono is not None:
        truth = {"kind": "mono", "T2": args.t2_mono, "snr": snr}
        rows = []
        for i, n in enumerate(n_list):
            tr = sim.monte_carlo(sim.trial, n_trials, seed=ctx.seed + 700_000 + 1000 * i, te=te, truth=truth,
                                 sigma=sigma, n_vox=n)
            r = robust(tr["mono_T2"], args.t2_mono)
            rows.append([f"{n}", pc(100 * (1 - float(np.mean(tr["bi_kept"]))), 0),
                         f"{fr(r['med'], 2)} [{fr(r['q1'], 2)} – {fr(r['q3'], 2)}]",
                         pc(100 * sim.crlb(te, truth, sigma, n_vox=n)["T2"], 2)])
        sec.text(f"<b>Tissu mono attendu : T2 = {fr(args.t2_mono, 1)} ms.</b>")
        sec.table(["voxels dans la zone", "mono retenu", "T2 retrouvé : médiane [Q1 – Q3]", "précision possible"], rows)
    if args.couples:
        rows = []
        n_zone = n_list[-1]
        for i, (t2c, t2l) in enumerate(COUPLES):
            truth = {"kind": "bi", "f": 0.5, "T2c": float(t2c), "T2l": float(t2l), "snr": snr}
            tr = sim.monte_carlo(sim.trial, n_trials, seed=ctx.seed + 800_000 + 1000 * i, te=te, truth=truth,
                                 sigma=sigma, n_vox=n_zone)
            kept = tr["bi_kept"] == 1
            txt = (f"T2c {fr(np.median(tr['bi_T2c'][kept]), 1)}  T2l {fr(np.median(tr['bi_T2l'][kept]), 1)}  "
                   f"f {fr(np.median(tr['bi_f'][kept]), 2)}") if kept.sum() >= 5 else "non retrouvé"
            b = sim.crlb(te, truth, sigma, n_vox=n_zone)
            rows.append([f"{t2c:g} / {t2l:g}", pc(100 * float(np.mean(tr["bi_kept"])), 0), txt,
                         f"{pc(100 * b['T2c'], 0)} / {pc(100 * b['T2l'], 0)}"])
        sec.text(f"<b>Plusieurs couples T2c / T2l</b> (f = 0,5, zone de {n_zone} voxels, SNR {fr(snr, 0)}) : lesquels retrouve-t-on ? "
                 "La première colonne est la vérité simulée ; la dernière, la précision que la physique autorise au mieux "
                 "(quand elle est grande, le résultat peut s'écarter de la vérité sans que le programme y soit pour quelque chose).")
        sec.table(["couple vrai (ms)", "bi retenu", "retrouvé (médiane)", "précision possible (T2c / T2l)"], rows)
    sec.conclusion("Ces chiffres sont des ordres de grandeur pour votre acquisition, valables si votre tissu est bien décrit par "
                   "les valeurs données. La « précision possible » est une limite basse : aucune méthode ne fait mieux.")


# ── Bilan ────────────────────────────────────────────────────────────────────

def write_intro_and_summary(rep, ctx, args, elapsed):
    R = ctx.R
    rep.intro.append(
        "<b>Comment lire ce rapport.</b> On fabrique des mesures dont on connaît la vérité, avec le même bruit que de vraies "
        "images IRM, on lance les calculs de Sphene-MRI, et on regarde si on retrouve la vérité. Chaque section pose une question, "
        "décrit le test, montre une figure, puis liste des <b>contrôles</b> marqués RÉUSSI ou ÉCHEC avec leur critère. "
        "Un contrôle est RÉUSSI quand l'écart observé reste dans une tolérance fixée à l'avance ; quand le résultat dépend "
        "du hasard des tirages, la tolérance inclut une marge statistique (plusieurs erreurs-types) ou un test binomial : "
        "un contrôle n'échoue donc que si l'écart est trop grand pour être dû au hasard. "
        "Les tirages sont reproductibles (graine " + str(args.seed) + ").")
    rep.intro.append(
        "<b>Mots utilisés.</b> <b>T2</b> : vitesse à laquelle le signal décroît (en ms). <b>Mono</b> : un seul T2. <b>Bi</b> : deux "
        "composantes, une courte (T2c), une longue (T2l), dans la proportion f. <b>SNR</b> : signal divisé par le bruit. "
        "<b>Biais</b> : erreur moyenne (devrait être nulle). <b>Dispersion</b> : à quel point les résultats varient d'un voxel à "
        "l'autre. <b>Limite de Cramér-Rao</b> : la meilleure précision qu'une méthode puisse atteindre avec ce bruit. "
        "<b>ΔAICc</b> : mesure de combien le bi explique mieux les données que le mono, une fois retiré le prix de ses "
        "paramètres en plus. <b>χ² réduit</b> : 1 si le modèle explique les mesures au niveau du bruit.")
    quick = args.mode_name == "quick"
    import platform

    import scipy
    rep.intro.append(
        "<b>Traçabilité.</b> Commande : <code>python validation.py " + " ".join(sys.argv[1:]) + "</code> ; Python "
        f"{platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}. Pour relancer à l'identique, "
        "reprendre cette commande : mêmes graines, donc mêmes résultats, à de très petites différences d'arrondi près.")
    rep.intro.append(
        f"<b>Version {'rapide' if quick else 'complète'}.</b> Bruit σ = {ctx.sigma:.0f}, {len(ctx.te)} échos de "
        f"{fr(ctx.te[0], 2)} à {fr(ctx.te[-1], 2)} ms. Durée du calcul : {fr(elapsed / 60, 1)} min. "
        + ("Moins de tirages, donc des chiffres un peu moins nets (les tolérances des contrôles sont élargies en conséquence) ; "
           "pour un document à citer, lancer la version complète (<code>pixi run validate</code>)." if quick else
           "Tirages complets."))
    L = rep.summary_lines
    if "ls_asym_bias" in R:
        L.append(f"<b>Bruit.</b> Le plancher de bruit vaut 1,25 σ comme prévu ; un ajustement classique en tire un biais "
                 f"de {pc(R['ls_asym_bias'], 1, True)} sur T2 (T2 = {R['ls_asym_t2']:.0f} ms, SNR 30), même sans aucun "
                 "bruit aléatoire.")
    if "mono_bias" in R:
        L.append(f"<b>T2 mono.</b> Méthode de Sphene : erreur typique {pc(R['mono_med'], 1)} au plus (biais moyen {pc(R['mono_bias'], 1)} "
                 f"au plus), dispersion égale à la limite théorique (rapport {fr(R['mono_ratio'], 2)} au plus). Moindres carrés "
                 f"classiques : erreur typique jusqu'à {pc(R['ls_bias'], 0, True)}.")
    if "bi_err" in R:
        L.append(f"<b>T2 bi.</b> Avec peu de bruit, T2c et T2l à {pc(max(R['bi_err']['T2c'], R['bi_err']['T2l']), 1)} près et f à "
                 f"{fr(R['bi_err']['f'], 3)} près pour les couples séparables ; au bruit du grain, la dispersion est celle "
                 "imposée par la physique.")
    if "false_alarm" in R:
        L.append(f"<b>Choix mono / bi.</b> Bi retenu à tort dans {pc(R['false_alarm'], 1)} des voxels mono ; à SNR élevé, "
                 f"le bi est détecté dans {pc(R['power_min'], 0)} des voxels au moins, dès que T2l ≥ 3 × T2c.")
    if "zone_bias" in R:
        L.append(f"<b>Zone.</b> Ajustement joint sans biais ({pc(R['zone_bias'], 1)} au plus) ; la moyenne des modules garde "
                 f"{pc(R['zone_ls_bias'], 1, True)} sur {max(ctx.cfg['zone_sizes'])} voxels.")
    if "sigma_bias10" in R:
        L.append(f"<b>Niveau de bruit.</b> Une erreur de 10 % sur σ change T2 de {pc(R['sigma_bias10'], 1)} au plus.")
    if "echo_bias3" in R:
        costs = list(R["echo_cost"].values())
        L.append(f"<b>Premiers échos.</b> Un décalage de 3 % du 2<sup>e</sup> écho change T2 de {pc(R['echo_bias3'], 1, True)} "
                 f"(bi retenu à tort : {pc(R['echo_fa3'], 1)} à SNR 150) ; à 12 %, {pc(R['echo_bias12'], 1, True)} et "
                 f"{pc(R['echo_fa12'], 0)}. Ignorer les 3 premiers échos multiplie la dispersion de T2 par {fr(min(costs), 1)} à "
                 f"{fr(max(costs), 1)} : l'option reste à 0 sauf décalage supérieur à 5 % environ ; dans ce cas ignorer tous les "
                 f"échos décalés, jamais un seul (zone de 64 voxels, motif de l'acanthe : {pc(R['echo_zone_bias'][1], 1, True)} "
                 "si l'on n'ignore qu'un seul écho).")
    L.append("<b>Limites connues.</b> (1) Couples T2c/T2l peu séparés non résolubles au bruit du grain. (2) σ doit venir d'un fond de "
             "bruit de Rice pur (images brutes, pas filtrées, une seule antenne). (3) Un bi sur une zone étendue peut venir d'un "
             "mélange de voxels. (4) Les premiers échos peuvent s'écarter du modèle (voir vos données). (5) Le T2 mesuré est un T2 "
             "« apparent » (échos stimulés, séquence).")


# ── Figures de présentation (en anglais, un seul graphique chacune) ──────────

PRES_T2, PRES_SNR = 12.0, 30          # tissue of the "noise floor" figure (same as section 1)
PRES_SNRS = (10, 20, 30, 50, 100, 150)  # SNR axis of the "error" figure (same as the full report)


def _plain_ticks(ax, axis, ticks):
    """Axis ticks written as plain numbers (1, 2, 5, 10...), decimal point, no minor labels."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks)
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def presentation_figures(te, sigma, seed, n_mono, out_dir):
    """Write the two single-graph English figures of the presentation; return their numbers.

    ``presentation_noise_floor.png``: true signal, measured average and the noise
    floor, in multiples of the noise (tissue T2 = 12 ms, SNR 30).
    ``presentation_error_vs_snr.png``: typical error on T2 for classical least squares
    and for the Rician maximum likelihood, from the same simulated trials (and the same
    seeds) as the full report, so the numbers are the ones of table 2.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    te = np.asarray(te, dtype=float)

    # Figure 1: the noise floor
    t = np.linspace(te[0], 95.0, 400)
    true = sim.true_curve(t, {"kind": "mono", "T2": PRES_T2, "snr": PRES_SNR}, sigma) / sigma
    mean = rice_mean(true * sigma, sigma) / sigma
    floor = math.sqrt(math.pi / 2)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(t, true, "--", color=C_TRUE, lw=1.8)
    ax.semilogy(t, mean, color=C_SPHENE, lw=2.2)
    ax.axhline(floor, color=C_LS, ls=":", lw=2.2)
    ax.set_xlim(0, 95)
    ax.set_ylim(0.55, 45)
    _plain_ticks(ax, "y", [1, 2, 5, 10, 20, 40])
    ax.annotate("True signal\n(no noise)", xy=(PRES_T2 * math.log(PRES_SNR), 1.0), xytext=(21, 0.85),
                va="center", fontsize=9, fontweight="bold", color=C_TRUE,
                arrowprops=dict(arrowstyle="->", color=C_TRUE, lw=1))
    ax.text(93, 2.2, "What we actually\nmeasure (average)", ha="right", va="bottom", fontsize=9,
            fontweight="bold", color=C_SPHENE)
    ax.text(93, 0.86, "Noise floor = 1.25 × noise", ha="right", va="center", fontsize=9,
            fontweight="bold", color=C_LS)
    ax.set_xlabel("Echo time (ms)")
    ax.set_ylabel("Signal, in multiples of the noise (log scale)")
    ax.set_title("Signal stops at the floor")
    fig.tight_layout()
    fig.savefig(out_dir / "presentation_noise_floor.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: typical error on T2 against the SNR (T2 = 12 ms)
    ctx = SimpleNamespace(cfg={"t2s": (PRES_T2,), "snrs": PRES_SNRS, "n_mono": n_mono}, te=te, sigma=sigma, seed=seed)
    trials = run_mono_trials(ctx)
    err_ls = [robust(trials[(PRES_T2, s)]["ls"], PRES_T2)["err"] for s in PRES_SNRS]
    err_mle = [robust(trials[(PRES_T2, s)]["mono_T2"], PRES_T2)["err"] for s in PRES_SNRS]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(PRES_SNRS, err_ls, "o-", color=C_LS, lw=2, label="Classical least squares")
    ax.plot(PRES_SNRS, err_mle, "o-", color=C_SPHENE, lw=2, label="Rician MLE")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xscale("log")
    _plain_ticks(ax, "x", list(PRES_SNRS))
    ax.set_xlim(8.5, 175)
    ax.set_ylim(-8, 118)
    ax.annotate(f"{err_ls[0]:+.0f} %".replace("-", "−"), (PRES_SNRS[0], err_ls[0]), xytext=(8, -4),
                textcoords="offset points", fontsize=9, fontweight="bold", color=C_LS)
    ax.annotate(f"{err_mle[0]:+.1f} %".replace("-", "−"), (PRES_SNRS[0], err_mle[0]), xytext=(8, 5),
                textcoords="offset points", fontsize=9, fontweight="bold", color=C_SPHENE)
    ax.set_xlabel("Signal-to-noise ratio (SNR)")
    ax.set_ylabel("Typical error on T2 (%)")
    ax.set_title(f"True T2 = {PRES_T2:g} ms")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "presentation_error_vs_snr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "presentation_numbers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["snr", "typical_error_least_squares_pct", "typical_error_rician_mle_pct"])
        for snr, a, b in zip(PRES_SNRS, err_ls, err_mle):
            w.writerow([snr, f"{a:.2f}", f"{b:.2f}"])
    return {"snr": list(PRES_SNRS), "err_ls": err_ls, "err_mle": err_mle, "floor": floor}


# ── Point d'entrée ───────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Rapport de validation de Sphene-MRI (simulations, graphes, statistiques, explications).")
    ap.add_argument("--quick", action="store_true", help="version rapide (moins de tirages)")
    ap.add_argument("--out", default="validation_report", help="dossier de sortie (défaut : validation_report)")
    ap.add_argument("--seed", type=int, default=2026, help="graine des tirages aléatoires")
    ap.add_argument("--nifti", help="vos données : ajoute la section 10 (et utilise votre niveau de bruit)")
    ap.add_argument("--acqp", help="temps d'écho de vos données (utilisés par toutes les simulations)")
    ap.add_argument("--choose-files", action="store_true", help="ouvre des fenêtres pour choisir le NIfTI et l'ACQP")
    ap.add_argument("--t2c", type=float), ap.add_argument("--t2l", type=float), ap.add_argument("--f", type=float)
    ap.add_argument("--t2-mono", type=float, help="T2 mono attendu")
    ap.add_argument("--snr", type=float, help="SNR du tissu attendu (signal à TE=0 / bruit)")
    ap.add_argument("--couples", action="store_true", help="section 11 : plusieurs couples T2c/T2l")
    ap.add_argument("--presentation", action="store_true",
                    help="écrit seulement les 2 figures anglaises de la présentation (plancher du bruit, erreur selon le SNR)")
    a = ap.parse_args(argv)
    a.mode_name = "quick" if a.quick else "full"
    return a


def main(argv=None):
    args = parse_args(argv)
    cfg = MODES[args.mode_name]
    t_start = time.time()
    te, sigma = TE_DEFAULT, SIGMA_DEFAULT
    data = mask = voxel_dims = None
    nifti_name = None

    if args.choose_files:
        from functions.io import choose_acqp, choose_nifti
        args.nifti, args.acqp = choose_nifti() or None, choose_acqp() or None
    if args.acqp:
        from functions.io import load_acqp
        te = np.asarray(load_acqp(args.acqp), dtype=float)
    if args.nifti:
        from functions.io import load_nifti, voxel_size_mm
        if not args.acqp:
            raise SystemExit("--nifti demande aussi --acqp (temps d'écho), ou utiliser --choose-files.")
        data, img = load_nifti(args.nifti)
        if data.shape[-1] != len(te):
            raise SystemExit(f"Le NIfTI a {data.shape[-1]} échos mais l'ACQP en donne {len(te)}.")
        voxel_dims = voxel_size_mm(img)[0]   # in mm, whatever the header unit
        mask = silent(compute_mask, data, method="rician")
        sigma = estimate_sigma(data)
        nifti_name = Path(args.nifti).name

    out_dir = Path(args.out)
    if args.presentation:
        t0 = time.time()
        # always the full number of draws (300): a figure that is quoted must not depend on --quick
        res = presentation_figures(te, sigma, args.seed, MODES["full"]["n_mono"], out_dir / "presentation")
        print(f"Figures de présentation écrites dans {(out_dir / 'presentation').resolve()} ({time.time() - t0:.0f} s)")
        print(f"  T2 vrai {PRES_T2:g} ms, SNR {res['snr'][0]} : erreur typique {res['err_ls'][0]:+.1f} % "
              f"(moindres carrés) contre {res['err_mle'][0]:+.1f} % (maximum de vraisemblance de Rice)")
        print(f"  plancher du bruit : {res['floor']:.4f} x sigma")
        return 0
    rep = Report(out_dir, "Sphene-MRI : rapport de validation",
                 "Les calculs de T2 retrouvent-ils la vérité ? (simulations, graphes, statistiques)")
    ctx = SimpleNamespace(cfg=cfg, te=te, sigma=sigma, seed=args.seed, R={}, mono=None, bi=None, heat=None)

    def step(label, func, *a):
        t0 = time.time()
        print(f"\n== {label} ...", flush=True)
        out = func(*a)
        print(f"   fait en {time.time() - t0:.0f} s", flush=True)
        return out

    ctx.mono = step("Tirages T2 mono", run_mono_trials, ctx)
    ctx.bi = step("Tirages T2 bi", run_bi_trials, ctx)
    ctx.heat = step("Carte de détection du bi", run_heatmap_trials, ctx)
    step("1. Le bruit", section_noise, rep, ctx)
    step("2. T2 mono", section_mono, rep, ctx)
    step("3. T2 bi", section_bi, rep, ctx)
    step("4. Choix mono / bi", section_choice, rep, ctx)
    step("5. Zone", section_zone, rep, ctx)
    step("6. Erreur sur le niveau de bruit", section_sigma, rep, ctx)
    step("7. Premiers échos perturbés", section_echoes, rep, ctx)
    step("8. Cartes complètes sur jeux simulés", section_spatial, rep, ctx)
    step("9. Zone hétérogène", section_populations, rep, ctx)
    snr_data = None
    if data is not None:
        real = step("10. Vos données", section_real, rep, ctx, data, te, mask, voxel_dims, nifti_name)
        first = np.median(data[:, :, :, 0][mask]) if mask.any() else None
        snr_data = float(first / real["sigma"]) if first else None
    if any(v is not None for v in (args.t2c, args.t2l, args.f, args.t2_mono)) or args.couples:
        step("11. Votre tissu attendu", section_expected, rep, ctx, args, snr_data)

    write_intro_and_summary(rep, ctx, args, time.time() - t_start)
    path = rep.render()

    counts = rep.counts()
    print("\n" + "=" * 78)
    for num, title, c in rep.all_checks():
        if c["status"] != "INFO":
            print(f"  [{c['status']:^9}] {num}. {c['name']} : {c['value']}")
    print("=" * 78)
    print(f"  {counts['RÉUSSI']} réussis, {counts[FAIL]} échec(s), {counts[WARN]} attention(s)")
    print(f"  Rapport : {path.resolve()}")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
