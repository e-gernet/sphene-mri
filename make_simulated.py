"""
Jeux de données simulés avec une vérité connue, pour tester le pipeline.

Pour chaque jeu, trois fichiers sont écrits dans ``simulated/`` :

- ``<nom>.nii``  : volume 4-D (x, y, coupe, écho), à ouvrir avec ``pixi run sphene``
- ``<nom>.acqp`` : temps d'écho (32 échos, pas de 2,9172 ms, comme le grain de blé)
- ``<nom>.png``  : image du résultat ATTENDU (cartes T2c, T2l, f + liste des disques)

Chaque jeu est un ensemble de disques (3 lignes x 4 colonnes) sur un fond de bruit.
Dans un disque, le signal vaut  I0 * [ f*exp(-t/T2c) + (1-f)*exp(-t/T2l) ].
Un disque « mono » a T2c = T2l = T2 et f = 1.
Le bruit est celui d'une vraie image IRM en module (loi de Rice) : c'est ce qui
rend les valeurs difficiles à retrouver quand le signal est faible.

Trois jeux :

``simule_couples_SNR30``
    12 disques : 4 mono et 8 bi (plusieurs couples T2c/T2l/f). SNR 30 = proche
    de ton grain de blé. Les bi peu séparés ne seront pas retrouvés : c'est normal.
``simule_couples_SNR150``
    Les mêmes disques avec peu de bruit (SNR 150). Tout doit être retrouvé à
    quelques % près : c'est le test « le code est-il juste ? ».
``simule_populations_SNR150``
    Disques dont la moyenne ressemble à un bi mais où les voxels sont en réalité
    mono (mélange, deux tissus côte à côte, dégradés) à côté de vrais bi.

Usage
-----
    python make_simulated.py
    python make_simulated.py --out mon_dossier --sigma 513 --seed 1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

NX, NY, NZ = 96, 128, 2          # 3 lignes x 4 colonnes de disques, pas de 32 voxels
RADIUS = 12
VOXEL_MM = (0.05, 0.05, 0.5)


# ── Disques : chacun renvoie (T2c, T2l, f) par voxel ────────────────────────

def _mono(t2):
    def build(rng, xx, yy, c, r):
        n = xx.size
        return np.full(n, float(t2)), np.full(n, float(t2)), np.ones(n)
    return f"mono {t2:g} ms", build


def _bi(t2c, t2l, f):
    def build(rng, xx, yy, c, r):
        n = xx.size
        return np.full(n, float(t2c)), np.full(n, float(t2l)), np.full(n, float(f))
    return f"bi {t2c:g}/{t2l:g} ms, f={f:g}", build


def _mix(t2a, t2b):
    def build(rng, xx, yy, c, r):
        t2 = np.where(rng.random(xx.size) < 0.5, float(t2a), float(t2b))
        return t2, t2.copy(), np.ones(xx.size)
    return f"MÉLANGE de voxels mono {t2a:g} / {t2b:g} ms (50/50)", build


def _spread(t2, cv):
    def build(rng, xx, yy, c, r):
        t2v = t2 * np.exp(rng.normal(0, cv, xx.size))
        return t2v, t2v.copy(), np.ones(xx.size)
    return f"mono dispersé, médiane {t2:g} ms, dispersion {cv:.0%}", build


def _split(t2_left, t2_right):
    def build(rng, xx, yy, c, r):
        t2 = np.where(yy < c[1], float(t2_left), float(t2_right))
        return t2, t2.copy(), np.ones(xx.size)
    return f"2 tissus côte à côte : mono {t2_left:g} | {t2_right:g} ms", build


def _t2_gradient(lo, hi):
    def build(rng, xx, yy, c, r):
        t2 = lo + (hi - lo) * (yy - (c[1] - r)) / (2 * r)
        return t2, t2.copy(), np.ones(xx.size)
    return f"mono, T2 de {lo:g} à {hi:g} ms (dégradé)", build


def _f_gradient(t2c, t2l, f_lo, f_hi):
    def build(rng, xx, yy, c, r):
        f = f_lo + (f_hi - f_lo) * (yy - (c[1] - r)) / (2 * r)
        return np.full(xx.size, float(t2c)), np.full(xx.size, float(t2l)), f
    return f"bi {t2c:g}/{t2l:g} ms, f de {f_lo:g} à {f_hi:g} (dégradé)", build


COUPLES = [
    _mono(12), _mono(30), _mono(60), _mono(120),
    _bi(12, 36, 0.5), _bi(12, 60, 0.5), _bi(12, 120, 0.5), _bi(25, 50, 0.5),
    _bi(12, 60, 0.3), _bi(12, 60, 0.7), _bi(8, 40, 0.5), _bi(20, 100, 0.5),
]
POPULATIONS = [
    _bi(12, 36, 0.5), _mix(12, 36), _spread(30, 0.20), _mono(30),
    _bi(12, 60, 0.5), _mix(12, 60), _spread(30, 0.40), _split(12, 36),
    _t2_gradient(15, 45), _f_gradient(12, 40, 0.2, 0.8), _mono(40), _bi(12, 36, 0.5),
]

# (nom, disques, SNR = I0/sigma) ; le SNR est le même pour tout le fichier
DATASETS = [
    ("simule_couples_SNR30", COUPLES, 30),
    ("simule_couples_SNR150", COUPLES, 150),
    ("simule_populations_SNR150", POPULATIONS, 150),
]


def _write_acqp(path, te):
    values = "\n".join(" ".join(f"{v:.4f}" for v in te[i:i + 8]) for i in range(0, len(te), 8))
    path.write_text(
        "##TITLE=Parameter List, SIMULATED (make_simulated.py)\n"
        "##JCAMPDX=4.24\n"
        f"##$ACQ_echo_time=( {len(te)} )\n{values}\n"
        f"##$ACQ_inter_echo_time={te[1] - te[0]:.4f}\n"
        "##END=\n"
    )


def _write_expected_png(path, name, snr, labels, t2c, t2l, frac, centres):
    """Image du résultat attendu : 3 cartes + légende des disques."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, arr, title, vmax in (
        (axes[0], t2c, "T2c attendu (ms)", 130),
        (axes[1], t2l, "T2l attendu (ms)", 130),
        (axes[2], frac, "f attendu", 1),
    ):
        im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04)
        for k, (cx, cy) in enumerate(centres):
            ax.text(cy, cx, str(k + 1), color="white", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))
        ax.set_xlabel("y (colonne)")
        ax.set_ylabel("x (ligne)")
    legend = "\n".join(f"{k + 1:>2} : {lab}" for k, lab in enumerate(labels))
    fig.suptitle(f"{name}  (SNR = {snr:g})   mono : T2c = T2l = T2 et f = 1", fontsize=12)
    fig.text(0.5, -0.02, legend, ha="center", va="top", fontsize=9, family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build(name, disks, snr, te, sigma, seed, out_dir):
    rng = np.random.default_rng(seed)
    t2c = np.full((NX, NY), np.nan)
    t2l = np.full((NX, NY), np.nan)
    frac = np.full((NX, NY), np.nan)
    tissue = np.zeros((NX, NY), dtype=bool)
    xs, ys = np.mgrid[0:NX, 0:NY]
    labels, centres = [], []

    for k, (label, builder) in enumerate(disks):
        r_i, c_i = divmod(k, 4)
        centre = (16 + 32 * r_i, 16 + 32 * c_i)          # (x = ligne, y = colonne)
        disk = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 <= RADIUS ** 2
        a, b, g = builder(rng, xs[disk], ys[disk], centre, RADIUS)
        t2c[disk], t2l[disk], frac[disk] = a, b, g
        tissue |= disk
        labels.append(label)
        centres.append(centre)

    i0 = float(snr) * sigma
    amp = np.zeros((NX, NY, NZ, len(te)))
    decay = (frac[tissue, None] * np.exp(-te[None, :] / t2c[tissue, None])
             + (1 - frac[tissue, None]) * np.exp(-te[None, :] / t2l[tissue, None]))
    for z in range(NZ):
        amp[tissue, z, :] = i0 * decay
    # Bruit de Rice : module d'un signal complexe avec bruit gaussien sur chaque canal
    data = np.hypot(amp + rng.normal(0, sigma, amp.shape), rng.normal(0, sigma, amp.shape))

    img = nib.Nifti1Image(data.astype(np.float32), np.diag([*VOXEL_MM, 1.0]))
    img.header.set_zooms((*VOXEL_MM, 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(img, out_dir / f"{name}.nii")
    _write_acqp(out_dir / f"{name}.acqp", te)
    _write_expected_png(out_dir / f"{name}.png", name, snr, labels, t2c, t2l, frac, centres)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="simulated", help="dossier de sortie (défaut : simulated)")
    ap.add_argument("--sigma", type=float, default=513.0, help="bruit sigma (défaut 513 = grain de blé)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    te = 2.9172 * np.arange(1, 33)
    out = Path(a.out)
    for i, (name, disks, snr) in enumerate(DATASETS):
        build(name, disks, snr, te, a.sigma, a.seed + 10 * i, out)
        print(f"[simulé] {out / name}.nii / .acqp / .png")


if __name__ == "__main__":
    main()
