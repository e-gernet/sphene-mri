import numpy as np
from scipy import ndimage


# ── AIC ─────────────────────────────────────────────────────────────────────

def compute_aic(signal, fit, k):
    if fit is None:
        return np.inf
    n = len(signal)
    residuals = signal - fit
    rss = np.sum(residuals**2)
    if rss <= 0:
        rss = 1e-10
    return 2 * k + n * np.log(rss / n)


# ── Bruit ────────────────────────────────────────────────────────────────────

def estimate_noise(data, corner_fraction=0.05):
    """Estime µ et σ du bruit de fond depuis les coins du volume."""
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
    corners = corners[corners > 0]
    mean, std = np.mean(corners), np.std(corners)
    print(f"[Noise] Mean: {mean:.4f} | Std: {std:.4f}")
    return mean, std


# ── Masque ───────────────────────────────────────────────────────────────────

def _apply_morphology(mask):
    """Fermeture + remplissage des trous."""
    struct = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=4)
    mask = ndimage.binary_fill_holes(mask)
    return mask


def mask_histogram(data, k=3.5, use_morpho=False):
    """Seuillage µ + k·σ sur le volume entier."""
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    flat = vol.flatten()
    mu = np.mean(flat)
    sigma = np.std(flat)
    threshold = mu + k * sigma
    mask = vol > threshold
    if use_morpho:
        mask = _apply_morphology(mask)
    return mask


def mask_otsu(data, use_morpho=False):
    """Seuillage automatique d'Otsu. Référence : Otsu 1979."""
    from skimage.filters import threshold_otsu
    vol = np.max(data, axis=-1) if data.ndim == 4 else data
    threshold = threshold_otsu(vol)
    mask = vol > threshold
    if use_morpho:
        mask = _apply_morphology(mask)
    return mask


def mask_rician(data, corner_fraction=0.05, k=4.0, use_morpho=False):
    """
    Estimation σ ricien depuis les coins, seuil à k·σ_rician.
    Référence : Gudbjartsson & Patz, 1995 – Magnetic Resonance in Medicine.
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
    mean_corner = np.mean(corners[corners > 0])
    sigma_rician = mean_corner / np.sqrt(np.pi / 2)
    threshold = k * sigma_rician
    mask = vol > threshold
    if use_morpho:
        mask = _apply_morphology(mask)
    return mask


def compute_mask(data, method="rician", use_morpho=False, **kwargs):
    """
    Point d'entrée unique pour la création du masque 3D.
    method : "histogram" | "otsu" | "rician"
    """
    methods = {
        "histogram": mask_histogram,
        "otsu":      mask_otsu,
        "rician":    mask_rician,
    }
    if method not in methods:
        raise ValueError(f"Méthode inconnue : '{method}'. Choisir parmi {list(methods.keys())}")
    return methods[method](data, use_morpho=use_morpho, **kwargs)

def compute_r2(signal, fitted):
    if fitted is None:
        return np.nan
    ss_res = np.sum((signal - fitted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

def compute_rmse(signal, fitted):
    if fitted is None:
        return np.nan
    return np.sqrt(np.mean((signal - fitted) ** 2))


def compute_pearson_r(signal, fitted):
    """Coefficient de corrélation de Pearson entre signal et fit."""
    if fitted is None:
        return np.nan
    signal = np.asarray(signal, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    if np.std(signal) == 0 or np.std(fitted) == 0:
        return np.nan
    return np.corrcoef(signal, fitted)[0, 1]