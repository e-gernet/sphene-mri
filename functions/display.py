import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import time
import matplotlib.patches as mpatches
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
from .model import fit_mono, fit_mono_offset, fit_bi, fit_bi_offset
from .utils import compute_aic, compute_r2, compute_rmse, estimate_noise
from .mapping import (run_parallel, _fit_voxel_mono, _fit_voxel_bi,
                      _fit_voxel_utils, _fit_voxel_noise, plot_histogram, _fit_voxel_mono_cfix, _fit_global,_fit_voxel_error)
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

# Palette AIC — 4 couleurs discrètes
AIC_COLORS  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
AIC_LABELS  = ["mono", "mono+offset", "bi", "bi+offset"]

def display_slice(data, te, mask=None):
    z = data.shape[2] // 2
    t = 0
    mode = {"value": None}
    cache = {}  # clé: ("mono"|"bi"|"utils"|"noise", z) → données calculées

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.2)

    img = ax.imshow(data[:, :, z, t], cmap="viridis")
    #ax.set_title(f"Slice z={z}")
    #fig.colorbar(img, ax=ax)
    cbar = fig.colorbar(img, ax=ax)
    axecho = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    echo_slider = Slider(
        ax=axecho, label='Echo (ms)',
        valstep=1, valmin=0, valmax=data.shape[3]-1, valinit=t,
    )

    axslice = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    slice_slider = Slider(
        ax=axslice, label="Slice",
        valstep=1, valmin=0, valmax=data.shape[2]-1, valinit=z,
        orientation="vertical"
    )

    def update(val):
        img.set_data(data[:, :, slice_slider.val, echo_slider.val])
        fig.canvas.draw_idle()

    echo_slider.on_changed(update)
    slice_slider.on_changed(update)

    resetax = plt.axes([0.87, 0.05, 0.10, 0.04])
    button_reset = Button(resetax, 'Reset')
    button_reset.label.set_fontsize(8)

    def reset(event):
        slice_slider.reset()
        echo_slider.reset()
        mode["value"] = None
        for b in [button_fit, button_mono_mapping, button_bi_mapping,
                button_utils, button_error, button_noise, button_3d, button_reset]:
            b.color = "lightgray"
            b.hovercolor = "lightgray"
            b.ax.set_facecolor("lightgray")
        fig.canvas.draw_idle()

    button_reset.on_clicked(reset)

    ax_button_fit          = plt.axes([0.76, 0.05, 0.10, 0.04])
    ax_button_bi_mapping   = plt.axes([0.65, 0.05, 0.10, 0.04])
    ax_button_mono_mapping = plt.axes([0.54, 0.05, 0.10, 0.04])
    ax_button_utils        = plt.axes([0.43, 0.05, 0.10, 0.04])
    ax_button_error        = plt.axes([0.32, 0.05, 0.10, 0.04])
    ax_button_noise        = plt.axes([0.21, 0.05, 0.10, 0.04])
    ax_button_3d           = plt.axes([0.10, 0.05, 0.10, 0.04])

    button_fit          = Button(ax_button_fit,          "Fit")
    button_mono_mapping = Button(ax_button_mono_mapping, "Mono Map")
    button_bi_mapping   = Button(ax_button_bi_mapping,   "Bi Map")
    button_utils        = Button(ax_button_utils,        "Utils")
    button_error        = Button(ax_button_error,        "Error")
    button_noise        = Button(ax_button_noise,        "Noise")
    button_3d           = Button(ax_button_3d,           "3D View")

    for b in [button_fit, button_mono_mapping, button_bi_mapping,
            button_utils, button_error, button_noise, button_3d]:
        b.label.set_fontsize(7)

    def _set_button(button, color, fig):
        """Helper pour changer couleur bouton + flush."""
        button.color = color
        button.hovercolor = color
        button.ax.set_facecolor(color)
        fig.canvas.draw()
        plt.pause(0.001)


    def _get_voxels(mask, z):
        nx, ny = mask.shape[0], mask.shape[1]
        return [(x, y) for x in range(nx) for y in range(ny) if mask[x, y, z]]


    # ── Remplacer le bloc make_callback complet ──────────────────────────────────

    def make_callback(button):
        def toggle_mode(event):
            for b in [button_fit, button_mono_mapping, button_bi_mapping,
                    button_utils, button_error, button_noise, button_3d, button_reset]:
                b.color = "lightgray"
                b.hovercolor = "lightgray"
                b.ax.set_facecolor("lightgray")

            if mode["value"] == "fit" and button != button_fit:
                button_fit.color = "#7bff23"
                button_fit.hovercolor = "#7bff23"
                button_fit.ax.set_facecolor("#7bff23")

            # ── Fit ──────────────────────────────────────────────────────────────
            if button == button_fit:
                if mode["value"] == "fit":
                    mode["value"] = None
                    button.color = "lightgray"
                    button.hovercolor = "lightgray"
                    button.ax.set_facecolor("lightgray")
                else:
                    mode["value"] = "fit"
                    button.color = "#7bff23"
                    button.hovercolor = "#7bff23"
                    button.ax.set_facecolor("#7bff23")

            # ── Mono Mapping ──────────────────────────────────────────────────────
            elif button == button_mono_mapping:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("mono", z)

                if cache_key in cache:
                    t2_mono, t2_off, t2_cfix, c_fixed, t_elapsed = cache[cache_key]
                    print(f"[Cache] Mono z={z} récupéré")
                else:
                    _set_button(button, "#ff8c00", fig)
                    voxels = _get_voxels(mask, z)
                    nx, ny = data.shape[0], data.shape[1]
                    t2_mono = np.full((nx, ny), np.nan)
                    t2_off  = np.full((nx, ny), np.nan)
                    t2_cfix = np.full((nx, ny), np.nan)

                    t_start = time.time()

                    # Étape 1 : mono et mono+offset
                    results = run_parallel(_fit_voxel_mono, voxels, data, te, z)
                    c_vals  = []
                    for x, y, v_mono, v_off in results:
                        t2_mono[x, y] = v_mono
                        t2_off[x, y]  = v_off

                    # Étape 2 : estimer C moyen depuis mono+offset
                    results_c = run_parallel(_fit_voxel_noise, voxels, data, te, z)
                    for x, y, c_mono, _ in results_c:
                        if np.isfinite(c_mono):
                            c_vals.append(c_mono)
                    c_fixed = float(np.median(c_vals)) if c_vals else 0.0
                    print(f"C fixe estimé : {c_fixed:.1f}")

                    # Étape 3 : mono+offset avec C fixe
                    results_cf = run_parallel(_fit_voxel_mono_cfix, voxels, data, te, z, c_fixed=c_fixed)
                    for x, y, v_cfix in results_cf:
                        t2_cfix[x, y] = v_cfix

                    t_elapsed = time.time() - t_start
                    print(f"Mono mapping terminé en {t_elapsed:.1f}s")
                    cache[cache_key] = (t2_mono, t2_off, t2_cfix, c_fixed, t_elapsed)

                _set_button(button, "#7bff23", fig)

                fig_m, axes = plt.subplots(3, 2, figsize=(10, 12))
                #fig_m.suptitle(
                    #f"Mono Mapping — slice z={z} | C fixe={c_fixed:.1f} | {t_elapsed:.1f}s",
                    #fontsize=13)
                fig_m.canvas.manager.set_window_title("Mono Mapping")

                for row, arr, title in [
                    (0, t2_mono, "T2 mono (ms)"),
                    (1, t2_off,  "T2 mono exponentielle (ms)"),
                    (2, t2_cfix, f"T2 mono+offset C fixe={c_fixed:.1f} (ms)"),
                ]:
                    finite_vals = arr[np.isfinite(arr)]
                    vmin = np.percentile(finite_vals, 2)  if len(finite_vals) > 0 else None
                    vmax = np.percentile(finite_vals, 98) if len(finite_vals) > 0 else None

                    # Carte
                    im = axes[row, 0].imshow(arr, cmap="hot", origin="upper",
                                            interpolation="nearest", vmin=vmin, vmax=vmax)
                    axes[row, 0].set_title(title)
                    axes[row, 0].axis("off")
                    fig_m.colorbar(im, ax=axes[row, 0], fraction=0.046)

                    # Distribution
                    if len(finite_vals) > 0:
                        axes[row, 1].hist(finite_vals, bins=60, color="steelblue", density=True)
                        mu = np.mean(finite_vals)
                        axes[row, 1].axvline(mu, linestyle='--', color="cyan",
                                            label=f"µ={mu:.1f} ms")
                        axes[row, 1].legend()
                    axes[row, 1].set_title(f"Distribution {title}")
                    axes[row, 1].set_xlabel("T2 (ms)")
                    axes[row, 1].set_ylabel("Densité")

                plt.tight_layout()
                plt.show()

            # ── Bi Mapping ────────────────────────────────────────────────────────
            elif button == button_bi_mapping:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("bi", z)

                if cache_key in cache:
                    t2c_bi, t2l_bi, t2c_bioff, t2l_bioff, t2_eff_bi, t2_eff_bioff, t_elapsed = cache[cache_key]
                    print(f"[Cache] Bi z={z} récupéré")
                else:
                    _set_button(button, "#ff8c00", fig)
                    voxels = _get_voxels(mask, z)
                    nx, ny = data.shape[0], data.shape[1]
                    t2c_bi       = np.full((nx, ny), np.nan)
                    t2l_bi       = np.full((nx, ny), np.nan)
                    t2c_bioff    = np.full((nx, ny), np.nan)
                    t2l_bioff    = np.full((nx, ny), np.nan)
                    t2_eff_bi    = np.full((nx, ny), np.nan)
                    t2_eff_bioff = np.full((nx, ny), np.nan)

                    t_start = time.time()
                    results = run_parallel(_fit_voxel_bi, voxels, data, te, z)
                    for x, y, v_t2c, v_t2l, v_t2c_off, v_t2l_off, v_t2_eff_bi, v_t2_eff_bioff in results:
                        t2c_bi[x, y]       = v_t2c
                        t2l_bi[x, y]       = v_t2l
                        t2c_bioff[x, y]    = v_t2c_off
                        t2l_bioff[x, y]    = v_t2l_off
                        t2_eff_bi[x, y]    = v_t2_eff_bi
                        t2_eff_bioff[x, y] = v_t2_eff_bioff

                    t_elapsed = time.time() - t_start
                    print(f"Bi mapping terminé en {t_elapsed:.1f}s")
                    cache[cache_key] = (t2c_bi, t2l_bi, t2c_bioff, t2l_bioff, t2_eff_bi, t2_eff_bioff, t_elapsed)

                _set_button(button, "#7bff23", fig)

                fig_b, axes = plt.subplots(3, 2, figsize=(10, 12))
                #fig_b.suptitle(
                    #f"Bi Mapping — slice z={z} | {t_elapsed:.1f}s", fontsize=13)
                fig_b.canvas.manager.set_window_title("Bi Mapping")

                for ax_b, arr, title in [
                    (axes[0, 0], t2c_bi,      "T2c bi (ms)"),
                    (axes[0, 1], t2l_bi,      "T2l bi (ms)"),
                    (axes[1, 0], t2c_bioff,   "T2 court bi exponentielle (ms)"),
                    (axes[1, 1], t2l_bioff,   "T2 long bi exponentielle (ms)"),
                    (axes[2, 0], t2_eff_bi,   "T2 effectif bi (ms)"),
                    (axes[2, 1], t2_eff_bioff,"T2 effectif bi+offset (ms)"),
                ]:
                    finite_vals = arr[np.isfinite(arr)]
                    if len(finite_vals) > 0:
                        vmin = np.percentile(finite_vals, 2)
                        # Borner T2l à 100ms pour la lisibilité
                        if "T2l" in title:
                            vmax = 100.0
                        else:
                            vmax = np.percentile(finite_vals, 98)
                    else:
                        vmin, vmax = None, None

                    im = ax_b.imshow(arr, cmap="hot", origin="upper",
                                    interpolation="nearest", vmin=vmin, vmax=vmax)
                    ax_b.set_title(title)
                    ax_b.axis("off")
                    fig_b.colorbar(im, ax=ax_b, fraction=0.046)

                plt.tight_layout()
                plt.show()

            # ── Utils ─────────────────────────────────────────────────────────────
            elif button == button_utils:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("utils", z)

                if cache_key in cache:
                    (aic_map, i0_voxelwise, i0_global, r2_map, rmse_map,
                    f_bi_map, f_bioff_map, t2_eff_map,
                    global_best, model_counts, t_elapsed) = cache[cache_key]
                    print(f"[Cache] Utils z={z} récupéré")
                else:
                    _set_button(button, "#ff8c00", fig)
                    voxels = _get_voxels(mask, z)
                    nx, ny = data.shape[0], data.shape[1]
                    model_labels = ["mono", "mono+offset", "bi", "bi+offset"]
                    model_counts     = {k: 0 for k in model_labels}
                    aic_map          = np.full((nx, ny), np.nan)
                    i0_voxelwise     = np.full((nx, ny), np.nan)
                    r2_map           = np.full((nx, ny), np.nan)
                    rmse_map         = np.full((nx, ny), np.nan)
                    f_bi_map         = np.full((nx, ny), np.nan)
                    f_bioff_map      = np.full((nx, ny), np.nan)
                    t2_eff_map       = np.full((nx, ny), np.nan)
                    i0_per_model_map = {k: np.full((nx, ny), np.nan) for k in model_labels}

                    t_start = time.time()
                    results = run_parallel(_fit_voxel_utils, voxels, data, te, z)
                    for (x, y, best_idx, best, i0_vw, r2, rmse,
                        i0_per_model, f_bi_val, f_bioff_val, t2_eff, _) in results:
                        aic_map[x, y]      = best_idx
                        i0_voxelwise[x, y] = i0_vw
                        r2_map[x, y]       = r2
                        rmse_map[x, y]     = rmse
                        f_bi_map[x, y]     = f_bi_val
                        f_bioff_map[x, y]  = f_bioff_val
                        t2_eff_map[x, y]   = t2_eff
                        model_counts[best] += 1
                        for k in model_labels:
                            i0_per_model_map[k][x, y] = i0_per_model[k]

                    global_best = max(model_counts, key=model_counts.get)
                    i0_global   = i0_per_model_map[global_best]

                    t_elapsed = time.time() - t_start
                    print(f"Utils terminées en {t_elapsed:.1f}s | Global : {global_best}")
                    print(f"Counts : {model_counts}")

                    cache[cache_key] = (aic_map, i0_voxelwise, i0_global, r2_map, rmse_map,
                                        f_bi_map, f_bioff_map, t2_eff_map,
                                        global_best, model_counts, t_elapsed)

                _set_button(button, "#7bff23", fig)

                fig_u, axes = plt.subplots(3, 2, figsize=(10, 12))
                fig_u.suptitle(
                    f"Utils — z={z} | {global_best} | {t_elapsed:.1f}s", fontsize=13)
                fig_u.canvas.manager.set_window_title("Utils")

                # [0,0] AIC
                cmap_aic = ListedColormap(AIC_COLORS)
                axes[0, 0].imshow(aic_map, cmap=cmap_aic, vmin=-0.5, vmax=3.5,
                                origin="upper", interpolation="nearest")
                axes[0, 0].set_title("Meilleur modèle (AIC)")
                axes[0, 0].axis("off")
                patches = [mpatches.Patch(color=AIC_COLORS[i], label=AIC_LABELS[i])
                        for i in range(4)]
                axes[0, 0].legend(handles=patches, loc="lower right",
                                fontsize=7, framealpha=0.7)

                # [0,1] T2 effectif
                # [0,1] Image lissée
                vol_slice  = data[:, :, z, 0].astype(float)
                vol_masked = np.where(mask[:, :, z], vol_slice, np.nan) if mask is not None else vol_slice
                vol_smooth = gaussian_filter(np.nan_to_num(vol_masked), sigma=1.5)
                vol_smooth = np.where(mask[:, :, z], vol_smooth, np.nan) if mask is not None else vol_smooth

                im_smooth = axes[0, 1].imshow(vol_smooth, cmap="viridis",
                                            origin="upper", interpolation="nearest")
                axes[0, 1].set_title("Image lissée (σ=1.5, echo 0)")
                axes[0, 1].axis("off")
                fig_u.colorbar(im_smooth, ax=axes[0, 1], fraction=0.046)

                # [1,0] Fraction f bi
                im_fbi = axes[1, 0].imshow(f_bi_map, cmap="RdBu_r", vmin=0, vmax=1,
                                            origin="upper", interpolation="nearest")
                axes[1, 0].set_title("Fraction f (bi)\neau restreinte vs libre")
                axes[1, 0].axis("off")
                fig_u.colorbar(im_fbi, ax=axes[1, 0], fraction=0.046)

                # [1,1] Fraction f bi+offset
                im_fbioff = axes[1, 1].imshow(f_bioff_map, cmap="RdBu_r", vmin=0, vmax=1,
                                            origin="upper", interpolation="nearest")
                axes[1, 1].set_title("Fraction f (bi+offset)\neau restreinte vs libre")
                axes[1, 1].axis("off")
                fig_u.colorbar(im_fbioff, ax=axes[1, 1], fraction=0.046)

                # [2,0] I0 voxel-wise
                finite_i0 = i0_voxelwise[np.isfinite(i0_voxelwise)]
                vmin_i0 = np.percentile(finite_i0, 1)  if len(finite_i0) > 0 else None
                vmax_i0 = np.percentile(finite_i0, 99) if len(finite_i0) > 0 else None
                im_i0v = axes[2, 0].imshow(i0_voxelwise, cmap="viridis",
                                            vmin=vmin_i0, vmax=vmax_i0,
                                            origin="upper", interpolation="nearest")
                axes[2, 0].set_title("I0 voxel-wise")
                axes[2, 0].axis("off")
                fig_u.colorbar(im_i0v, ax=axes[2, 0], fraction=0.046)

                # [2,1] I0 global
                im_i0g = axes[2, 1].imshow(i0_global, cmap="viridis",
                                            origin="upper", interpolation="nearest")
                axes[2, 1].set_title(f"I0 global ({global_best})")
                axes[2, 1].axis("off")
                fig_u.colorbar(im_i0g, ax=axes[2, 1], fraction=0.046)

                plt.tight_layout()
                plt.show()
            # ── Noise ─────────────────────────────────────────────────────────────
            elif button == button_noise:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("noise", z)

                if cache_key in cache:
                    c_mono_off, c_bi_off, t_elapsed = cache[cache_key]
                    print(f"[Cache] Noise z={z} récupéré")
                else:
                    _set_button(button, "#ff8c00", fig)
                    voxels = _get_voxels(mask, z)
                    nx, ny = data.shape[0], data.shape[1]
                    c_mono_off = np.full((nx, ny), np.nan)
                    c_bi_off   = np.full((nx, ny), np.nan)

                    t_start = time.time()
                    results = run_parallel(_fit_voxel_noise, voxels, data, te, z)
                    for x, y, c_mono, c_bi in results:
                        c_mono_off[x, y] = c_mono
                        c_bi_off[x, y]   = c_bi
                    t_elapsed = time.time() - t_start
                    print(f"Noise maps terminées en {t_elapsed:.1f}s")
                    cache[cache_key] = (c_mono_off, c_bi_off, t_elapsed)

                mean_n, std_n = estimate_noise(data)
                _set_button(button, "#7bff23", fig)

                fig_n, axes = plt.subplots(3, 2, figsize=(10, 12))
                #fig_n.suptitle(f"Noise — slice z={z} | {t_elapsed:.1f}s", fontsize=13)
                fig_n.canvas.manager.set_window_title("Noise")

                plot_histogram(data, ax=axes[0, 0])

                if mask is not None:
                    axes[0, 1].imshow(mask[:, :, z], cmap="gray",
                                    origin="upper", interpolation="nearest")
                #axes[0, 1].set_title(f"Masque · z={z}")
                axes[0, 1].axis("off")

                c_vals_mono = c_mono_off[np.isfinite(c_mono_off)]
                im_cm = axes[1, 0].imshow(c_mono_off, cmap="plasma",
                                        origin="upper", interpolation="nearest")
                axes[1, 0].set_title("Carte C (mono+offset)")
                axes[1, 0].axis("off")
                fig_n.colorbar(im_cm, ax=axes[1, 0], fraction=0.046)

                if len(c_vals_mono) > 0:
                    axes[1, 1].hist(c_vals_mono, bins=50, color="steelblue", density=True)
                    mu_cm = np.mean(c_vals_mono)
                    axes[1, 1].axvline(mu_cm, linestyle='--', color="cyan",
                                    label=f"µ={mu_cm:.1f}")
                    axes[1, 1].legend()
                axes[1, 1].set_title("Distribution C (mono+offset)")
                axes[1, 1].set_xlabel("C")
                axes[1, 1].set_ylabel("Densité")

                c_vals_bi = c_bi_off[np.isfinite(c_bi_off)]
                im_cb = axes[2, 0].imshow(c_bi_off, cmap="plasma",
                                        origin="upper", interpolation="nearest")
                axes[2, 0].set_title("Carte C (bi+offset)")
                axes[2, 0].axis("off")
                fig_n.colorbar(im_cb, ax=axes[2, 0], fraction=0.046)

                if len(c_vals_bi) > 0:
                    axes[2, 1].hist(c_vals_bi, bins=50, color="steelblue", density=True)
                    mu_cb = np.mean(c_vals_bi)
                    axes[2, 1].axvline(mu_cb, linestyle='--', color="cyan",
                                    label=f"µ={mu_cb:.1f}")
                    axes[2, 1].legend()
                axes[2, 1].set_title("Distribution C (bi+offset)")
                axes[2, 1].set_xlabel("C")
                axes[2, 1].set_ylabel("Densité")

                plt.tight_layout()
                plt.show()



            elif button == button_error:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key_error = ("error", z)

                if cache_key_error in cache:
                    (r2_map, rmse_map, r2_glob_map, rmse_glob_map,
                    global_best, t_elapsed) = cache[cache_key_error]
                    print(f"[Cache] Error z={z} récupéré")
                else:
                    _set_button(button, "#ff8c00", fig)
                    voxels = _get_voxels(mask, z)
                    nx, ny = data.shape[0], data.shape[1]
                    r2_map        = np.full((nx, ny), np.nan)
                    rmse_map      = np.full((nx, ny), np.nan)
                    r2_glob_map   = np.full((nx, ny), np.nan)
                    rmse_glob_map = np.full((nx, ny), np.nan)
                    all_fitted    = {}
                    counts        = {"mono": 0, "mono+offset": 0, "bi": 0, "bi+offset": 0}

                    t_start = time.time()

                    # Passe 1 — tous les fits, R²/RMSE voxel-wise
                    results = run_parallel(_fit_voxel_error, voxels, data, te, z)
                    for x, y, best, r2_vw, rmse_vw, fitted_map in results:
                        r2_map[x, y]   = r2_vw
                        rmse_map[x, y] = rmse_vw
                        counts[best]  += 1
                        all_fitted[(x, y)] = fitted_map

                    global_best = max(counts, key=counts.get)
                    print(f"Modèle global : {global_best} | Counts : {counts}")

                    # Passe 2 — R²/RMSE global avec modèle majoritaire
                    for x, y in voxels:
                        f = all_fitted[(x, y)][global_best]
                        signal = data[x, y, z, :]
                        r2_glob_map[x, y]   = compute_r2(signal, f)   if f is not None else np.nan
                        rmse_glob_map[x, y] = compute_rmse(signal, f) if f is not None else np.nan

                    t_elapsed = time.time() - t_start
                    print(f"Error terminé en {t_elapsed:.1f}s")
                    cache[cache_key_error] = (r2_map, rmse_map, r2_glob_map,
                                            rmse_glob_map, global_best, t_elapsed)

                _set_button(button, "#7bff23", fig)

                fig_e, axes = plt.subplots(2, 3, figsize=(14, 8))
                fig_e.suptitle(
                    f"Error — z={z} | global={global_best} | {t_elapsed:.1f}s", fontsize=13)
                fig_e.canvas.manager.set_window_title("Error")

                for row, (arr_vw, arr_glob, label, cmap_e, vmin_e, vmax_e) in enumerate([
                    (rmse_map, rmse_glob_map, "RMSE", "hot_r",   None, None),
                    (r2_map,   r2_glob_map,   "R²",  "RdYlGn",  0,    1   ),
                ]):
                    for col, (arr, title) in enumerate([
                        (arr_vw,   f"{label} voxel-wise"),
                        (arr_glob, f"{label} global ({global_best})"),
                    ]):
                        finite_v = arr[np.isfinite(arr)]
                        vm_min = vmin_e if vmin_e is not None else (np.percentile(finite_v, 2)  if len(finite_v) > 0 else None)
                        vm_max = vmax_e if vmax_e is not None else (np.percentile(finite_v, 98) if len(finite_v) > 0 else None)

                        im_e = axes[row, col].imshow(arr, cmap=cmap_e, vmin=vm_min, vmax=vm_max,
                                                    origin="upper", interpolation="nearest")
                        axes[row, col].set_title(title)
                        axes[row, col].axis("off")
                        fig_e.colorbar(im_e, ax=axes[row, col], fraction=0.046)

                    # Distribution superposée [col 2]
                    ax_dist = axes[row, 2]
                    for arr, label_d, color in [
                        (arr_vw,   "voxel-wise", "steelblue"),
                        (arr_glob, "global",     "tomato"),
                    ]:
                        finite_v = arr[np.isfinite(arr)]
                        if len(finite_v) > 0:
                            mu = np.mean(finite_v)
                            ax_dist.hist(finite_v, bins=60, density=True,
                                        alpha=0.5, color=color,
                                        label=f"{label_d} µ={mu:.3f}" if label == "R²" else f"{label_d} µ={mu:.1f}")
                            ax_dist.axvline(mu, color=color, linestyle="--", linewidth=1.5)

                    ax_dist.set_title(f"Distribution {label}")
                    ax_dist.set_xlabel(label)
                    ax_dist.set_ylabel("Densité")
                    ax_dist.legend(fontsize=8)

                plt.tight_layout()
                plt.show()


            elif button == button_3d:
                mode["value"] = None
                _set_button(button, "#ff8c00", fig)

                # Volume 3D — echo 0 sur toutes les slices
                vol = data[:, :, :, 0].astype(float)

                # Appliquer le masque
                if mask is not None:
                    vol_masked = np.where(mask, vol, np.nan)
                else:
                    vol_masked = vol.copy()

                # Seuil pour l'isosurface — percentile 30 des valeurs masquées
                finite_vals = vol_masked[np.isfinite(vol_masked)]
                if len(finite_vals) == 0:
                    print("[3D] Aucune valeur valide.")
                    _set_button(button, "lightgray", fig)
                    return

                iso_min = np.percentile(finite_vals, 30)
                iso_max = np.percentile(finite_vals, 98)

                nx, ny, nz = vol_masked.shape
                x_idx, y_idx, z_idx = np.meshgrid(
                    np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
                )

                # Remplacer NaN par 0 pour plotly
                vol_plot = np.nan_to_num(vol_masked, nan=0.0)

                fig_3d = go.Figure(data=go.Volume(
                    x=x_idx.flatten(),
                    y=y_idx.flatten(),
                    z=z_idx.flatten(),
                    value=vol_plot.flatten(),
                    isomin=iso_min,
                    isomax=iso_max,
                    opacity=0.15,           # transparence globale
                    surface_count=20,       # nombre de couches de rendu
                    colorscale="Viridis",
                    caps=dict(x_show=False, y_show=False, z_show=False),
                ))

                fig_3d.update_layout(
                    title=f"Vue 3D — echo 0 | seuil {iso_min:.0f}–{iso_max:.0f}",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z (slice)",
                        aspectmode="data",   # respecte les proportions réelles
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                )

                _set_button(button, "#7bff23", fig)
                fig_3d.show()   # ouvre dans le navigateur
                fig.canvas.draw_idle()


            fig.canvas.draw_idle()

        return toggle_mode

    button_fit.on_clicked(make_callback(button_fit))
    button_mono_mapping.on_clicked(make_callback(button_mono_mapping))
    button_bi_mapping.on_clicked(make_callback(button_bi_mapping))
    button_utils.on_clicked(make_callback(button_utils))
    button_noise.on_clicked(make_callback(button_noise))
    button_error.on_clicked(make_callback(button_error))
    button_3d.on_clicked(make_callback(button_3d))
    

    def onclick(event):
        toolbar = event.canvas.toolbar
        ui_axes = [
            button_fit.ax, button_mono_mapping.ax, button_bi_mapping.ax,
            button_utils.ax, button_noise.ax, button_reset.ax,
            axecho, axslice,
        ]
        if event.inaxes in ui_axes:
            return
        if toolbar is not None and toolbar.mode != '':
            return
        if event.xdata is None or event.ydata is None:
            return
        if mode["value"] != "fit":
            return
        if event.inaxes is not ax:
            return

        x = int(event.xdata)
        y = int(event.ydata)
        signal = data[y, x, int(slice_slider.val), :]

        p_mono,  f_mono,  cov_mono  = fit_mono(te, signal)
        p_off,   f_off,   cov_off   = fit_mono_offset(te, signal)
        p_bi,    f_bi,    cov_bi    = fit_bi(te, signal)
        p_bioff, f_bioff, cov_bioff = fit_bi_offset(te, signal)

        aic_mono  = compute_aic(signal, f_mono,  2)
        aic_off   = compute_aic(signal, f_off,   3)
        aic_bi    = compute_aic(signal, f_bi,    4)
        aic_bioff = compute_aic(signal, f_bioff, 5)

        r2_mono  = compute_r2(signal, f_mono)
        r2_off   = compute_r2(signal, f_off)
        r2_bi    = compute_r2(signal, f_bi)
        r2_bioff = compute_r2(signal, f_bioff)

        rmse_mono  = compute_rmse(signal, f_mono)
        rmse_off   = compute_rmse(signal, f_off)
        rmse_bi    = compute_rmse(signal, f_bi)
        rmse_bioff = compute_rmse(signal, f_bioff)

        aic_dict   = {
            "mono": aic_mono, "mono+offset": aic_off,
            "bi": aic_bi, "bi+offset": aic_bioff
        }
        best_model = min(aic_dict, key=aic_dict.get)

        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  Voxel ({x}, {y}) | slice z={int(slice_slider.val)}")
        print(sep)
        print(f"  {'Modèle':<18} {'I0':>8} {'T2':>7} {'AIC':>8} "
            f"{'R²':>6} {'RMSE':>8}")
        print(sep)

        def _row(name, params, aic, r2, rmse, extra=""):
            if params is None:
                return f"  {name:<18} {'—':>8} {'—':>7} {'—':>8} {'—':>6} {'—':>8}"
            I0 = params.get("I0", float("nan"))
            T2 = params.get("T2", params.get("T2c", float("nan")))
            return (f"  {name:<18} {I0:>8.1f} {T2:>7.2f} {aic:>8.1f} "
                    f"{r2:>6.3f} {rmse:>8.1f}{extra}")

        print(_row("mono",        p_mono,  aic_mono,  r2_mono,  rmse_mono))
        print(_row("mono+offset", p_off,   aic_off,   r2_off,   rmse_off,
                f"  C={p_off['C']:.1f}" if p_off else ""))
        print(_row("bi",          p_bi,    aic_bi,    r2_bi,    rmse_bi,
                f"  f={p_bi['f']:.2f} T2l={p_bi['T2l']:.1f}" if p_bi else ""))
        print(_row("bi+offset", p_bioff, aic_bioff, r2_bioff, rmse_bioff,
           f"  f={p_bioff['f']:.2f} T2l={p_bioff['T2l']:.1f} C={p_bioff['C']:.1f}" 
           if p_bioff else ""))

        print(sep)
        print(f"  ★ Meilleur modèle (AIC) : {best_model}")
        print(sep)

        fig2, ax2 = plt.subplots()
        ax2.plot(te, signal, 'o', label="data")
        ax2.set_ylim(top=15000)
        #if f_mono  is not None: ax2.plot(te, f_mono,  label=f"mono (AIC={aic_mono:.1f})")
        if f_off   is not None: ax2.plot(te, f_off,   label=f"mono exponentielle")
        #if f_bi    is not None: ax2.plot(te, f_bi,    label=f"bi (AIC={aic_bi:.1f})")
        #if f_bioff is not None: ax2.plot(te, f_bioff, label=f"bi+offset (AIC={aic_bioff:.1f})")
        ax2.legend()
        ax2.set_title(f"Voxel ({x},{y})")
        fig2.canvas.draw_idle()
        fig2.show()
        plt.pause(0.001)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()