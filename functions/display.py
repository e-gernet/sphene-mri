"""
Interactive matplotlib-based viewer for MRI T2 relaxometry data.

This module provides a single entry point :func:`display_slice` that opens
a matplotlib figure with slice/echo navigation sliders and a set of analysis
buttons. All heavy computation is cached per slice so repeated button clicks
are instantaneous.

Buttons
-------
Fit
    Click any voxel to fit all four models and plot decay curves with AIC
    scores. The best model (lowest AIC) is drawn thicker with a star label.
Mono Map
    Compute T2 maps from mono-exponential fits (standard, with offset,
    and with a globally fixed offset C) for the current slice.
Bi Map
    Compute T2 maps from bi-exponential fits (short T2c, long T2l, effective
    T2) for the current slice.
Utils
    AIC-based model selection map, water fraction maps, I0 maps, and a
    smoothed reference image.
Error
    R² and RMSE maps (voxel-wise best model and globally-selected model).
Noise
    Offset C maps from models with additive offset, plus the background
    noise histogram and the tissue mask overlay.
3D View
    Interactive volumetric rendering of the first echo using Plotly.
"""

import csv
import time
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
from scipy.ndimage import gaussian_filter

from .model import fit_mono, fit_mono_offset, fit_bi, fit_bi_offset
from .utils import compute_aic, compute_r2, compute_rmse, estimate_noise
from .io import export_table
from .mapping import (
    run_parallel,
    _fit_voxel_mono, _fit_voxel_bi, _fit_voxel_utils,
    _fit_voxel_noise, _fit_voxel_mono_cfix, _fit_global,
    _fit_voxel_error, plot_histogram,
)

# Discrete colour palette for AIC model-selection maps (4 models)
_AIC_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
_AIC_LABELS = ["mono", "mono+offset", "bi", "bi+offset"]

# Colour cycle for the per-voxel fit plot (one colour per model)
_FIT_COLORS = {
    "mono":        "#4C72B0",
    "mono+offset": "#55A868",
    "bi":          "#C44E52",
    "bi+offset":   "#8172B2",
}


def _use_array_coords(ax, arr):
    """Make the toolbar coordinate readout match array/CSV indexing.

    By default matplotlib's toolbar shows ``(x, y)`` as
    ``(horizontal position, vertical position)`` on screen — which does NOT
    match ``arr[x, y]`` indexing (numpy's first axis is vertical, second is
    horizontal). This swaps the readout so it directly shows the same
    ``x`` (row) and ``y`` (column) indices used in the exported CSV,
    removing the need to mentally swap coordinates by hand.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which an image (``imshow``) was drawn from ``arr``.
    arr : np.ndarray of shape (nx, ny)
        The 2-D array displayed on ``ax``.
    """
    def format_coord(plot_x, plot_y):
        row = int(round(plot_y))
        col = int(round(plot_x))
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            val = arr[row, col]
            return f"x={row}  y={col}  val={val:.4f}"
        return ""
    ax.format_coord = format_coord


def display_slice(data, te, mask=None):
    """Launch the interactive T2 relaxometry viewer.

    Opens a matplotlib figure showing the MRI volume with slice and echo
    navigation sliders. Seven analysis buttons trigger per-slice computations
    that are cached after the first run.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz, n_te)
        4-D MRI data array (float64).
    te : array-like of shape (n_te,)
        Echo times in milliseconds, in acquisition order.
    mask : np.ndarray of shape (nx, ny, nz), dtype bool, optional
        Binary tissue mask. If ``None``, all voxels are processed (slow).

    Notes
    -----
    The viewer is blocking: ``plt.show()`` at the end of this function holds
    execution until the window is closed.

    Examples
    --------
    >>> display_slice(data, te_values, mask=mask)
    """
    z = data.shape[2] // 2
    t = 0
    mode   = {"value": None}
    cache  = {}

    # ── Main figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.2)

    img  = ax.imshow(data[:, :, z, t], cmap="viridis")
    cbar = fig.colorbar(img, ax=ax)  # noqa: F841  (kept for layout)
    _use_array_coords(ax, data[:, :, z, t])

    axecho = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    echo_slider = Slider(
        ax=axecho, label="Echo (ms)",
        valstep=1, valmin=0, valmax=data.shape[3] - 1, valinit=t,
    )

    axslice = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    slice_slider = Slider(
        ax=axslice, label="Slice",
        valstep=1, valmin=0, valmax=data.shape[2] - 1, valinit=z,
        orientation="vertical",
    )

    def update(val):
        img.set_data(data[:, :, slice_slider.val, echo_slider.val])
        fig.canvas.draw_idle()

    echo_slider.on_changed(update)
    slice_slider.on_changed(update)

    # ── Buttons ───────────────────────────────────────────────────────────────
    # 9 buttons on one row: width + gap chosen so they all fit from x=0.02 to x=0.93
    _btn_w = 0.095
    _btn_gap = 0.005
    _btn_x = [0.02 + i * (_btn_w + _btn_gap) for i in range(9)]

    resetax            = plt.axes([_btn_x[8], 0.05, _btn_w, 0.04])
    ax_button_fit      = plt.axes([_btn_x[7], 0.05, _btn_w, 0.04])
    ax_button_bi       = plt.axes([_btn_x[6], 0.05, _btn_w, 0.04])
    ax_button_mono     = plt.axes([_btn_x[5], 0.05, _btn_w, 0.04])
    ax_button_utils    = plt.axes([_btn_x[4], 0.05, _btn_w, 0.04])
    ax_button_error    = plt.axes([_btn_x[3], 0.05, _btn_w, 0.04])
    ax_button_noise    = plt.axes([_btn_x[2], 0.05, _btn_w, 0.04])
    ax_button_3d       = plt.axes([_btn_x[1], 0.05, _btn_w, 0.04])
    ax_button_export   = plt.axes([_btn_x[0], 0.05, _btn_w, 0.04])

    button_reset        = Button(resetax,         "Reset")
    button_fit          = Button(ax_button_fit,   "Fit")
    button_bi_mapping   = Button(ax_button_bi,    "Bi Map")
    button_mono_mapping = Button(ax_button_mono,  "Mono Map")
    button_utils        = Button(ax_button_utils, "Utils")
    button_error        = Button(ax_button_error, "Error")
    button_noise        = Button(ax_button_noise, "Noise")
    button_3d           = Button(ax_button_3d,    "3D View")
    button_export       = Button(ax_button_export, "Export")

    _all_buttons = [
        button_fit, button_mono_mapping, button_bi_mapping,
        button_utils, button_error, button_noise, button_3d,
        button_export, button_reset,
    ]
    for b in _all_buttons:
        b.label.set_fontsize(7)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_button(button, color):
        button.color      = color
        button.hovercolor = color
        button.ax.set_facecolor(color)
        fig.canvas.draw()
        plt.pause(0.001)

    def _reset_buttons():
        for b in _all_buttons:
            _set_button(b, "lightgray")

    def _get_voxels(mask, z):
        nx, ny = mask.shape[0], mask.shape[1]
        return [(x, y) for x in range(nx) for y in range(ny) if mask[x, y, z]]

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(event):
        slice_slider.reset()
        echo_slider.reset()
        mode["value"] = None
        _reset_buttons()
        fig.canvas.draw_idle()

    button_reset.on_clicked(reset)

    # ── Button callbacks ──────────────────────────────────────────────────────

    def make_callback(button):  # noqa: C901  (complexity is inherent here)
        def toggle_mode(event):
            _reset_buttons()

            # Keep Fit button highlighted when another button is active
            if mode["value"] == "fit" and button is not button_fit:
                _set_button(button_fit, "#7bff23")

            # ── Fit (toggle) ──────────────────────────────────────────────────
            if button is button_fit:
                if mode["value"] == "fit":
                    mode["value"] = None
                else:
                    mode["value"] = "fit"
                    _set_button(button_fit, "#7bff23")
                fig.canvas.draw_idle()
                return

            # ── Mono Map ──────────────────────────────────────────────────────
            if button is button_mono_mapping:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("mono", z)

                if cache_key in cache:
                    t2_mono, t2_off, t2_cfix, c_fixed, t_elapsed = cache[cache_key]
                    print(f"[Cache] Mono z={z}")
                else:
                    _set_button(button, "#ff8c00")
                    voxels = _get_voxels(mask, z)
                    nx, ny = data.shape[0], data.shape[1]
                    t2_mono = np.full((nx, ny), np.nan)
                    t2_off  = np.full((nx, ny), np.nan)
                    t2_cfix = np.full((nx, ny), np.nan)

                    t_start = time.time()

                    results = run_parallel(_fit_voxel_mono, voxels, data, te, z)
                    for x, y, v_mono, v_off in results:
                        t2_mono[x, y] = v_mono
                        t2_off[x, y]  = v_off

                    c_vals = []
                    results_c = run_parallel(_fit_voxel_noise, voxels, data, te, z)
                    for x, y, c_mono, _ in results_c:
                        if np.isfinite(c_mono):
                            c_vals.append(c_mono)
                    c_fixed = float(np.median(c_vals)) if c_vals else 0.0
                    print(f"Fixed C estimate: {c_fixed:.1f}")

                    results_cf = run_parallel(
                        _fit_voxel_mono_cfix, voxels, data, te, z, c_fixed=c_fixed
                    )
                    for x, y, v_cfix in results_cf:
                        t2_cfix[x, y] = v_cfix

                    t_elapsed = time.time() - t_start
                    print(f"Mono mapping done in {t_elapsed:.1f} s")
                    cache[cache_key] = (t2_mono, t2_off, t2_cfix, c_fixed, t_elapsed)

                _set_button(button, "#7bff23")
                fig_m, axes = plt.subplots(3, 2, figsize=(10, 12))
                fig_m.canvas.manager.set_window_title("Mono Mapping")

                for row, arr, title in [
                    (0, t2_mono, "T2 mono (ms)"),
                    (1, t2_off,  "T2 mono+offset (ms)"),
                    (2, t2_cfix, f"T2 mono+offset fixed C={c_fixed:.1f} (ms)"),
                ]:
                    finite_vals = arr[np.isfinite(arr)]
                    vmin = np.percentile(finite_vals, 2)  if len(finite_vals) > 0 else None
                    vmax = np.percentile(finite_vals, 98) if len(finite_vals) > 0 else None

                    im = axes[row, 0].imshow(
                        arr, cmap="hot", origin="upper",
                        interpolation="nearest", vmin=vmin, vmax=vmax,
                    )
                    axes[row, 0].set_title(title)
                    axes[row, 0].axis("off")
                    fig_m.colorbar(im, ax=axes[row, 0], fraction=0.046)
                    _use_array_coords(axes[row, 0], arr)

                    if len(finite_vals) > 0:
                        mu = np.mean(finite_vals)
                        axes[row, 1].hist(finite_vals, bins=60, color="steelblue", density=True)
                        axes[row, 1].axvline(mu, linestyle="--", color="cyan",
                                             label=f"µ={mu:.1f} ms")
                        axes[row, 1].legend()
                    axes[row, 1].set_title(f"Distribution — {title}")
                    axes[row, 1].set_xlabel("T2 (ms)")
                    axes[row, 1].set_ylabel("Density")

                plt.tight_layout()
                plt.show()

            # ── Bi Map ────────────────────────────────────────────────────────
            elif button is button_bi_mapping:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("bi", z)

                if cache_key in cache:
                    t2c_bi, t2l_bi, t2c_bioff, t2l_bioff, t2_eff_bi, t2_eff_bioff, t_elapsed = cache[cache_key]
                    print(f"[Cache] Bi z={z}")
                else:
                    _set_button(button, "#ff8c00")
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
                    for x, y, v_t2c, v_t2l, v_t2c_off, v_t2l_off, v_eff_bi, v_eff_bioff in results:
                        t2c_bi[x, y]       = v_t2c
                        t2l_bi[x, y]       = v_t2l
                        t2c_bioff[x, y]    = v_t2c_off
                        t2l_bioff[x, y]    = v_t2l_off
                        t2_eff_bi[x, y]    = v_eff_bi
                        t2_eff_bioff[x, y] = v_eff_bioff

                    t_elapsed = time.time() - t_start
                    print(f"Bi mapping done in {t_elapsed:.1f} s")
                    cache[cache_key] = (
                        t2c_bi, t2l_bi, t2c_bioff, t2l_bioff,
                        t2_eff_bi, t2_eff_bioff, t_elapsed,
                    )

                _set_button(button, "#7bff23")
                fig_b, axes = plt.subplots(3, 2, figsize=(10, 12))
                fig_b.canvas.manager.set_window_title("Bi Mapping")

                for ax_b, arr, title in [
                    (axes[0, 0], t2c_bi,       "T2c bi (ms)"),
                    (axes[0, 1], t2l_bi,        "T2l bi (ms)"),
                    (axes[1, 0], t2c_bioff,     "T2c bi+offset (ms)"),
                    (axes[1, 1], t2l_bioff,     "T2l bi+offset (ms)"),
                    (axes[2, 0], t2_eff_bi,     "T2 effective bi (ms)"),
                    (axes[2, 1], t2_eff_bioff,  "T2 effective bi+offset (ms)"),
                ]:
                    finite_vals = arr[np.isfinite(arr)]
                    if len(finite_vals) > 0:
                        vmin = np.percentile(finite_vals, 2)
                        vmax = 100.0 if "T2l" in title else np.percentile(finite_vals, 98)
                    else:
                        vmin, vmax = None, None

                    im = ax_b.imshow(
                        arr, cmap="hot", origin="upper",
                        interpolation="nearest", vmin=vmin, vmax=vmax,
                    )
                    ax_b.set_title(title)
                    ax_b.axis("off")
                    fig_b.colorbar(im, ax=ax_b, fraction=0.046)
                    _use_array_coords(ax_b, arr)

                plt.tight_layout()
                plt.show()

            # ── Utils ─────────────────────────────────────────────────────────
            elif button is button_utils:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("utils", z)

                if cache_key in cache:
                    (aic_map, i0_voxelwise, i0_global, r2_map, rmse_map,
                     f_bi_map, f_bioff_map, t2_eff_map,
                     global_best, model_counts, t_elapsed) = cache[cache_key]
                    print(f"[Cache] Utils z={z}")
                else:
                    _set_button(button, "#ff8c00")
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
                    print(f"Utils done in {t_elapsed:.1f} s | global: {global_best}")
                    print(f"Model counts: {model_counts}")
                    cache[cache_key] = (
                        aic_map, i0_voxelwise, i0_global, r2_map, rmse_map,
                        f_bi_map, f_bioff_map, t2_eff_map,
                        global_best, model_counts, t_elapsed,
                    )

                _set_button(button, "#7bff23")
                fig_u, axes = plt.subplots(3, 2, figsize=(10, 12))
                fig_u.suptitle(
                    f"Utils — z={z} | {global_best} | {t_elapsed:.1f} s", fontsize=13
                )
                fig_u.canvas.manager.set_window_title("Utils")

                cmap_aic = ListedColormap(_AIC_COLORS)
                axes[0, 0].imshow(
                    aic_map, cmap=cmap_aic, vmin=-0.5, vmax=3.5,
                    origin="upper", interpolation="nearest",
                )
                axes[0, 0].set_title("Best model (AIC)")
                axes[0, 0].axis("off")
                patches = [
                    mpatches.Patch(color=_AIC_COLORS[i], label=_AIC_LABELS[i])
                    for i in range(4)
                ]
                axes[0, 0].legend(handles=patches, loc="lower right",
                                  fontsize=7, framealpha=0.7)

                vol_slice  = data[:, :, z, 0].astype(float)
                vol_masked = (
                    np.where(mask[:, :, z], vol_slice, np.nan)
                    if mask is not None else vol_slice
                )
                vol_smooth = gaussian_filter(np.nan_to_num(vol_masked), sigma=1.5)
                vol_smooth = (
                    np.where(mask[:, :, z], vol_smooth, np.nan)
                    if mask is not None else vol_smooth
                )
                im_s = axes[0, 1].imshow(
                    vol_smooth, cmap="viridis", origin="upper", interpolation="nearest"
                )
                axes[0, 1].set_title("Smoothed image (σ=1.5, echo 0)")
                axes[0, 1].axis("off")
                fig_u.colorbar(im_s, ax=axes[0, 1], fraction=0.046)

                for ax_u, arr, title in [
                    (axes[1, 0], f_bi_map,    "Water fraction f (bi)\nrestricted vs free"),
                    (axes[1, 1], f_bioff_map, "Water fraction f (bi+offset)\nrestricted vs free"),
                ]:
                    im_f = ax_u.imshow(
                        arr, cmap="RdBu_r", vmin=0, vmax=1,
                        origin="upper", interpolation="nearest",
                    )
                    ax_u.set_title(title)
                    ax_u.axis("off")
                    fig_u.colorbar(im_f, ax=ax_u, fraction=0.046)

                finite_i0 = i0_voxelwise[np.isfinite(i0_voxelwise)]
                vmin_i0 = np.percentile(finite_i0, 1)  if len(finite_i0) > 0 else None
                vmax_i0 = np.percentile(finite_i0, 99) if len(finite_i0) > 0 else None
                im_i0v = axes[2, 0].imshow(
                    i0_voxelwise, cmap="viridis",
                    vmin=vmin_i0, vmax=vmax_i0,
                    origin="upper", interpolation="nearest",
                )
                axes[2, 0].set_title("I0 voxel-wise")
                axes[2, 0].axis("off")
                fig_u.colorbar(im_i0v, ax=axes[2, 0], fraction=0.046)
                _use_array_coords(axes[2, 0], i0_voxelwise)

                im_i0g = axes[2, 1].imshow(
                    i0_global, cmap="viridis", origin="upper", interpolation="nearest"
                )
                axes[2, 1].set_title(f"I0 global ({global_best})")
                axes[2, 1].axis("off")
                fig_u.colorbar(im_i0g, ax=axes[2, 1], fraction=0.046)

                plt.tight_layout()
                plt.show()

            # ── Noise ─────────────────────────────────────────────────────────
            elif button is button_noise:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("noise", z)

                if cache_key in cache:
                    c_mono_off, c_bi_off, t_elapsed = cache[cache_key]
                    print(f"[Cache] Noise z={z}")
                else:
                    _set_button(button, "#ff8c00")
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
                    print(f"Noise maps done in {t_elapsed:.1f} s")
                    cache[cache_key] = (c_mono_off, c_bi_off, t_elapsed)

                estimate_noise(data)
                _set_button(button, "#7bff23")

                fig_n, axes = plt.subplots(3, 2, figsize=(10, 12))
                fig_n.canvas.manager.set_window_title("Noise")

                plot_histogram(data, ax=axes[0, 0])

                if mask is not None:
                    axes[0, 1].imshow(
                        mask[:, :, z], cmap="gray",
                        origin="upper", interpolation="nearest",
                    )
                axes[0, 1].set_title(f"Tissue mask — z={z}")
                axes[0, 1].axis("off")

                for row, arr, title in [
                    (1, c_mono_off, "Offset C map (mono+offset)"),
                    (2, c_bi_off,   "Offset C map (bi+offset)"),
                ]:
                    im_c = axes[row, 0].imshow(
                        arr, cmap="plasma", origin="upper", interpolation="nearest"
                    )
                    axes[row, 0].set_title(title)
                    axes[row, 0].axis("off")
                    fig_n.colorbar(im_c, ax=axes[row, 0], fraction=0.046)

                    c_vals_all = arr[np.isfinite(arr)]
                    # curve_fit rarely returns an exact 0.0 when hitting the
                    # lower bound (C >= 0) — it converges to a tiny residual
                    # instead (e.g. 1e-4). A strict "!= 0" filter misses
                    # these boundary solutions, so use a small threshold.
                    c_floor = 1.0  # ms/intensity units — well below any real offset
                    c_vals = c_vals_all[c_vals_all > c_floor]
                    n_excluded = len(c_vals_all) - len(c_vals)

                    if len(c_vals) > 0:
                        mu_c = np.mean(c_vals)
                        axes[row, 1].hist(c_vals, bins=50, color="steelblue", density=True)
                        axes[row, 1].axvline(mu_c, linestyle="--", color="cyan",
                                             label=f"µ={mu_c:.1f}")
                        axes[row, 1].legend()

                        if n_excluded > 0 and len(c_vals_all) > 0:
                            mu_with_zeros = np.mean(c_vals_all)
                            pct = 100 * n_excluded / len(c_vals_all)
                            print(
                                f"[Noise] {title}: {n_excluded}/{len(c_vals_all)} "
                                f"({pct:.1f}%) voxel(s) at the C>=0 lower bound "
                                f"excluded — mean {mu_with_zeros:.2f} → {mu_c:.2f}"
                            )
                    axes[row, 1].set_title(f"Distribution — {title}")
                    axes[row, 1].set_xlabel("C")
                    axes[row, 1].set_ylabel("Density")

                plt.tight_layout()
                plt.show()

            # ── Error ─────────────────────────────────────────────────────────
            elif button is button_error:
                mode["value"] = None
                z = int(slice_slider.val)
                cache_key = ("error", z)

                if cache_key in cache:
                    (r2_map, rmse_map, r2_glob_map, rmse_glob_map,
                     global_best, t_elapsed) = cache[cache_key]
                    print(f"[Cache] Error z={z}")
                else:
                    _set_button(button, "#ff8c00")
                    voxels = _get_voxels(mask, z)
                    nx, ny = data.shape[0], data.shape[1]
                    r2_map        = np.full((nx, ny), np.nan)
                    rmse_map      = np.full((nx, ny), np.nan)
                    r2_glob_map   = np.full((nx, ny), np.nan)
                    rmse_glob_map = np.full((nx, ny), np.nan)
                    all_fitted    = {}
                    counts        = {"mono": 0, "mono+offset": 0, "bi": 0, "bi+offset": 0}

                    t_start = time.time()
                    results = run_parallel(_fit_voxel_error, voxels, data, te, z)
                    for x, y, best, r2_vw, rmse_vw, fitted_map in results:
                        r2_map[x, y]   = r2_vw
                        rmse_map[x, y] = rmse_vw
                        counts[best]  += 1
                        all_fitted[(x, y)] = fitted_map

                    global_best = max(counts, key=counts.get)
                    print(f"Global model: {global_best} | Counts: {counts}")

                    for x, y in voxels:
                        f = all_fitted[(x, y)][global_best]
                        signal = data[x, y, z, :]
                        r2_glob_map[x, y]   = compute_r2(signal, f)   if f is not None else np.nan
                        rmse_glob_map[x, y] = compute_rmse(signal, f) if f is not None else np.nan

                    t_elapsed = time.time() - t_start
                    print(f"Error done in {t_elapsed:.1f} s")
                    cache[cache_key] = (
                        r2_map, rmse_map, r2_glob_map, rmse_glob_map, global_best, t_elapsed
                    )

                _set_button(button, "#7bff23")
                fig_e, axes = plt.subplots(2, 3, figsize=(14, 8))
                fig_e.suptitle(
                    f"Error — z={z} | global={global_best} | {t_elapsed:.1f} s", fontsize=13
                )
                fig_e.canvas.manager.set_window_title("Error")

                for row, (arr_vw, arr_glob, label, cmap_e, vmin_e, vmax_e) in enumerate([
                    (rmse_map, rmse_glob_map, "RMSE", "hot_r",  None, None),
                    (r2_map,   r2_glob_map,   "R²",  "RdYlGn", 0,    1   ),
                ]):
                    for col, (arr, title) in enumerate([
                        (arr_vw,   f"{label} voxel-wise"),
                        (arr_glob, f"{label} global ({global_best})"),
                    ]):
                        finite_v = arr[np.isfinite(arr)]
                        vm_min = vmin_e if vmin_e is not None else (
                            np.percentile(finite_v, 2)  if len(finite_v) > 0 else None)
                        vm_max = vmax_e if vmax_e is not None else (
                            np.percentile(finite_v, 98) if len(finite_v) > 0 else None)

                        im_e = axes[row, col].imshow(
                            arr, cmap=cmap_e, vmin=vm_min, vmax=vm_max,
                            origin="upper", interpolation="nearest",
                        )
                        axes[row, col].set_title(title)
                        axes[row, col].axis("off")
                        fig_e.colorbar(im_e, ax=axes[row, col], fraction=0.046)
                        _use_array_coords(axes[row, col], arr)

                    ax_dist = axes[row, 2]
                    for arr, label_d, color in [
                        (arr_vw,   "voxel-wise", "steelblue"),
                        (arr_glob, "global",     "tomato"),
                    ]:
                        finite_v = arr[np.isfinite(arr)]
                        if len(finite_v) > 0:
                            mu = np.mean(finite_v)
                            fmt = f".3f" if label == "R²" else ".1f"
                            ax_dist.hist(finite_v, bins=60, density=True,
                                         alpha=0.5, color=color,
                                         label=f"{label_d} µ={mu:{fmt}}")
                            ax_dist.axvline(mu, color=color, linestyle="--", linewidth=1.5)
                    ax_dist.set_title(f"Distribution {label}")
                    ax_dist.set_xlabel(label)
                    ax_dist.set_ylabel("Density")
                    ax_dist.legend(fontsize=8)

                plt.tight_layout()
                plt.show()

            # ── 3D View ───────────────────────────────────────────────────────
            elif button is button_3d:
                mode["value"] = None
                _set_button(button, "#ff8c00")

                vol = data[:, :, :, 0].astype(float)
                vol_masked = np.where(mask, vol, np.nan) if mask is not None else vol.copy()

                finite_vals = vol_masked[np.isfinite(vol_masked)]
                if len(finite_vals) == 0:
                    print("[3D] No valid values found.")
                    _set_button(button, "lightgray")
                    return

                iso_min = np.percentile(finite_vals, 30)
                iso_max = np.percentile(finite_vals, 98)

                nx, ny, nz = vol_masked.shape
                x_idx, y_idx, z_idx = np.meshgrid(
                    np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
                )
                vol_plot = np.nan_to_num(vol_masked, nan=0.0)

                fig_3d = go.Figure(data=go.Volume(
                    x=x_idx.flatten(),
                    y=y_idx.flatten(),
                    z=z_idx.flatten(),
                    value=vol_plot.flatten(),
                    isomin=iso_min,
                    isomax=iso_max,
                    opacity=0.15,
                    surface_count=20,
                    colorscale="Viridis",
                    caps=dict(x_show=False, y_show=False, z_show=False),
                ))
                fig_3d.update_layout(
                    title=f"3D view — echo 0 | threshold {iso_min:.0f}–{iso_max:.0f}",
                    scene=dict(
                        xaxis_title="X", yaxis_title="Y", zaxis_title="Z (slice)",
                        aspectmode="data",
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                _set_button(button, "#7bff23")
                fig_3d.show()
                fig.canvas.draw_idle()

            # ── Export ────────────────────────────────────────────────────────
            elif button is button_export:
                mode["value"] = None
                z = int(slice_slider.val)
                _set_button(button, "#ff8c00")
                fig.canvas.draw_idle()
                plt.pause(0.001)  # force le rendu orange avant l'écriture disque

                out_path = Path("exports") / f"T2_slice_z{z}"

                # (cache_key, index -> nom de colonne) pour chaque carte 2D déjà calculée
                export_names = {
                    ("mono", z): {
                        0: "T2_mono", 1: "T2_mono_offset", 2: "T2_mono_cfix",
                    },
                    ("bi", z): {
                        0: "T2c_bi", 1: "T2l_bi", 2: "T2c_bi_offset",
                        3: "T2l_bi_offset", 4: "T2eff_bi", 5: "T2eff_bi_offset",
                    },
                    ("noise", z): {0: "C_mono_offset", 1: "C_bi_offset"},
                    ("error", z): {0: "R2_voxelwise", 1: "RMSE_voxelwise"},
                    ("utils", z): {
                        0: "AIC_best_model_index", 1: "I0_voxelwise", 2: "I0_global",
                        3: "R2_map", 4: "RMSE_map", 5: "f_bi", 6: "f_bi_offset",
                        7: "T2_effective",
                    },
                }

                # Construit un seul dict {nom_colonne: array 2D} à partir de
                # tout ce qui est déjà en cache pour cette coupe.
                maps_dict = {}
                for cache_key, names in export_names.items():
                    if cache_key not in cache:
                        continue
                    values = cache[cache_key][:-1]  # dernier élément = t_elapsed
                    for idx, arr in enumerate(values):
                        if idx in names and isinstance(arr, np.ndarray) and arr.ndim == 2:
                            maps_dict[names[idx]] = arr

                if not maps_dict:
                    print(
                        "[Export] Rien à exporter — calcule d'abord une carte "
                        "(Mono Map, Bi Map, Noise, Error ou Utils) pour cette coupe."
                    )
                    _set_button(button, "#C44E52")
                else:
                    # Masque tissulaire (grain) de cette coupe : ne garde que
                    # les voxels non nuls / non bruit de fond.
                    slice_mask = mask[:, :, z] if mask is not None else None
                    written_path, n_rows = export_table(
                        maps_dict, z, out_path, mask=slice_mask
                    )

                    # Self-check : relit le CSV et compare chaque valeur à
                    # l'array en mémoire, avec une tolérance cohérente avec
                    # les 6 décimales écrites.
                    col_names = list(maps_dict.keys())
                    mismatch = False
                    with open(written_path, newline="") as f:
                        reader = csv.reader(f, delimiter=";")
                        header = next(reader)
                        col_idx = {name: header.index(name) for name in col_names}
                        for row in reader:
                            rx, ry = int(row[1]), int(row[2])
                            for name in col_names:
                                expected = maps_dict[name][rx, ry]
                                got = float(row[col_idx[name]].replace(",", "."))
                                if not np.isclose(expected, got, atol=1e-6, equal_nan=True):
                                    mismatch = True
                                    print(
                                        f"[Export] MISMATCH detected in {written_path} "
                                        f"at x={rx}, y={ry}, column='{name}' !"
                                    )
                                    break
                            if mismatch:
                                break

                    if mismatch:
                        _set_button(button, "#C44E52")
                    else:
                        print(
                            f"[Export] {n_rows} voxel(s) × {len(col_names)} "
                            f"colonne(s) exporté(s) et vérifié(s) dans "
                            f"{written_path}"
                        )
                        _set_button(button, "#7bff23")

            fig.canvas.draw_idle()

        return toggle_mode

    button_fit.on_clicked(make_callback(button_fit))
    button_mono_mapping.on_clicked(make_callback(button_mono_mapping))
    button_bi_mapping.on_clicked(make_callback(button_bi_mapping))
    button_utils.on_clicked(make_callback(button_utils))
    button_noise.on_clicked(make_callback(button_noise))
    button_error.on_clicked(make_callback(button_error))
    button_3d.on_clicked(make_callback(button_3d))
    button_export.on_clicked(make_callback(button_export))

    # ── Voxel click — per-voxel fit plot ─────────────────────────────────────

    def onclick(event):
        toolbar = event.canvas.toolbar
        ui_axes = [
            button_fit.ax, button_mono_mapping.ax, button_bi_mapping.ax,
            button_utils.ax, button_noise.ax, button_reset.ax,
            axecho, axslice,
        ]
        if event.inaxes in ui_axes:
            return
        if toolbar is not None and toolbar.mode != "":
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

        p_mono,  f_mono,  _ = fit_mono(te, signal)
        p_off,   f_off,   _ = fit_mono_offset(te, signal)
        p_bi,    f_bi,    _ = fit_bi(te, signal)
        p_bioff, f_bioff, _ = fit_bi_offset(te, signal)

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
            "bi": aic_bi,     "bi+offset":   aic_bioff,
        }
        best_model = min(aic_dict, key=aic_dict.get)

        # ── Terminal summary ──────────────────────────────────────────────────
        sep = "─" * 58
        print(f"\n{sep}")
        print(f"  Voxel ({x}, {y}) | slice z={int(slice_slider.val)}")
        print(sep)
        print(f"  {'Model':<18} {'I0':>8} {'T2':>7} {'AIC':>8} {'R²':>6} {'RMSE':>8}")
        print(sep)

        def _row(name, params, aic, r2, rmse, extra=""):
            if params is None:
                return f"  {name:<18} {'—':>8} {'—':>7} {'—':>8} {'—':>6} {'—':>8}"
            I0 = params.get("I0", float("nan"))
            T2 = params.get("T2", params.get("T2c", float("nan")))
            star = " ★" if name == best_model else ""
            return (
                f"  {name:<18} {I0:>8.1f} {T2:>7.2f} {aic:>8.1f} "
                f"{r2:>6.3f} {rmse:>8.1f}{extra}{star}"
            )

        print(_row("mono",        p_mono,  aic_mono,  r2_mono,  rmse_mono))
        print(_row("mono+offset", p_off,   aic_off,   r2_off,   rmse_off,
                   f"  C={p_off['C']:.1f}" if p_off else ""))
        print(_row("bi", p_bi, aic_bi, r2_bi, rmse_bi,
                   f"  f={p_bi['f']:.2f} T2l={p_bi['T2l']:.1f}" if p_bi else ""))
        print(_row("bi+offset", p_bioff, aic_bioff, r2_bioff, rmse_bioff,
                   (f"  f={p_bioff['f']:.2f} T2l={p_bioff['T2l']:.1f}"
                    f" C={p_bioff['C']:.1f}") if p_bioff else ""))
        print(sep)
        print(f"  Best model (AIC): {best_model}")
        print(sep)

        # ── Decay curve plot ──────────────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(te, signal, "o", color="black", label="data", zorder=5)

        model_fits = [
            ("mono",        f_mono,  aic_mono),
            ("mono+offset", f_off,   aic_off),
            ("bi",          f_bi,    aic_bi),
            ("bi+offset",   f_bioff, aic_bioff),
        ]
        for name, fitted, aic in model_fits:
            if fitted is None:
                continue
            is_best  = (name == best_model)
            lw       = 2.5 if is_best else 1.2
            label    = f"★ {name} (AIC={aic:.1f})" if is_best else f"{name} (AIC={aic:.1f})"
            ax2.plot(te, fitted, color=_FIT_COLORS[name],
                     linewidth=lw, label=label, zorder=4 if is_best else 3)

        ax2.set_xlabel("Echo time (ms)")
        ax2.set_ylabel("Signal intensity")
        ax2.set_title(f"Voxel ({x}, {y}) — slice z={int(slice_slider.val)}")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fig2.canvas.draw_idle()
        fig2.show()
        plt.pause(0.001)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()