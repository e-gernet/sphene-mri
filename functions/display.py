"""
Interactive matplotlib-based viewer for MRI T2 relaxometry data.

This module provides a single entry point :func:`display_slice` that opens
a matplotlib figure with slice/echo navigation sliders and a set of analysis
buttons. Every analysis button reads the same per-slice result (the mono and bi
models fitted on all the voxels of the slice, see
:func:`functions.mapping.fit_slice`), computed once and then cached, so the
other buttons and repeated clicks are instantaneous.

Buttons
-------
Fit
    Click any voxel to fit the mono and bi models by maximum likelihood and plot
    the decay curves. The model the data justify (lowest AICc, see
    :func:`functions.model.fit_both`) is drawn thicker with a star.
Mono Map
    T2 map of the mono-exponential fit for the current slice.
Bi Map
    T2c, T2l, water fraction f and effective T2 of the bi-exponential fit, shown
    only in the voxels where the data justify two components.
Utils
    Model kept in each voxel, evidence for bi (delta AICc), water fraction, I0
    maps and a smoothed reference image.
Error
    Reduced chi-square, RMSE and R² maps: how well the kept model (and one single
    model for the whole slice) explains the data.
Noise
    Background histogram against the Rayleigh law of pure noise, tissue mask,
    noise level echo by echo and mean residual of each echo.
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
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, LassoSelector, Slider
from scipy.ndimage import gaussian_filter

from .io import export_rois, export_table
from .mapping import fit_slice, map_stats, plot_histogram, slice_summary, stats_header, stats_line
from .model import DELTA_AICC, MODELS, fit_both, fit_quality
from .utils import check_rice_noise, estimate_noise_auto, sigma_per_echo

# One colour per model, in the order of ``MODELS`` (maps and decay-curve plots)
_MODEL_COLORS = {"mono": "#4C72B0", "bi": "#C44E52"}

# Map of the per-slice fit  ->  column name in the exported CSV
_EXPORT_COLUMNS = {
    "T2_mono": "T2_mono",
    "T2c_bi": "T2c_bi", "T2l_bi": "T2l_bi", "f_bi": "f_bi", "T2eff_bi": "T2eff_bi",
    "best": "bi_kept",             # 1 = the data justify bi, 0 = mono
    "delta_aicc": "delta_AICc",    # evidence for bi
    "I0_best": "I0", "chi2_best": "chi2_red", "r2_best": "R2", "rmse_best": "RMSE",
    "limit_best": "T2_at_limit",   # 1 = a T2 ended on a limit of the search range
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


def _percentile_range(arr, low=2, high=98):
    """Colour range covering the central part of the finite values of ``arr``.

    A few extreme voxels (for instance a T2 on a limit of the search range) would
    otherwise stretch the colour scale and flatten everything else.
    Returns ``(None, None)`` if ``arr`` holds no finite value.
    """
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return None, None
    return float(np.percentile(vals, low)), float(np.percentile(vals, high))


def _draw_map(fig, ax, arr, title, cmap, vmin=None, vmax=None, norm=None, tissue=None):
    """Show a 2-D map with its colour bar and the array-index coordinate readout.

    ``tissue`` (bool array) draws in light grey the tissue voxels that have no
    value on the map (e.g. the bi map where mono is kept), to tell them apart
    from the background.
    """
    if tissue is not None:
        ax.imshow(np.where(tissue, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3,
                  origin="upper", interpolation="nearest")
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                   origin="upper", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _use_array_coords(ax, arr)
    return im


def _draw_distribution(ax, arr, title, xlabel):
    """Histogram of the finite values of ``arr`` with their mean.

    The horizontal range leaves out the 0.5 % most extreme values on each side,
    so a handful of outliers does not squeeze the histogram; the mean is computed
    on all the values.
    """
    vals = arr[np.isfinite(arr)]
    if vals.size:
        low, high = np.percentile(vals, [0.5, 99.5])
        mu = float(np.mean(vals))
        ax.hist(vals, bins=60, range=(low, high), color="steelblue", density=True)
        ax.axvline(mu, linestyle="--", color="cyan", label=f"µ={mu:.1f}")
        ax.legend()
    ax.set_title(f"Distribution — {title}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")


def _print_map_stats(entries, where):
    """Print n, mean, SD, median, min and max of maps in the console.

    ``entries`` is a list of ``(label, array, decimals)``; ``where`` says what
    the values come from ("slice 8", "zone of 64 voxel(s), ..."). Same layout
    as ``quickstats.py``.
    """
    width = max(len(label) for label, _, _ in entries) + 2
    print(f"[Stats] {where}")
    print(stats_header("map", width))
    for label, arr, decimals in entries:
        print(stats_line(label, map_stats(arr), decimals, width))


def _count_from_one(slider):
    """Show the value of an index slider counted from 1 (the stored value stays 0-based)."""
    def refresh(val):
        slider.valtext.set_text(str(int(val) + 1))
    slider.on_changed(refresh)
    refresh(slider.val)


def display_slice(data, te, mask=None, voxel_dims=None):
    """Launch the interactive T2 relaxometry viewer.

    Opens a matplotlib figure showing the MRI volume with slice and echo
    navigation sliders. The analysis buttons all read one per-slice fit (mono and
    bi on every voxel), computed at the first click and cached afterwards.

    Parameters
    ----------
    data : np.ndarray of shape (nx, ny, nz, n_te)
        4-D MRI data array (float64).
    te : array-like of shape (n_te,)
        Echo times in milliseconds, in acquisition order.
    mask : np.ndarray of shape (nx, ny, nz), dtype bool, optional
        Binary tissue mask. If ``None``, all voxels are processed (slow).
    voxel_dims : tuple of 3 floats, optional
        Real-world voxel size ``(dx, dy, dz)`` in mm, e.g. from
        ``img.header.get_zooms()[:3]``. If provided, the CSV export adds
        ``x_mm``/``y_mm``/``z_mm`` columns alongside the raw voxel-index
        ``x``/``y``/``slice`` columns. If ``None``, export stays
        index-only (previous behaviour).

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
    cache_bg = {}   # background mask of the whole volume, computed once (Noise histogram)
    # Separate from `mask`: flagged voxels stay IN the tissue mask (still
    # fitted/exported) but are tagged, e.g. capillary vs sample.
    capillary_mask = np.zeros_like(mask, dtype=bool) if mask is not None else None
    # Every "Zone fit" region drawn in this session: voxel coordinates + fit
    # results, written next to the map CSV by the Export button.
    roi_list = []
    # Tissue mask WITHOUT any "Fit only" restriction. "Exclude" edits both it and
    # `mask`; "Fit only" always restarts from it, so successive "Fit only" zones
    # are independent of each other (but excluded voxels stay excluded).
    mask_base = mask.copy() if mask is not None else None
    # Automatic mask as first computed, kept for the "Restore mask" button.
    mask_original = mask.copy() if mask is not None else None

    # ── Main figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, bottom=0.2)

    img  = ax.imshow(data[:, :, z, t], cmap="viridis")
    fig.colorbar(img, ax=ax)
    _use_array_coords(ax, data[:, :, z, t])

    axecho = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    echo_slider = Slider(
        ax=axecho, label="Echo (ms)",
        valstep=1, valmin=0, valmax=data.shape[3] - 1, valinit=t,
    )
    _count_from_one(echo_slider)

    axslice = fig.add_axes([0.155, 0.25, 0.0225, 0.63])
    slice_slider = Slider(
        ax=axslice, label="Slice",
        valstep=1, valmin=0, valmax=data.shape[2] - 1, valinit=z,
        orientation="vertical",
    )
    _count_from_one(slice_slider)

    # ── SNR readout (console) ───────────────────────────────────────────────
    # σ is measured once for the whole acquisition, then SNR is reported
    # per slice over the tissue mask whenever the slice changes.
    _, _, bg_otsu, _ = estimate_noise_auto(np.max(data, axis=-1), verbose=False)
    sigma_echo = sigma_per_echo(data, bg_otsu)      # one estimate per echo
    sigma_noise = float(np.median(sigma_echo))      # the value the fits use
    check_rice_noise(data, bg_otsu)                 # is the background really Rician?
    last_snr_slice = {"z": None}

    def _print_snr(z):
        if mask is None or not mask[:, :, z].any():
            print(f"[SNR] slice {z + 1}: no tissue voxel in the mask.")
            return
        snr = np.median(data[:, :, z, 0][mask[:, :, z]]) / sigma_noise
        print(f"[SNR] slice {z + 1} : SNR = {snr:.1f}  "
              f"(median tissue signal at echo 1 / background noise)")

    _print_snr(z)
    last_snr_slice["z"] = z

    def update(val):
        img.set_data(data[:, :, slice_slider.val, echo_slider.val])
        _draw_mask_overlay()
        if last_snr_slice["z"] != int(slice_slider.val):
            last_snr_slice["z"] = int(slice_slider.val)
            _print_snr(last_snr_slice["z"])
        fig.canvas.draw_idle()

    echo_slider.on_changed(update)
    slice_slider.on_changed(update)

    # ── Mask overlay (always available, independent of Select mode) ────────
    # Lets you see what the mask actually covers before drawing a selection
    # — otherwise it's easy to draw over background/noise voxels that the
    # fit silently ignores (mask is applied on top of every selection).
    from matplotlib.widgets import CheckButtons as _CheckButtonsMaskOverlay

    ax_mask_overlay = fig.add_axes([0.01, 0.945, 0.13, 0.045])
    check_mask_overlay = _CheckButtonsMaskOverlay(ax_mask_overlay, ["Show mask"], [False])
    check_mask_overlay.labels[0].set_fontsize(7)

    _mask_overlay = {"artist": None}

    def _draw_mask_overlay():
        if _mask_overlay["artist"] is not None:
            artist = _mask_overlay["artist"]
            # matplotlib >= 3.10 removed QuadContourSet.collections: the
            # contour set itself is now directly removable. Older versions
            # still expose a list of Collection sub-artists that each need
            # removing individually. Support both.
            if hasattr(artist, "collections"):
                for coll in artist.collections:
                    coll.remove()
            else:
                artist.remove()
            _mask_overlay["artist"] = None
        if mask is not None and check_mask_overlay.get_status()[0]:
            z = int(slice_slider.val)
            _mask_overlay["artist"] = ax.contour(
                mask[:, :, z].astype(float), levels=[0.5],
                colors="red", linewidths=1.2, origin="lower",
                # NOTE: counter-intuitive but verified empirically —
                # ax.contour()'s `origin` parameter does not use the same
                # convention as ax.imshow()'s. origin="upper" here produces
                # a vertical mirror of the mask relative to the image
                # (flipped around row n/2), even though "upper" is what
                # matches imshow. origin="lower" is the one that actually
                # aligns row-for-row with imshow's default display. This
                # only affected the visual overlay — mask_rician, the
                # fits, and the CSV export never call contour() and were
                # never affected.
            )

    def _on_mask_overlay_toggle(label):
        _draw_mask_overlay()
        fig.canvas.draw_idle()

    check_mask_overlay.on_clicked(_on_mask_overlay_toggle)

    # ── Selection controls (shape / action / propagation) ───────────────────
    # Hidden by default, shown only while "Select" mode is active — avoids
    # cluttering the figure the rest of the time.
    from matplotlib.widgets import (
        CheckButtons,
        EllipseSelector,
        RadioButtons,
        RectangleSelector,
    )

    ax_shape = fig.add_axes([0.01, 0.785, 0.13, 0.13])
    ax_shape.set_title("Shape", fontsize=8)
    radio_shape = RadioButtons(ax_shape, ("Lasso", "Box", "Circle"))
    for lbl in radio_shape.labels:
        lbl.set_fontsize(7)

    ax_action = fig.add_axes([0.01, 0.603, 0.13, 0.18])
    ax_action.set_title("Action", fontsize=8)
    radio_action = RadioButtons(
        ax_action, ("Zone fit", "Exclude", "Fit only", "Flag capillary", "Show signal"),
    )
    for lbl in radio_action.labels:
        lbl.set_fontsize(7)

    ax_propagate = fig.add_axes([0.01, 0.545, 0.13, 0.05])
    check_propagate = CheckButtons(ax_propagate, ["All slices"], [False])
    check_propagate.labels[0].set_fontsize(7)

    for a in (ax_shape, ax_action, ax_propagate):
        a.set_visible(False)
    radio_shape.active = False
    radio_action.active = False
    check_propagate.active = False

    # ── Buttons ───────────────────────────────────────────────────────────────
    # 11 buttons on one row: width + gap chosen so they all fit from x=0.05 to x=0.93
    _btn_gap = 0.004
    _btn_w = (0.88 - 10 * _btn_gap) / 11
    _btn_x = [0.05 + i * (_btn_w + _btn_gap) for i in range(11)]

    resetax            = plt.axes([_btn_x[10], 0.05, _btn_w, 0.04])
    ax_button_fit      = plt.axes([_btn_x[9], 0.05, _btn_w, 0.04])
    ax_button_bi       = plt.axes([_btn_x[8], 0.05, _btn_w, 0.04])
    ax_button_mono     = plt.axes([_btn_x[7], 0.05, _btn_w, 0.04])
    ax_button_utils    = plt.axes([_btn_x[6], 0.05, _btn_w, 0.04])
    ax_button_error    = plt.axes([_btn_x[5], 0.05, _btn_w, 0.04])
    ax_button_noise    = plt.axes([_btn_x[4], 0.05, _btn_w, 0.04])
    ax_button_3d       = plt.axes([_btn_x[3], 0.05, _btn_w, 0.04])
    ax_button_export   = plt.axes([_btn_x[2], 0.05, _btn_w, 0.04])
    ax_button_select   = plt.axes([_btn_x[0], 0.05, _btn_w, 0.04])
    ax_button_restore  = plt.axes([_btn_x[1], 0.05, _btn_w, 0.04])

    button_reset        = Button(resetax,         "Reset")
    button_fit          = Button(ax_button_fit,   "Fit")
    button_bi_mapping   = Button(ax_button_bi,    "Bi Map")
    button_mono_mapping = Button(ax_button_mono,  "Mono Map")
    button_utils        = Button(ax_button_utils, "Utils")
    button_error        = Button(ax_button_error, "Error")
    button_noise        = Button(ax_button_noise, "Noise")
    button_3d           = Button(ax_button_3d,    "3D View")
    button_export       = Button(ax_button_export, "Export")
    button_select       = Button(ax_button_select, "Select")
    button_restore      = Button(ax_button_restore, "Restore\nmask")

    _all_buttons = [
        button_fit, button_mono_mapping, button_bi_mapping,
        button_utils, button_error, button_noise, button_3d,
        button_export, button_select, button_restore, button_reset,
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

    # ── Selection (lasso / box / circle → exclude or flag) ──────────────────

    _selector = {"obj": None}

    def _grid_points(nx, ny):
        """(col, row) points matching the plot-coordinate convention used by
        matplotlib selectors, reshaped later to the (x, y) = (row, col)
        array convention used everywhere else (mask[x, y, z])."""
        rows, cols = np.mgrid[0:nx, 0:ny]
        return np.column_stack((cols.ravel(), rows.ravel())), (nx, ny)

    # ── Fit of the whole slice (mono + bi on every voxel), done once ─────────
    # Every map button reads this same result: the first click computes it,
    # the next ones (any button) use the cache.

    def _get_fit(z, button):
        """Per-slice fit result ``{"maps", "resid", "elapsed"}`` (cached), or None
        if the slice has no voxel in the mask."""
        key = ("fit", z)
        if key in cache:
            print(f"[Cache] Fit slice {z + 1}")
            return cache[key]
        voxels = _get_voxels(mask, z)
        if not voxels:
            print(f"[Fit] Slice {z + 1}: no voxel in the mask, nothing to fit.")
            _set_button(button, "lightgray")
            return None
        _set_button(button, "#ff8c00")
        t_start = time.time()
        maps, resid = fit_slice(data, te, z, voxels, sigma_noise)
        result = {"maps": maps, "resid": resid, "elapsed": time.time() - t_start}
        cache[key] = result

        s = slice_summary(maps)
        print(f"[Fit] Slice {z + 1}: {s['n']} voxel(s) fitted in {result['elapsed']:.1f} s | "
              f"mono kept in {s['n_mono']}, bi kept in {s['n_bi']} ({100 * s['share_bi']:.1f} %) | "
              f"median chi2_red = {s['chi2_median']:.2f}")
        if s["n_limit"]:
            print(f"        {s['n_limit']} voxel(s) have a T2 on a limit of the search range "
                  "(value not reliable; flagged in the export).")
        return result

    # ── Mono + bi fit of one curve or one zone, comparison and decay-curve plot ──
    # Used by both the single-voxel click ("Fit" mode) and the zone fit
    # ("Zone fit" selection action), so the two stay consistent.

    def _fit_and_plot(signal, title, header_lines=()):
        """Fit mono and bi on a curve (1-D) or on a zone of voxels (2-D, fitted
        jointly, see :mod:`functions.model`), print the comparison and plot the
        decay curves.

        Returns the :func:`functions.model.fit_both` result and the reduced
        chi-square of each model."""
        curves = np.atleast_2d(signal)
        fit = fit_both(te, signal, sigma_noise)
        if fit["mono"] is None and fit["bi"] is None:
            print("[Fit] No fit possible on this signal (empty or invalid).")
            return fit, {}
        chi2 = {m: fit_quality(fit[m], signal, sigma_noise) for m in MODELS if fit[m] is not None}
        best = fit["best"]

        sep = "─" * 70
        print(f"\n{sep}")
        for line in header_lines:
            print(f"  {line}")
        print(sep)
        print(f"  {'Model':<8} {'I0':>9} {'T2 (ms)':>14} {'f':>6} {'AICc':>9} {'chi2_red':>9}")
        print(sep)
        for name in MODELS:
            res = fit[name]
            if res is None:
                continue
            p = res["params"]
            t2_txt = f"{p['T2']:.2f}" if name == "mono" else f"{p['T2c']:.1f} / {p['T2l']:.1f}"
            f_txt = f"{p['f']:.2f}" if name == "bi" else "—"
            star = " ★" if name == best else ""
            warn = "   T2 on a search limit: not reliable" if res["at_bound"] else ""
            print(f"  {name:<8} {p['I0']:>9.1f} {t2_txt:>14} {f_txt:>6} "
                  f"{res['aicc']:>9.1f} {chi2[name]:>9.2f}{star}{warn}")
        print(sep)
        d_aicc = fit["delta_aicc"]
        if np.isfinite(d_aicc):
            print(f"  dAICc = AICc(mono) - AICc(bi) = {d_aicc:.1f}  (bi needs > {DELTA_AICC:g})"
                  f"  ->  {best} kept")
        print(sep)

        fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
        mean = curves.mean(axis=0)
        if curves.shape[0] > 1:
            ax2.errorbar(te, mean, yerr=curves.std(axis=0), fmt="o", color="black", markersize=3.5,
                         capsize=2, elinewidth=0.8, zorder=5,
                         label=f"data: mean ± sd of {curves.shape[0]} voxels")
        else:
            ax2.plot(te, mean, "o", color="black", label="data", zorder=5)
        floor = sigma_noise * np.sqrt(np.pi / 2)
        ax2.axhline(floor, color="darkorange", linestyle=":", linewidth=1.2,
                    label=f"noise floor ({floor:.0f})")
        for name in MODELS:
            res = fit[name]
            if res is None:
                continue
            is_best = name == best
            ax2.plot(te, res["expected"], color=_MODEL_COLORS[name],
                     linewidth=2.5 if is_best else 1.2, zorder=4 if is_best else 3,
                     label=f"{'★ ' if is_best else ''}{name}  (AICc={res['aicc']:.1f}, chi2={chi2[name]:.2f})")
            if is_best:
                ax2.plot(te, res["fitted"], "--", color=_MODEL_COLORS[name], linewidth=1.0,
                         label=f"{name} without noise (the decay that was fitted)")
        ax2.set_xlabel("Echo time (ms)")
        ax2.set_ylabel("Signal intensity")
        ax2.set_title(title)
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fig2.canvas.draw_idle()
        fig2.show()
        plt.pause(0.001)

        return fit, chi2

    # ── Raw signal of a zone (no fit) ────────────────────────────────────────

    def _show_signal(signals, scope):
        """Plot the raw decay curves of the voxels of a zone, with no fit.

        ``signals`` is (n_voxels, n_echoes). Left: linear scale. Right: same
        data with a logarithmic signal axis, where a single exponential decay
        is a straight line and a bi-exponential one is a curve that flattens.
        Thin grey lines = individual voxels (at most 200 drawn), black =
        mean of the zone, grey band = mean ± 1 standard deviation across
        voxels, dashed orange = expected noise floor sigma*sqrt(pi/2)."""
        n_vox = signals.shape[0]
        mean = signals.mean(axis=0)
        std = signals.std(axis=0)
        floor = sigma_noise * np.sqrt(np.pi / 2)

        fig3, (axl, axg) = plt.subplots(1, 2, figsize=(11, 4.2))
        shown = signals if n_vox <= 200 else signals[
            np.random.default_rng(0).choice(n_vox, 200, replace=False)
        ]
        for ax_i in (axl, axg):
            for curve in shown:
                ax_i.plot(te, curve, color="0.6", alpha=0.25, linewidth=0.7)
            ax_i.fill_between(te, np.maximum(mean - std, 1e-9), mean + std,
                              color="0.5", alpha=0.25, linewidth=0)
            ax_i.plot(te, mean, "o-", color="black", markersize=3.5, linewidth=1.5,
                      label=f"mean of {n_vox} voxel(s)", zorder=5)
            ax_i.axhline(floor, color="darkorange", linestyle="--", linewidth=1.2,
                         label=f"expected noise floor ({floor:.0f})")
            ax_i.set_xlabel("Echo time (ms)")
            ax_i.set_ylabel("Signal intensity")
        axl.set_title("Linear scale")
        axg.set_yscale("log")
        axg.set_ylim(bottom=floor * 0.5, top=float(np.max(mean + std)) * 1.2)
        axg.set_title("Log scale (straight line = single exponential)")
        axl.legend(fontsize=8)
        fig3.suptitle(f"Raw signal, no fit — {scope}", fontsize=11)
        fig3.tight_layout()
        fig3.canvas.draw_idle()
        fig3.show()
        plt.pause(0.001)

        print(f"[Signal] {n_vox} voxel(s), {scope}: mean at echo 1 = {mean[0]:.0f}, "
              f"at last echo = {mean[-1]:.0f} (noise floor {floor:.0f}).")

    def _zone_t2_mono(coords):
        """T2 mono of every voxel of a zone, as an array (NaN where a fit failed).

        ``coords`` is a list of ``(z, x, y)``. Same fit as the Mono Map: read from
        the cached fit of the slice when there is one, otherwise computed on the
        voxels of the zone only (and not stored, the cache holds whole slices).
        """
        t2 = []
        for z in sorted({c[0] for c in coords}):
            pts = [(x, y) for cz, x, y in coords if cz == z]
            if ("fit", z) in cache:
                maps = cache[("fit", z)]["maps"]
            else:
                maps, _ = fit_slice(data, te, z, pts, sigma_noise)
            t2.extend(maps["T2_mono"][x, y] for x, y in pts)
        return np.array(t2)

    def _apply_selection(inside):
        """Apply the current action (exclude/fit only/flag/zone fit) over `inside`
        (nx, ny bool mask, array convention), on the current slice or all
        slices."""
        action = radio_action.value_selected
        propagate = check_propagate.get_status()[0]
        z_list = range(mask.shape[2]) if propagate else [int(slice_slider.val)]

        if action in ("Zone fit", "Show signal"):
            voxel_signals = []
            roi_coords = []  # (slice, x, y) of every voxel entering the average
            for z in z_list:
                sel = inside & mask[:, :, z]
                xs, ys = np.nonzero(sel)
                for vx, vy in zip(xs, ys):
                    voxel_signals.append(data[vx, vy, z, :])
                    roi_coords.append((z, int(vx), int(vy)))

            # Select mode stays active (button stays green): draw the next ROI directly.
            fig.canvas.draw_idle()

            if not voxel_signals:
                print(f"[Select] {action}: no voxel of the mask falls inside "
                      "the drawn region — nothing to do.")
                return

            scope = "all slices" if propagate else f"slice {int(slice_slider.val) + 1}"
            if action == "Show signal":
                _show_signal(np.array(voxel_signals), scope)
                _print_map_stats([("T2 mono (ms)", _zone_t2_mono(roi_coords), 1)],
                                 f"zone of {len(voxel_signals)} voxel(s), {scope}, T2 of each voxel")
                return

            roi_id = len(roi_list) + 1
            n_vox = len(voxel_signals)
            fit, chi2 = _fit_and_plot(
                np.array(voxel_signals),
                title=f"ROI #{roi_id} — joint fit of {n_vox} voxel(s), {scope}",
                header_lines=[
                    f"ROI #{roi_id} | {n_vox} voxel(s) fitted jointly, curves not averaged "
                    f"before the fit | {scope}",
                ],
            )
            if not chi2:
                return
            if fit["best"] == "bi" and n_vox > 1:
                print("  Note: bi kept on a zone of several voxels can also come from a mixture of "
                      "different voxels (a zone that is not homogeneous), not only from two "
                      "components inside each voxel. 'Show signal' on smaller zones helps to tell.")
            _print_map_stats([("T2 mono (ms)", _zone_t2_mono(roi_coords), 1)],
                             f"ROI #{roi_id}, {n_vox} voxel(s), {scope}, T2 of each voxel")
            roi_list.append({
                "id": roi_id, "scope": scope, "coords": roi_coords,
                "fit": fit, "chi2": chi2,
            })
            arr = np.array(roi_coords)
            print(f"  ROI #{roi_id} kept in memory: x {arr[:, 1].min()}–{arr[:, 1].max()}, "
                  f"y {arr[:, 2].min()}–{arr[:, 2].max()} "
                  f"({len(roi_list)} ROI(s) so far, written to exports/ by the Export button)")
            return

        n_total = 0
        for z in z_list:
            if action == "Exclude":
                n_total += int(np.sum(inside & mask[:, :, z]))
                mask[:, :, z] &= ~inside
                mask_base[:, :, z] &= ~inside
                # Stale cache for this slice would silently mix old/new masks.
                for key in [k for k in cache if k[1] == z]:
                    del cache[key]
            elif action == "Fit only":
                # Inverse of Exclude: keep ONLY the voxels inside the drawn
                # region, drop every other tissue voxel from the mask. Maps,
                # fits and the CSV export then cover this zone only.
                # Restarts from mask_base, so a new "Fit only" replaces the
                # previous zone instead of intersecting with it.
                n_total += int(np.sum(mask_base[:, :, z] & ~inside))  # voxels dropped
                mask[:, :, z] = mask_base[:, :, z] & inside
                for key in [k for k in cache if k[1] == z]:
                    del cache[key]
            else:  # "Flag capillary"
                n_total += int(np.sum(inside & ~capillary_mask[:, :, z]))
                capillary_mask[:, :, z] |= inside

        scope = "all slices" if propagate else f"slice {int(slice_slider.val) + 1}"
        if action == "Fit only":
            kept = sum(int(mask[:, :, z].sum()) for z in z_list)
            print(f"[Select] Fit only: {kept} voxel(s) kept, {n_total} other tissue voxel(s) "
                  f"removed from the mask ({scope}).")
        else:
            verb = "excluded from the mask" if action == "Exclude" else "flagged as capillary"
            print(f"[Select] {n_total} voxel(s) {verb} ({scope}).")
        if action in ("Exclude", "Fit only"):
            print("         Cached results cleared where affected — recompute maps to see the effect.")

        # Select mode stays active here too; click "Select" again (or Reset) to leave it.
        _draw_mask_overlay()
        fig.canvas.draw_idle()

    def _on_lasso_select(verts):
        points, (nx, ny) = _grid_points(mask.shape[0], mask.shape[1])
        inside = MplPath(verts).contains_points(points).reshape(nx, ny)
        _apply_selection(inside)

    def _on_box_select(eclick, erelease):
        nx, ny = mask.shape[0], mask.shape[1]
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        rows, cols = np.mgrid[0:nx, 0:ny]
        inside = (cols >= x0) & (cols <= x1) & (rows >= y0) & (rows <= y1)
        _apply_selection(inside)

    def _on_circle_select(eclick, erelease):
        nx, ny = mask.shape[0], mask.shape[1]
        # EllipseSelector reports eclick/erelease as the two opposite
        # corners of the bounding box (same convention as
        # RectangleSelector) — NOT a center + drag-radius. Using eclick
        # directly as the center produced a circle offset from what was
        # actually drawn, which is why voxels near the true edge were
        # missed.
        cx = (eclick.xdata + erelease.xdata) / 2
        cy = (eclick.ydata + erelease.ydata) / 2
        rx = abs(erelease.xdata - eclick.xdata) / 2
        ry = abs(erelease.ydata - eclick.ydata) / 2
        rows, cols = np.mgrid[0:nx, 0:ny]
        inside = ((cols - cx) / max(rx, 1e-9)) ** 2 + ((rows - cy) / max(ry, 1e-9)) ** 2 <= 1
        _apply_selection(inside)

    def _activate_selector():
        """(Re)create the matplotlib selector matching the chosen shape."""
        if _selector["obj"] is not None:
            _selector["obj"].disconnect_events()
            _selector["obj"] = None
        shape = radio_shape.value_selected
        if shape == "Lasso":
            _selector["obj"] = LassoSelector(ax, onselect=_on_lasso_select, useblit=True)
        elif shape == "Box":
            _selector["obj"] = RectangleSelector(
                ax, onselect=_on_box_select, useblit=True, button=[1], interactive=False,
            )
        else:  # "Circle"
            _selector["obj"] = EllipseSelector(
                ax, onselect=_on_circle_select, useblit=True, button=[1], interactive=False,
            )

    def _on_shape_change(label):
        if mode["value"] == "select":
            _activate_selector()

    radio_shape.on_clicked(_on_shape_change)

    def _close_select():
        """Leave Select mode: drop the selector, hide and disable the panel."""
        if mode["value"] == "select":
            mode["value"] = None
        if _selector["obj"] is not None:
            _selector["obj"].disconnect_events()
            _selector["obj"] = None
        for a in (ax_shape, ax_action, ax_propagate):
            a.set_visible(False)
        radio_shape.active = False
        radio_action.active = False
        check_propagate.active = False

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(event):
        slice_slider.reset()
        echo_slider.reset()
        mode["value"] = None
        if _selector["obj"] is not None:
            _selector["obj"].disconnect_events()
            _selector["obj"] = None
        for a in (ax_shape, ax_action, ax_propagate):
            a.set_visible(False)
        radio_shape.active = False
        radio_action.active = False
        check_propagate.active = False
        _reset_buttons()
        fig.canvas.draw_idle()

    button_reset.on_clicked(reset)

    # ── Restore mask ──────────────────────────────────────────────────────────

    def restore_mask(event):
        """Back to the automatic mask: undo every Exclude / Fit only, clear the
        capillary flags and all cached maps (they were computed on the old mask)."""
        if mask is None:
            return
        n_back = int(np.sum(mask_original & ~mask))
        n_cap = int(capillary_mask.sum())
        mask[...] = mask_original
        mask_base[...] = mask_original
        capillary_mask[...] = False
        cache.clear()
        print(f"[Restore] Automatic mask restored on all slices: {n_back} voxel(s) back in the mask, "
              f"{n_cap} capillary flag(s) cleared, cached maps cleared.")
        _draw_mask_overlay()
        fig.canvas.draw_idle()

    button_restore.on_clicked(restore_mask)

    # ── Button callbacks ──────────────────────────────────────────────────────

    def make_callback(button):
        def toggle_mode(event):
            # Any other button ends Select mode (it would otherwise stay armed
            # now that a selection no longer closes it).
            if mode["value"] == "select" and button is not button_select:
                _close_select()
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

            # ── Select (toggle lasso/box/circle exclude or flag) ────────────────
            if button is button_select:
                if mode["value"] == "select":
                    mode["value"] = None
                    if _selector["obj"] is not None:
                        _selector["obj"].disconnect_events()
                        _selector["obj"] = None
                    for a in (ax_shape, ax_action, ax_propagate):
                        a.set_visible(False)
                    radio_shape.active = False
                    radio_action.active = False
                    check_propagate.active = False
                else:
                    mode["value"] = "select"
                    _set_button(button_select, "#7bff23")
                    for a in (ax_shape, ax_action, ax_propagate):
                        a.set_visible(True)
                    radio_shape.active = True
                    radio_action.active = True
                    check_propagate.active = True
                    _activate_selector()
                    print("[Select] Draw on the image with the chosen shape. "
                          "Pick action/shape/propagation on the left before drawing.")
                fig.canvas.draw_idle()
                return

            # ── Mono Map ──────────────────────────────────────────────────────
            if button is button_mono_mapping:
                mode["value"] = None
                z = int(slice_slider.val)
                res = _get_fit(z, button)
                if res is None:
                    return
                _set_button(button, "#7bff23")

                arr = res["maps"]["T2_mono"]
                fig_m, (ax_map, ax_dist) = plt.subplots(1, 2, figsize=(11, 4.8))
                fig_m.canvas.manager.set_window_title("Mono Mapping")
                fig_m.suptitle(f"Mono mapping — slice {z + 1}", fontsize=12)
                _draw_map(fig_m, ax_map, arr, "T2 mono (ms)", "hot", *_percentile_range(arr))
                _draw_distribution(ax_dist, arr, "T2 mono (ms)", "T2 (ms)")
                _print_map_stats([("T2 mono (ms)", arr, 1)], f"slice {z + 1}")
                plt.tight_layout()
                plt.show()

            # ── Bi Map ────────────────────────────────────────────────────────
            elif button is button_bi_mapping:
                mode["value"] = None
                z = int(slice_slider.val)
                res = _get_fit(z, button)
                if res is None:
                    return
                _set_button(button, "#7bff23")

                maps = res["maps"]
                s = slice_summary(maps)
                fig_b, axes = plt.subplots(2, 2, figsize=(10, 8.5))
                fig_b.canvas.manager.set_window_title("Bi Mapping")
                fig_b.suptitle(
                    f"Bi mapping — slice {z + 1} | two components kept in {s['n_bi']} of {s['n']} voxels "
                    f"({100 * s['share_bi']:.1f} %), shown in colour; light grey = mono kept",
                    fontsize=11,
                )
                tissue = np.isfinite(maps["best"])
                stat_entries = []
                for ax_b, key, title, cmap_b, fixed_range in [
                    (axes[0, 0], "T2c_bi",   "T2c bi — short component (ms)", "hot", None),
                    (axes[0, 1], "T2l_bi",   "T2l bi — long component (ms)",  "hot", None),
                    (axes[1, 0], "f_bi",     "Water fraction f (bi)\nrestricted vs free", "RdBu_r", (0, 1)),
                    (axes[1, 1], "T2eff_bi", "T2 effective bi: f·T2c + (1-f)·T2l (ms)", "hot", None),
                ]:
                    arr = maps[key]
                    vmin_b, vmax_b = fixed_range if fixed_range else _percentile_range(arr)
                    _draw_map(fig_b, ax_b, arr, title, cmap_b, vmin_b, vmax_b, tissue=tissue)
                    stat_entries.append((title.split("\n")[0], arr, 2 if key == "f_bi" else 1))
                _print_map_stats(stat_entries, f"slice {z + 1}")
                plt.tight_layout()
                plt.show()

            # ── Utils ─────────────────────────────────────────────────────────
            elif button is button_utils:
                mode["value"] = None
                z = int(slice_slider.val)
                res = _get_fit(z, button)
                if res is None:
                    return
                _set_button(button, "#7bff23")

                maps = res["maps"]
                s = slice_summary(maps)
                global_best = "bi" if s["n_bi"] > s["n_mono"] else "mono"
                tissue = np.isfinite(maps["best"])

                fig_u, axes = plt.subplots(3, 2, figsize=(10, 12))
                fig_u.suptitle(
                    f"Utils — slice {z + 1} | {global_best} for most voxels | {res['elapsed']:.1f} s",
                    fontsize=13,
                )
                fig_u.canvas.manager.set_window_title("Utils")

                # Model kept in each voxel: 0 = mono, 1 = bi
                axes[0, 0].imshow(
                    maps["best"], cmap=ListedColormap([_MODEL_COLORS[m] for m in MODELS]),
                    vmin=-0.5, vmax=1.5, origin="upper", interpolation="nearest",
                )
                axes[0, 0].set_title("Model kept (AICc)")
                axes[0, 0].axis("off")
                _use_array_coords(axes[0, 0], maps["best"])
                patches = [
                    mpatches.Patch(color=_MODEL_COLORS[m], label=f"{m}: {s['n_' + m]} voxels")
                    for m in MODELS
                ]
                axes[0, 0].legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                                  ncol=2, fontsize=7, frameon=False)

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
                _draw_map(fig_u, axes[0, 1], vol_smooth, "Smoothed image (σ=1.5, echo 1)", "viridis")

                _draw_map(fig_u, axes[1, 0], maps["f_bi"],
                          "Water fraction f (bi, where kept)\nrestricted vs free",
                          "RdBu_r", 0, 1, tissue=tissue)
                # Evidence for bi: white at the decision threshold, red = bi justified
                _draw_map(fig_u, axes[1, 1], np.clip(maps["delta_aicc"], -10, 20),
                          f"Evidence for bi: dAICc = AICc(mono) - AICc(bi)\n(red above {DELTA_AICC:g}: bi kept)",
                          "RdBu_r", norm=TwoSlopeNorm(vmin=-10, vcenter=DELTA_AICC, vmax=20))

                _draw_map(fig_u, axes[2, 0], maps["I0_best"], "I0 voxel-wise (kept model)",
                          "viridis", *_percentile_range(maps["I0_best"], 1, 99))
                _draw_map(fig_u, axes[2, 1], maps[f"I0_{global_best}"], f"I0 global ({global_best})",
                          "viridis", *_percentile_range(maps[f"I0_{global_best}"], 1, 99))

                plt.tight_layout()
                plt.show()

            # ── Noise ─────────────────────────────────────────────────────────
            elif button is button_noise:
                mode["value"] = None
                z = int(slice_slider.val)
                res = _get_fit(z, button)
                if res is None:
                    return
                _set_button(button, "#7bff23")

                fig_n, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig_n.canvas.manager.set_window_title("Noise")

                t_cur = int(echo_slider.val)
                if "bg3d" not in cache_bg:
                    # Background voxels kept by the sigma-clipping (same population
                    # as the auto noise estimate): drops the bright halo around the
                    # tissue, which would otherwise stretch the single-echo histogram.
                    vol_max = np.max(data, axis=-1)
                    mu_bg, sd_bg, bg_pre, _ = estimate_noise_auto(vol_max, verbose=False)
                    cache_bg["bg3d"] = bg_pre & (vol_max > 0) & (np.abs(vol_max - mu_bg) < 3.0 * sd_bg)
                plot_histogram(
                    data, ax=axes[0, 0], z=z, t=t_cur,
                    bg_mask=cache_bg["bg3d"], te_ms=float(te[t_cur]), sigma=sigma_noise,
                )

                if mask is not None:
                    axes[0, 1].imshow(
                        mask[:, :, z], cmap="gray",
                        origin="upper", interpolation="nearest",
                    )
                    _use_array_coords(axes[0, 1], mask[:, :, z])
                axes[0, 1].set_title(f"Tissue mask — slice {z + 1}")
                axes[0, 1].axis("off")

                # Noise level measured on each echo: should be the same everywhere
                axes[1, 0].plot(te, sigma_echo, "o-", color="steelblue", markersize=3.5)
                axes[1, 0].axhline(sigma_noise, color="darkorange", linestyle="--",
                                   label=f"sigma used = {sigma_noise:.0f} (median)")
                axes[1, 0].set_title("Noise level sigma measured in the background, echo by echo")
                axes[1, 0].set_xlabel("Echo time (ms)")
                axes[1, 0].set_ylabel("sigma")
                axes[1, 0].legend(fontsize=8)

                # Mean residual of each echo over the tissue voxels of the slice: a model
                # that explains the data leaves 0 at every echo.
                resid = res["resid"]
                n_ok = np.sum(np.isfinite(resid), axis=0)
                mean_r = np.nanmean(resid, axis=0)
                sem_r = np.nanstd(resid, axis=0) / np.sqrt(np.maximum(n_ok, 1))
                axes[1, 1].axhline(0, color="black", linewidth=0.8)
                axes[1, 1].errorbar(te, mean_r, yerr=2 * sem_r, fmt="o-", color="firebrick",
                                    markersize=3.5, capsize=2, elinewidth=0.8)
                axes[1, 1].set_title("Mean residual of each echo (kept model, tissue voxels)\n"
                                     "0 = the model explains this echo; bars = ±2 standard errors")
                axes[1, 1].set_xlabel("Echo time (ms)")
                axes[1, 1].set_ylabel("measured - expected, in noise units")
                # First three echoes: residual in noise units, and as a share of the signal
                # (residual x sigma / tissue signal), the scale used by the validation report.
                n_first = min(3, len(te))
                first = mean_r[:n_first]
                tissue_z = mask[:, :, z] if mask is not None else np.ones(data.shape[:2], dtype=bool)
                sig_first = np.array([np.median(data[:, :, z, i][tissue_z]) for i in range(n_first)])
                shift_pc = 100 * first * sigma_noise / sig_first
                rest = mean_r[n_first:]
                print("[Noise] Mean residual of the first echoes: "
                      + ", ".join(f"{v:+.2f}" for v in first) + " noise units, about "
                      + ", ".join(f"{v:+.1f} %" for v in shift_pc) + " of the signal (0 = model right)."
                      + (f" Other echoes: from {np.nanmin(rest):+.2f} to {np.nanmax(rest):+.2f}." if rest.size else ""))
                if np.any(np.abs(shift_pc) >= 5):
                    n_off = max(i + 1 for i, v in enumerate(shift_pc) if abs(v) >= 3)
                    print("        A first-echo shift of 5 % or more can bias T2 a little (more on a large zone): see SKIP_FIRST_ECHOES. "
                          f"If you use it, skip every shifted echo (here at least {n_off}), never fewer: skipping only one echo when the "
                          "next one is also shifted is worse than skipping none (section 7 of the validation report).")

                plt.tight_layout()
                plt.show()

            # ── Error ─────────────────────────────────────────────────────────
            elif button is button_error:
                mode["value"] = None
                z = int(slice_slider.val)
                res = _get_fit(z, button)
                if res is None:
                    return
                _set_button(button, "#7bff23")

                maps = res["maps"]
                s = slice_summary(maps)
                global_best = "bi" if s["n_bi"] > s["n_mono"] else "mono"

                fig_e, axes = plt.subplots(3, 3, figsize=(14, 11))
                fig_e.suptitle(
                    f"Error — slice {z + 1} | global={global_best} | {res['elapsed']:.1f} s", fontsize=13
                )
                fig_e.canvas.manager.set_window_title("Error")

                # chi2_red is about 1 where the model explains the data down to the noise
                for row, (key, label, cmap_e, fixed_range, fmt) in enumerate([
                    ("chi2", "chi2 reduced", "RdYlGn_r", (0.5, 2.0), ".2f"),
                    ("rmse", "RMSE",         "hot_r",    None,       ".1f"),
                    ("r2",   "R²",           "RdYlGn",   (0, 1),     ".3f"),
                ]):
                    arr_vw = maps[f"{key}_best"]
                    arr_glob = maps[f"{key}_{global_best}"]
                    for col, (arr, title) in enumerate([
                        (arr_vw,   f"{label} voxel-wise (kept model)"),
                        (arr_glob, f"{label} global ({global_best})"),
                    ]):
                        vm_min, vm_max = fixed_range if fixed_range else _percentile_range(arr)
                        _draw_map(fig_e, axes[row, col], arr, title, cmap_e, vm_min, vm_max)

                    ax_dist = axes[row, 2]
                    both = np.concatenate([arr_vw[np.isfinite(arr_vw)], arr_glob[np.isfinite(arr_glob)]])
                    hist_range = tuple(np.percentile(both, [0.5, 99.5])) if both.size else None
                    for arr, label_d, color in [
                        (arr_vw,   "voxel-wise", "steelblue"),
                        (arr_glob, "global",     "tomato"),
                    ]:
                        finite_v = arr[np.isfinite(arr)]
                        if len(finite_v) > 0:
                            mu = np.mean(finite_v)
                            ax_dist.hist(finite_v, bins=60, range=hist_range, density=True,
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

                if voxel_dims is not None:
                    dx, dy, dz = voxel_dims
                    x_coords, y_coords, z_coords = x_idx * dx, y_idx * dy, z_idx * dz
                    axis_titles = {
                        "xaxis_title": "X (mm)", "yaxis_title": "Y (mm)",
                        "zaxis_title": "Z (mm)",
                    }
                else:
                    x_coords, y_coords, z_coords = x_idx, y_idx, z_idx + 1   # slices counted from 1
                    axis_titles = {
                        "xaxis_title": "X (voxels)", "yaxis_title": "Y (voxels)",
                        "zaxis_title": "Z (slice index)",
                    }

                fig_3d = go.Figure(data=go.Volume(
                    x=x_coords.flatten(),
                    y=y_coords.flatten(),
                    z=z_coords.flatten(),
                    value=vol_plot.flatten(),
                    isomin=iso_min,
                    isomax=iso_max,
                    opacity=0.15,
                    surface_count=20,
                    colorscale="Viridis",
                    caps={"x_show": False, "y_show": False, "z_show": False},
                ))
                fig_3d.update_layout(
                    title=f"3D view — echo 1 | threshold {iso_min:.0f}–{iso_max:.0f}",
                    scene={**axis_titles, "aspectmode": "data"},
                    margin={"l": 0, "r": 0, "t": 40, "b": 0},
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

                out_path = Path("exports") / f"T2_slice_{z + 1}"

                # Toutes les cartes de la coupe viennent du même calcul (cache ("fit", z)).
                # Les colonnes du bi sont vides là où le mono suffit, comme sur la Bi Map.
                maps_dict = {}
                if ("fit", z) in cache:
                    fit_maps = cache[("fit", z)]["maps"]
                    maps_dict = {name: fit_maps[key] for key, name in _EXPORT_COLUMNS.items()}

                if maps_dict and capillary_mask is not None and capillary_mask[:, :, z].any():
                    # 1 = voxel flagged as capillary via the Select tool,
                    # 0 = sample tissue. Only added when at least one voxel
                    # was actually flagged on this slice — an always-0
                    # column would look like "no capillary found" (a
                    # computed result) rather than "never asked" (the
                    # actual reason), which is misleading.
                    maps_dict["is_capillary"] = capillary_mask[:, :, z].astype(float)

                if maps_dict and voxel_dims is not None:
                    # Real-world position in mm, same array convention as
                    # everywhere else (x=row, y=col). Without this, every
                    # exported row only had raw voxel indices — no way to
                    # know the physical size of anything (a capillary, the
                    # sillon...) without also knowing the acquisition's
                    # voxel size, which lived nowhere in the pipeline.
                    dx, dy, dz = voxel_dims
                    nx_e, ny_e = mask.shape[0], mask.shape[1]
                    rows, cols = np.mgrid[0:nx_e, 0:ny_e]
                    maps_dict["x_mm"] = rows * dx
                    maps_dict["y_mm"] = cols * dy
                    maps_dict["z_mm"] = np.full((nx_e, ny_e), z * dz)

                if roi_list:
                    # The ROIs do not depend on any map: written whatever the slice.
                    voxels_path, fits_path = export_rois(roi_list, Path("exports") / "T2_ROIs")
                    print(f"[Export] {len(roi_list)} ROI exportée(s) : {voxels_path} et {fits_path}")

                if not maps_dict:
                    if roi_list:
                        print(
                            "[Export] Aucune carte calculée pour cette coupe : seules les ROI "
                            "sont exportées (Mono Map, Bi Map, Utils, Error ou Noise pour la carte)."
                        )
                        _set_button(button, "#7bff23")
                    else:
                        print(
                            "[Export] Rien à exporter — calcule d'abord une carte "
                            "(Mono Map, Bi Map, Utils, Error ou Noise) pour cette coupe."
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
                                got = float(row[col_idx[name]])
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
    button_select.on_clicked(make_callback(button_select))

    # ── Voxel click — per-voxel fit plot ─────────────────────────────────────

    def onclick(event):
        toolbar = event.canvas.toolbar
        ui_axes = [
            button_fit.ax, button_mono_mapping.ax, button_bi_mapping.ax,
            button_utils.ax, button_noise.ax, button_reset.ax,
            button_select.ax, button_restore.ax, ax_shape, ax_action, ax_propagate,
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

        # A voxel is centred on integer plot coordinates and spans +-0.5 around
        # them (same rule as the hover readout), so round, do not truncate.
        col = int(round(event.xdata))   # plot x = array column
        row = int(round(event.ydata))   # plot y = array row
        z = int(slice_slider.val)
        if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
            return
        signal = data[row, col, z, :]

        # Labelled (x, y) here in the ARRAY convention used everywhere else
        # (mask[x, y, z], the hover readout, and the CSV export: x=row,
        # y=column) — not matplotlib's plot-coordinate convention. Same
        # voxel as before, just a label that now matches the rest of the UI.
        _fit_and_plot(
            signal,
            title=f"Voxel (x={row}, y={col}) — slice {z + 1}",
            header_lines=[f"Voxel (x={row}, y={col}) | slice {z + 1}"],
        )

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()