"""
Quick T2 statistics from the console: no window, no plot.

For each NIfTI file, this script fits the mono-exponential model on every tissue
voxel of a slice (the middle one, like the viewer) and prints the minimum, median,
mean, maximum and standard deviation of the resulting T2 map. It uses exactly the
same steps as the viewer (tissue mask, noise level, maximum-likelihood fit), so the
numbers are the ones its "Mono Map" button shows.

Usage
-----
Run from the project root::

    pixi run stats scan.nii                      # one image, middle slice
    pixi run stats scan.nii --acqp scan.acqp     # with its ACQP file
    pixi run stats data/                         # every .nii of a folder, one by one
    pixi run stats data/ --acqp acqp_folder/     # ACQP files looked up by name
    pixi run stats scan.nii --all-slices         # every slice, plus all slices pooled

Echo times
----------
For each image, the first source that works is used:

1. the ACQP file given with ``--acqp``;
2. in the ACQP folder given with ``--acqp``, the file named like the image
   (``scan.nii`` -> ``scan.acqp``);
3. next to the image, the file named like it (``scan.nii`` -> ``scan.acqp``);
4. synthetic echo times ``1, 2, 3, ... x --te-step`` ms (3 ms by default, like the
   viewer). The T2 values then carry that scale: comparisons between voxels or
   zones stay exact, but the values in ms are only approximate.

The echo times of an ACQP file must be as many as the echoes of the image.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time
import warnings
from pathlib import Path

# Progress bars off unless --verbose. Must be set before tqdm is imported.
if "--verbose" not in sys.argv and "-v" not in sys.argv:
    os.environ.setdefault("TQDM_DISABLE", "1")

import nibabel as nib
import numpy as np

from functions.io import load_acqp
from functions.mapping import STAT_KEYS, fit_slice, map_stats, stats_header, stats_line
from functions.utils import compute_mask, estimate_sigma

DEFAULT_TE_STEP = 3.0   # same synthetic echo spacing as the viewer (ms)
MIN_ECHOES = 3          # below that, a decay cannot be fitted


class SkipFile(Exception):
    """A file that cannot be analysed; the message says why."""


# ── Finding the files ─────────────────────────────────────────────────────────

def nifti_stem(path):
    """File name without ``.nii`` or ``.nii.gz``."""
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(path).stem


def is_nifti(path):
    """True for an existing ``.nii`` or ``.nii.gz`` file."""
    return path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))


def list_niftis(folder, recursive=False):
    """NIfTI files of a folder, sorted by name (sub-folders too if ``recursive``)."""
    items = folder.rglob("*") if recursive else folder.iterdir()
    return sorted((p for p in items if is_nifti(p)), key=lambda p: str(p).lower())


def find_acqp_in(folder, stem):
    """The ``<stem>.acqp`` file of a folder (case-insensitive), or ``None``."""
    wanted = stem.lower()
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() == ".acqp" and p.stem.lower() == wanted:
            return p
    return None


# ── Echo times ────────────────────────────────────────────────────────────────

def read_acqp(path, n_te):
    """Echo times (ms) of an ACQP file, checked against the number of echoes.

    Raises
    ------
    ValueError
        If the file has no usable echo times, not as many as ``n_te``, or if they
        do not increase.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            te = load_acqp(path)
        except SystemExit:
            raise ValueError(f"{path.name}: no echo times (##$ACQ_echo_time) found") from None
    if len(te) != n_te:
        raise ValueError(f"{path.name} has {len(te)} echo times but the image has {n_te} echoes")
    if te[0] <= 0 or np.any(np.diff(te) <= 0):
        raise ValueError(f"{path.name}: the echo times are not increasing")
    return te


def resolve_te(nii, n_te, acqp, te_step):
    """Echo times of an image and where they come from.

    Parameters
    ----------
    nii : pathlib.Path
        The image.
    n_te : int
        Number of echoes of the image.
    acqp : pathlib.Path or None
        The ``--acqp`` argument: an ACQP file, an ACQP folder, or ``None``.
    te_step : float
        Spacing (ms) of the synthetic echo times, used when no ACQP is found.

    Returns
    -------
    te : np.ndarray of shape (n_te,)
    source : str
        Short description of where ``te`` comes from.
    synthetic : bool
        True if the echo times are synthetic.
    notes : list of str
        Things worth telling the user (an ACQP looked for but not found...).

    Raises
    ------
    ValueError
        If an ACQP that the user asked for with ``--acqp`` does not fit the image.
        A file found by name next to the image only triggers a note.
    """
    notes = []
    stem = nifti_stem(nii)
    if acqp is not None and acqp.is_file():
        return read_acqp(acqp, n_te), f"{acqp.name} (given with --acqp)", False, notes
    if acqp is not None:                                   # an ACQP folder
        found = find_acqp_in(acqp, stem)
        if found is not None:
            return read_acqp(found, n_te), f"{found.name} (same name, in the ACQP folder)", False, notes
        notes.append(f"no '{stem}.acqp' in the folder given with --acqp")
    found = find_acqp_in(nii.parent, stem)
    if found is not None:
        try:
            return read_acqp(found, n_te), f"{found.name} (same name, next to the image)", False, notes
        except ValueError as exc:
            notes.append(f"{exc}: ignored")
    te = np.arange(1, n_te + 1) * te_step
    return te, f"synthetic, {te_step:g} ms steps", True, notes


# ── Statistics ────────────────────────────────────────────────────────────────

def progress(text):
    """Show ``text`` on the current line of the terminal (``None`` clears it).

    Written to stderr and only in an interactive terminal, so redirected output
    stays clean.
    """
    if not sys.stderr.isatty():
        return
    sys.stderr.write("\r" + (text or "").ljust(78) + ("" if text else "\r"))
    sys.stderr.flush()


@contextlib.contextmanager
def quiet(verbose):
    """Hide what the library functions print, unless ``verbose``."""
    if verbose:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)    # e.g. Otsu on volumes of 3-4 slices
            yield


# ── Analysis of one image ─────────────────────────────────────────────────────

def analyse(path, args, acqp):
    """Fit the mono model on the chosen slice(s) of one image and describe the T2 map.

    Returns
    -------
    result : dict
        ``name``, ``shape``, ``te``, ``source``, ``synthetic``, ``notes``, ``sigma``,
        ``mask_voxels``, ``mask_share``, ``rows`` (one dict per slice, see below),
        ``pooled`` (statistics of all the slices together, ``None`` for a single
        slice), ``quality`` (counts over the analysed voxels) and ``seconds``.
        Each row has ``z`` (slice index counted from 0), ``label``, ``snr``, ``seconds``, ``n_mask`` (voxels of
        the mask) and the keys of :func:`functions.mapping.map_stats`.

    Raises
    ------
    SkipFile
        If the image cannot be analysed.
    """
    t_start = time.time()
    try:
        data = nib.load(str(path)).get_fdata()
    except Exception as exc:  # noqa: BLE001  (broad on purpose: any unreadable file)
        raise SkipFile(f"cannot read the file ({exc})") from None
    if data.ndim != 4:
        raise SkipFile(f"{data.ndim}-D image: a 4-D image (x, y, slice, echoes) is needed")
    nx, ny, nz, n_te = data.shape
    if n_te < MIN_ECHOES:
        raise SkipFile(f"{n_te} echo(es): at least {MIN_ECHOES} are needed to fit a decay")

    try:
        te, source, synthetic, notes = resolve_te(path, n_te, acqp, args.te_step)
    except ValueError as exc:
        raise SkipFile(str(exc)) from None

    if args.all_slices:
        zs = list(range(nz))
    elif args.slice is not None:
        if not 1 <= args.slice <= nz:
            raise SkipFile(f"slice {args.slice} does not exist (this image has slices 1 to {nz})")
        zs = [args.slice - 1]      # the user counts from 1, the code from 0
    else:
        zs = [nz // 2]

    try:
        with quiet(args.verbose):
            mask = compute_mask(data, method="rician")
            sigma = estimate_sigma(data)
    except Exception as exc:  # noqa: BLE001  (broad on purpose: report, do not stop a folder)
        raise SkipFile(f"tissue mask / noise level failed ({exc})") from None

    rows, kept_all, best_all, n_limit, n_failed = [], [], [], 0, 0
    for k, z in enumerate(zs, 1):
        in_mask = mask[:, :, z]
        n_mask = int(in_mask.sum())
        label = f"slice {z + 1}" + (" (middle)" if not args.all_slices and args.slice is None else "")
        row = {"z": z, "label": label, "n_mask": n_mask, "snr": np.nan, "seconds": 0.0,
               "share_bi": np.nan, "n_limit": 0}
        if n_mask == 0:
            row.update(map_stats([]))
            rows.append(row)
            continue
        voxels = [(int(x), int(y)) for x, y in np.argwhere(in_mask)]
        if not args.verbose:
            progress(f"  fitting slice {z + 1}: {len(voxels)} voxels ({k} of {len(zs)}) ...")
        t0 = time.time()
        with quiet(args.verbose):
            maps, _ = fit_slice(data, te, z, voxels, sigma, n_jobs=args.jobs)
        row["seconds"] = time.time() - t0
        row["snr"] = float(np.median(data[:, :, z, 0][in_mask]) / sigma)

        t2 = maps["T2_mono"][in_mask]
        best = maps["best"][in_mask]
        limit = maps["limit_mono"][in_mask] == 1
        finite = np.isfinite(t2)
        n_failed += int((~finite).sum())
        row["n_limit"] = int((finite & limit).sum())
        n_limit += row["n_limit"]
        used = finite & ~limit if args.exclude_unreliable else finite
        row.update(map_stats(t2[used]))
        row["share_bi"] = float(np.mean(best[used] == 1)) if used.any() else np.nan
        rows.append(row)
        kept_all.append(t2[used])
        best_all.append(best[used])
    progress(None)

    pooled = None
    values = np.concatenate(kept_all) if kept_all else np.array([])
    best = np.concatenate(best_all) if best_all else np.array([])
    n_used = int(values.size)
    if len(zs) > 1:
        pooled = map_stats(values)
        pooled["share_bi"] = float(np.mean(best == 1)) if n_used else np.nan
        pooled["n_limit"] = n_limit
    return {
        "name": path.name, "shape": (nx, ny, nz, n_te), "te": te, "source": source,
        "synthetic": synthetic, "notes": notes, "sigma": float(sigma),
        "mask_voxels": int(mask.sum()), "mask_share": float(mask.mean()),
        "rows": rows, "pooled": pooled,
        "quality": {"n_used": n_used, "share_bi": float(np.mean(best == 1)) if n_used else np.nan,
                    "n_limit": n_limit, "n_failed": n_failed,
                    "excluded": bool(args.exclude_unreliable)},
        "seconds": time.time() - t_start,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_result(r, index=None, total=None):
    """Print the report of one image."""
    nx, ny, nz, n_te = r["shape"]
    te = r["te"]
    tag = f"[{index}/{total}] " if total and total > 1 else ""
    title = f"{tag}{r['name']}"
    print()
    print(title)
    print("=" * max(len(title), 60))
    print(f"  image    {nx} x {ny} x {nz} slices x {n_te} echoes")
    print(f"  TE       {te[0]:.1f} - {te[-1]:.1f} ms  |  {r['source']}")
    print(f"  noise    sigma = {r['sigma']:.0f}  |  tissue mask: {r['mask_voxels']} voxels "
          f"({100 * r['mask_share']:.1f} % of the volume)")
    for note in r["notes"]:
        print(f"  note     {note}")
    if r["synthetic"]:
        print("  WARNING  synthetic echo times: T2 values are on that scale. Comparisons between "
              "voxels or zones")
        print("           stay exact, but the values in ms are only approximate. "
              "Give the ACQP file for real ones.")
    print()
    print(stats_header("T2 mono (ms)") + (f"{'SNR':>8}" if len(r["rows"]) > 1 else ""))
    for row in r["rows"]:
        snr = (f"{row['snr']:>8.1f}" if np.isfinite(row["snr"]) else f"{'-':>8}") if len(r["rows"]) > 1 else ""
        print(stats_line(row["label"], row) + snr)
    if r["pooled"] is not None:
        print(f"  {'-' * (14 + 7 + 9 * len(STAT_KEYS))}")
        print(stats_line("all slices", r["pooled"]))
    q = r["quality"]
    print()
    bits = []
    if q["n_used"]:
        bits.append(f"bi model preferred in {100 * q['share_bi']:.1f} % of the voxels")
    if q["n_limit"]:
        bits.append(f"T2 on a search limit (not reliable): {q['n_limit']} voxel(s), "
                    + ("left out" if q["excluded"] else "kept in the statistics"))
    if q["n_failed"]:
        bits.append(f"fit failed: {q['n_failed']} voxel(s)")
    if bits:
        print("  " + "  |  ".join(bits))
    if r["pooled"] is not None:
        print("  'all slices' = the statistics of all the voxels of all the slices together "
              "(not an average of the rows).")
    print(f"  done in {r['seconds']:.1f} s")


def print_summary(results, skipped):
    """One line per image, after a folder run."""
    print()
    print("Summary")
    print("=" * 60)
    print(f"  {'image':<38}{'TE':<7}{'slice(s)':<11}" + stats_header("", 0)[2:])
    for r in results:
        s = r["pooled"] if r["pooled"] is not None else r["rows"][0]
        where = "all" if r["pooled"] is not None else r["rows"][0]["label"].replace(" (middle)", "")
        name = r["name"] if len(r["name"]) <= 37 else r["name"][:34] + "..."
        te_kind = "synth." if r["synthetic"] else "acqp"
        print(f"  {name:<38}{te_kind:<7}{where:<11}" + stats_line("", s, 1, 0)[2:])
    if any(r["synthetic"] for r in results):
        print("  synth. = synthetic echo times (T2 values only approximate in ms)")
    for name, why in skipped:
        print(f"  SKIPPED  {name}: {why}")


def write_csv(results, path):
    """Write one row per image and slice (plus an ``all`` row) to a ';'-separated CSV."""
    path = Path(path).with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["image", "slice", "n", "mean_ms", "std_ms", "median_ms", "min_ms", "max_ms", "snr",
            "bi_share", "n_unreliable", "n_echoes", "te_source", "sigma"]

    def cell(v, digits):
        return f"{v:.{digits}f}" if v is not None and np.isfinite(v) else ""

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(cols)
        for r in results:
            lines = [(row["z"] + 1, row) for row in r["rows"]]   # slice number as in the viewer (from 1)
            if r["pooled"] is not None:
                lines.append(("all", r["pooled"]))
            for z, s in lines:
                w.writerow([r["name"], z, s["n"]]
                           + [cell(s[k], 4) for k in STAT_KEYS]
                           + [cell(s.get("snr"), 2), cell(s.get("share_bi"), 4), s.get("n_limit", 0),
                              r["shape"][3], r["source"], cell(r["sigma"], 2)])
    print(f"\n[CSV] {path}")


# ── Command line ──────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="quickstats.py",
        description="Quick T2 statistics (n, mean, SD, median, min, max) of NIfTI files, in the console.",
        epilog="Echo times: --acqp file, then --acqp folder (by name), then '<image>.acqp' next to "
               "the image, then synthetic. See the top of this file.",
    )
    p.add_argument("path", help="a NIfTI file (.nii, .nii.gz) or a folder of NIfTI files")
    p.add_argument("--acqp", metavar="PATH",
                   help="an ACQP file (used for every image) or a folder of ACQP files "
                        "(each image takes the one named like it)")
    p.add_argument("--all-slices", action="store_true",
                   help="every slice instead of the middle one: one row per slice, plus all slices together")
    p.add_argument("--slice", type=int, metavar="N", help="this slice (counted from 1, as in the viewer) instead of the middle one")
    p.add_argument("--te-step", type=float, default=DEFAULT_TE_STEP, metavar="MS",
                   help=f"spacing of the synthetic echo times (default {DEFAULT_TE_STEP:g} ms)")
    p.add_argument("--exclude-unreliable", action="store_true",
                   help="leave out the voxels whose T2 ended on a limit of the search range")
    p.add_argument("--csv", metavar="FILE", help="also write the results to a CSV file (';' separated)")
    p.add_argument("--recursive", action="store_true", help="with a folder: look in its sub-folders too")
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show what the viewer functions print (mask, noise) and the progress bars")
    args = p.parse_args(argv)
    if args.all_slices and args.slice is not None:
        p.error("--all-slices and --slice cannot be used together")
    if args.te_step <= 0:
        p.error("--te-step must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    target = Path(args.path).expanduser()
    if not target.exists():
        print(f"Error: '{target}' does not exist.")
        return 2
    if target.is_dir():
        files = list_niftis(target, args.recursive)
        if not files:
            print(f"Error: no .nii / .nii.gz file in '{target}'" + ("" if args.recursive else " (use --recursive for sub-folders)."))
            return 2
    elif is_nifti(target):
        files = [target]
    else:
        print(f"Error: '{target}' is not a .nii / .nii.gz file or a folder.")
        return 2

    acqp = None
    if args.acqp is not None:
        acqp = Path(args.acqp).expanduser()
        if not acqp.exists():
            print(f"Error: '{acqp}' does not exist.")
            return 2
        if acqp.is_file() and len(files) > 1:
            print(f"[note] the same ACQP file ({acqp.name}) is used for all {len(files)} images.")

    results, skipped = [], []
    for i, f in enumerate(files, 1):
        try:
            r = analyse(f, args, acqp)
        except SkipFile as exc:
            skipped.append((f.name, str(exc)))
            print(f"\n[{i}/{len(files)}] {f.name}\n  SKIPPED: {exc}")
            continue
        results.append(r)
        print_result(r, i, len(files))

    if len(files) > 1 and results:
        print_summary(results, skipped)
    if args.csv and results:
        write_csv(results, args.csv)
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
