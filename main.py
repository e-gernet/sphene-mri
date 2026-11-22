"""
Entry point for the Sphene MRI T2 relaxometry pipeline.

Workflow
--------
1. Load a NIfTI file (GUI file picker)
2. Define echo times — from an ACQP file, manual entry, or synthetic values
3. Compute a binary tissue mask (Rician noise threshold)
4. Launch the interactive viewer (mono and bi T2 fits by maximum likelihood,
   see ``functions/model.py``)

Usage
-----
Run from the project root with::

    pixi run python main.py

or directly::

    python main.py
"""

import argparse

import numpy as np

from functions.display import display_slice
from functions.io import (
    choose_acqp,
    choose_nifti,
    enter_te,
    handle_acqp,
    load_acqp,
    load_nifti,
    voxel_size_mm,
)
from functions.utils import compute_mask, filter_data

# Pre-processing filter applied before masking and fitting.
# "none" | "gaussian_spatial" | "savgol_temporal"
# Keep "none" for the T2 fits: they model the noise of the raw magnitude images
# (Rician), and any filter changes that noise.
FILTER_METHOD = "none"

# Number of first echoes left out of the fits (0 = use them all). The first echoes
# of a spin-echo train can be slightly off (imperfect refocusing pulses, stimulated
# echoes). The "Noise" window of the viewer prints the shift of each of the first three
# echoes in % of the signal. Below about 5 % keep 0. Above, leave out EVERY shifted echo
# (the console says how many), never fewer: leaving out only one echo when the next one
# is also shifted is worse than leaving out none (section 7 of the validation report).
SKIP_FIRST_ECHOES = 0


def main():
    """Run the full T2 relaxometry analysis pipeline."""

    parser = argparse.ArgumentParser(description="Sphene-MRI T2 relaxometry pipeline")
    parser.add_argument(
        "--compare-filters", action="store_true",
        help="Compare filtering strategies on the middle slice instead of "
             "launching the viewer.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="Compute device for filtering (gaussian_spatial only). "
             "Falls back to CPU automatically if no CUDA device / cupy "
             "install is found. Default: cpu.",
    )
    args = parser.parse_args()

    # 1. Load NIfTI
    nifti_file = choose_nifti()
    data, img  = load_nifti(nifti_file)

    # Voxel size in mm, from the NIfTI header — previously loaded and
    # immediately discarded, meaning nothing downstream (maps, CSV export,
    # popups) had any notion of real-world distance, only raw voxel
    # indices. Kept here and threaded through to the viewer/export so a
    # capillary width or sillon size can eventually be reported in mm
    # instead of voxels.
    voxel_dims, unit, unit_known = voxel_size_mm(img)   # always in mm
    print(
        f"[Geometry] Voxel size: {voxel_dims[0]:.3f} x {voxel_dims[1]:.3f} "
        f"x {voxel_dims[2]:.3f} mm"
    )
    if not unit_known:
        print(
            f"[Geometry] ⚠️  The header gives no spatial unit ('{unit}'): "
            "the sizes are taken as mm."
        )
    elif unit != "mm":
        print(f"[Geometry] The header unit is '{unit}': sizes converted to mm.")
    if tuple(voxel_dims) == (1.0, 1.0, 1.0):
        print(
            "[Geometry] ⚠️  x/y/z = (1.0, 1.0, 1.0) mm. This can be a real "
            "isotropic 1mm resolution, or nibabel's silent default when the "
            "header has no usable pixdim. Double-check against the "
            "acquisition source (ParaVision, etc.) before trusting x_mm / "
            "y_mm / z_mm in the export."
        )

    # 2. Echo times
    te_choice = handle_acqp(data)
    if te_choice["choice"] == "load":
        acqp_file = choose_acqp()
        te_values = load_acqp(acqp_file)
    elif te_choice["choice"] == "manual":
        te_values = enter_te(data)
    else:
        n_echos   = data.shape[3]
        te_values = np.arange(1, n_echos + 1) * 3.0
        print(
            f"[TE] No file provided — synthetic TEs: "
            f"{te_values[0]:.1f} to {te_values[-1]:.1f} ms"
        )

    # 3. Optional: compare filter strategies instead of running the viewer
    if args.compare_filters:
        from functions.filter_compare import compare_filters
        compare_filters(data, te_values, z=data.shape[2] // 2, device=args.device)
        return

    # 4. Pre-processing filter (applied before masking, unlike the old
    #    display-only smoothing in the "Utils" button)
    data = filter_data(
        data, method=FILTER_METHOD, sigma=1.0, window=5, poly=2, device=args.device
    )
    if FILTER_METHOD != "none":
        print(f"[Filter] Applied '{FILTER_METHOD}' before masking and fitting.")
        print("[Filter] WARNING: the fits assume the noise of raw magnitude images (Rician). "
              "A filter changes that noise; see the Rayleigh check printed by the viewer.")

    # 4b. Optional: leave the first echoes out of the analysis
    if SKIP_FIRST_ECHOES > 0:
        data = data[..., SKIP_FIRST_ECHOES:]
        te_values = np.asarray(te_values)[SKIP_FIRST_ECHOES:]
        print(f"[TE] The first {SKIP_FIRST_ECHOES} echo(es) are left out: "
              f"{len(te_values)} echoes used, from {te_values[0]:.1f} to {te_values[-1]:.1f} ms.")

    # 5. Tissue mask
    mask = compute_mask(data, method="rician")

    # 6. Interactive viewer
    display_slice(data, te_values, mask=mask, voxel_dims=voxel_dims)


if __name__ == "__main__":
    main()