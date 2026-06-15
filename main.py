"""
Entry point for the Sphene MRI T2 relaxometry pipeline.

Workflow
--------
1. Load a NIfTI file (GUI file picker)
2. Define echo times — from an ACQP file, manual entry, or synthetic values
3. Compute a binary tissue mask (Rician noise threshold)
4. Launch the interactive viewer

Usage
-----
Run from the project root with::

    pixi run python main.py

or directly::

    python main.py
"""

import numpy as np

from functions.io import (
    choose_nifti, load_nifti,
    handle_acqp, choose_acqp, load_acqp, enter_te,
)
from functions.display import display_slice
from functions.utils import compute_mask


def main():
    """Run the full T2 relaxometry analysis pipeline."""

    # 1. Load NIfTI
    nifti_file = choose_nifti()
    data, img  = load_nifti(nifti_file)

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

    # 3. Tissue mask
    mask = compute_mask(data, method="rician")

    # 4. Interactive viewer
    display_slice(data, te_values, mask=mask)


if __name__ == "__main__":
    main()