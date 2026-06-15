"""
functions — core package for Sphene MRI T2 relaxometry.

Modules
-------
io
    NIfTI/ACQP loading and GUI dialogs for file selection and TE entry.
model
    Analytical T2 decay models and scipy-based fitting functions.
utils
    Goodness-of-fit metrics, noise estimation, tissue masking, and
    joblib/tqdm integration.
mapping
    Voxel-wise fitting workers and parallel execution via joblib.
display
    Interactive matplotlib viewer with slice navigation and analysis buttons.
"""