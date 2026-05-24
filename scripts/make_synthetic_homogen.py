"""Generate a *synthetic* homogenized permeability dataset.

The real homogenized tensor must be produced by the MATLAB script
``dataset/run_homogenization_dataset.m``.  This helper creates a
physically plausible *placeholder* so we can smoke-test the rest of the
pipeline (VAE training and multiscale topology optimization) on the
cloud while the user runs MATLAB locally.

The model used here is a very rough scaling law: solid-rich cells (low
fluid volume fraction) get small permeability, fluid-rich cells get
large permeability.  C00 and C11 are mildly anisotropic, controlled by
the super-shape parameters ``a``/``b``.
"""

import os
import sys

import numpy as np
import scipy.io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    dataset_num = 1
    dataset_dir = os.path.join(ROOT, "dataset")
    img_path = os.path.join(dataset_dir, f"mstr_images_{dataset_num}.mat")
    shape_path = os.path.join(dataset_dir, f"mstr_shape_parameters_{dataset_num}.mat")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(
            f"Missing {img_path}.  Run scripts/run_generate_data.py first."
        )

    images = scipy.io.loadmat(img_path)["mstr_images"]
    shape_params = scipy.io.loadmat(shape_path)["mstr_shape_parameters"]
    num_samples = images.shape[0]
    print(f"num_samples = {num_samples}, image grid = {images.shape[1:]}")

    fluid_fraction = images.reshape(num_samples, -1).mean(axis=1)
    a = shape_params[:, 0]
    b = shape_params[:, 1]

    rng = np.random.default_rng(42)

    base = np.clip(fluid_fraction, 1e-4, 1 - 1e-4)
    perm = 10 ** (4 * (base - 0.5))
    aniso = np.clip(a / np.clip(b, 1e-3, None), 0.25, 4.0)
    log_aniso = 0.25 * np.log10(aniso)

    c00 = perm * 10 ** (+log_aniso) * (1.0 + 0.05 * rng.standard_normal(num_samples))
    c11 = perm * 10 ** (-log_aniso) * (1.0 + 0.05 * rng.standard_normal(num_samples))
    c01 = 0.02 * perm * rng.standard_normal(num_samples)
    c10 = c01.copy()

    homogen = {
        "mstr": np.arange(1, num_samples + 1).reshape(-1, 1).astype(float),
        "c00": c00.reshape(-1, 1),
        "c11": c11.reshape(-1, 1),
        "c01": c01.reshape(-1, 1),
        "c10": c10.reshape(-1, 1),
    }

    out_path = os.path.join(dataset_dir, f"homogen_data_{dataset_num}.mat")
    scipy.io.savemat(out_path, homogen, do_compression=True)
    print(f"Wrote SYNTHETIC homogenization data -> {out_path}")
    print(
        "  c00 range:", f"{c00.min():.3e} .. {c00.max():.3e}",
        "  c11 range:", f"{c11.min():.3e} .. {c11.max():.3e}",
    )
    print(
        "REMINDER: replace this file with the real one produced by "
        "dataset/run_homogenization_dataset.m before publishing results."
    )


if __name__ == "__main__":
    main()
