"""Parallel Python port of ``generate_homogenized_data.m`` + ``mstr_data_gen_main.m``.

The original MATLAB driver loops *serially* over every micro-structure image and
calls ``fluidHomogenization`` (~1.4 s each -> 164 min for 7000 samples in the
paper). Here we map the (embarrassingly parallel) per-sample homogenization over
all CPU cores with joblib, which cuts wall-clock time roughly by the core count.

I/O is kept compatible with the rest of the pipeline:
    input : mstr_images_{N}.mat   key 'mstr_images'  shape (num, nelx, nely),
            pixel 0 = solid, 1 = fluid.
    output: homogen_data_{N}.mat  keys 'mstr','c00','c11','c01','c10'
            (consumed by notebooks/train_vae_main.ipynb).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import scipy.io
from joblib import Parallel, delayed

from fluid_homogenization import fluid_homogenization


def homogenize_one(img: np.ndarray,
                   solid_perm: float = 1.0e6,
                   fluid_perm: float = 0.0,
                   phi_deg: float = 90.0) -> tuple[float, float, float, float]:
    """Homogenize a single (nelx, nely) image. Returns (c00, c11, c01, c10).

    Mirrors the MATLAB per-sample logic: if a cell is entirely fluid, a single
    centre pixel is turned solid to keep the system non-degenerate; the image is
    transposed to (nely, nelx) before homogenization.
    """
    img = np.asarray(img, dtype=float)
    nelx, nely = img.shape
    if np.all(img == 1):
        img = img.copy()
        img[nelx // 2, nelx // 2] = 0.0
    CH = fluid_homogenization(img.T, solid_perm=solid_perm,
                              fluid_perm=fluid_perm, phi_deg=phi_deg)
    return CH[0, 0], CH[1, 1], CH[0, 1], CH[1, 0]


def generate_homogenized_data(images: np.ndarray, n_jobs: int = -1,
                              verbose: int = 5) -> dict:
    """Homogenize a stack of images in parallel.

    Args:
        images: (num, nelx, nely) array, 0 = solid, 1 = fluid.
        n_jobs: joblib worker count (-1 = all cores).
        verbose: joblib verbosity.

    Returns:
        dict with arrays 'mstr', 'c00', 'c11', 'c01', 'c10'.
    """
    num = images.shape[0]
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=8)(
        delayed(homogenize_one)(images[i]) for i in range(num))
    dt = time.time() - t0
    results = np.asarray(results)  # (num, 4)
    print(f"Homogenized {num} samples in {dt:.1f} s "
          f"({dt / max(num,1):.3f} s/sample, n_jobs={n_jobs}).")
    return {
        'mstr': np.arange(1, num + 1).reshape(-1, 1).astype(float),
        'c00': results[:, 0:1],
        'c11': results[:, 1:2],
        'c01': results[:, 2:3],
        'c10': results[:, 3:4],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-num', type=int, default=1)
    ap.add_argument('--data-dir', default='../TOMAS/dataset')
    ap.add_argument('--n-jobs', type=int, default=-1)
    args = ap.parse_args()

    in_file = os.path.join(args.data_dir, f'mstr_images_{args.dataset_num}.mat')
    out_file = os.path.join(args.data_dir, f'homogen_data_{args.dataset_num}.mat')

    images = scipy.io.loadmat(in_file)['mstr_images']
    print(f"Loaded {in_file}: {images.shape}")
    data = generate_homogenized_data(images, n_jobs=args.n_jobs)
    scipy.io.savemat(out_file, data)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
