"""End-to-end, parallel dataset generation for TOMAS (replaces the MATLAB step).

Pipeline (Algorithm 1 in the paper):
  1. sample N random super-shapes (uniform over the parameter ranges),
  2. convert to shapely polygons (pruning shapes that leave the unit cell),
  3. compute contact area (perimeter) and solid area (volume fraction),
  4. rasterize each shape to a 150x150 density image (0 = solid, 1 = fluid),
  5. homogenize every image **in parallel** to get the permeability tensor,
  6. save the four .mat files consumed by notebooks/train_vae_main.ipynb:
        mstr_shape_parameters_{N}.mat, mstr_area_{N}.mat,
        mstr_perim_{N}.mat, homogen_data_{N}.mat

This fuses the Python generate_data notebook and the (now Pythonized + parallel)
MATLAB homogenization so the full 150x150 image stack never has to be written to
disk (~1.3 GB for 7000 samples).
"""

from __future__ import annotations

import argparse
import os

# One MKL/BLAS thread per process: each joblib worker runs a single-threaded
# PARDISO solve and we parallelize across samples instead (avoids 32xN
# oversubscription). Must be set before numpy/pypardiso import.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time

import numpy as np
import scipy.io
import yaml
from joblib import Parallel, delayed

# Repo super-shape geometry utilities.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TOMAS', 'dataset'))
import supershape as ss  # noqa: E402

from fluid_homogenization import fluid_homogenization  # noqa: E402


def _homogenize_one(img: np.ndarray) -> tuple[float, float, float, float]:
    img = np.asarray(img, dtype=float)
    nelx = img.shape[0]
    if np.all(img == 1):
        img = img.copy()
        img[nelx // 2, nelx // 2] = 0.0
    CH = fluid_homogenization(img.T)
    return CH[0, 0], CH[1, 1], CH[0, 1], CH[1, 0]


def build_extents(sy: dict) -> ss.SuperShapeExtents:
    return ss.SuperShapeExtents(
        a=ss.Extents(sy['min_a'], sy['max_a']),
        b=ss.Extents(sy['min_b'], sy['max_b']),
        m=ss.Extents(sy['min_m'], sy['max_m']),
        n1=ss.Extents(sy['min_n1'], sy['max_n1']),
        n2=ss.Extents(sy['min_n2'], sy['max_n2']),
        n3=ss.Extents(sy['min_n3'], sy['max_n3']),
        center_x=ss.Extents(sy['min_cx'], sy['max_cx']),
        center_y=ss.Extents(sy['min_cy'], sy['max_cy']),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='../TOMAS/notebooks/datagen.yaml')
    ap.add_argument('--num-samples', type=int, default=None,
                    help='override DATASET.num_samples from the config')
    ap.add_argument('--data-dir', default='../TOMAS/dataset')
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--save-images', action='store_true',
                    help='also write the (large) mstr_images_{N}.mat file')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    num_samples = args.num_samples or cfg['DATASET']['num_samples']
    dataset_num = cfg['DATASET']['dataset_num']
    nelx, nely = cfg['MESH']['nelx'], cfg['MESH']['nely']
    extents = build_extents(cfg['SUPERSHAPE'])
    seed = cfg['SUPERSHAPE'].get('shape_seed', 27)

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"Generating {num_samples} super-shapes (seed={seed}) at "
          f"{nelx}x{nely}, n_jobs={args.n_jobs} ...")

    # 1-3. shapes -> polygons -> area/perimeter
    shape_param = ss.generate_random_super_shapes(num_samples, extents, seed=seed)
    polygons, pruned = ss.super_shape_to_shapely_polygon(shape_param)
    n_kept = len(polygons)
    print(f"  kept {n_kept}/{num_samples} shapes after bounding-box pruning")

    perim = ss.compute_shapely_polygon_perimeter(polygons)
    area = ss.compute_shapely_polygon_area(polygons)
    norm_area = area / (pruned.domain_length_x * pruned.domain_length_y)
    norm_perim = perim / (pruned.domain_length_x + pruned.domain_length_y)
    shape_params = pruned.to_stacked_array()

    # 4. rasterize (0 = solid inside shape, 1 = fluid outside)
    t0 = time.time()
    images = ss.project_shapely_polygons_to_density(polygons, nelx, nely, True)
    print(f"  rasterized {n_kept} images in {time.time()-t0:.1f}s -> {images.shape}")

    # 5. parallel homogenization
    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5, batch_size=8)(
        delayed(_homogenize_one)(images[i]) for i in range(n_kept))
    dt = time.time() - t0
    results = np.asarray(results)
    print(f"  homogenized {n_kept} samples in {dt:.1f}s "
          f"({dt/max(n_kept,1):.3f} s/sample)")

    # 6. save
    d = args.data_dir
    scipy.io.savemat(os.path.join(d, f'mstr_shape_parameters_{dataset_num}.mat'),
                     {'mstr_shape_parameters': shape_params})
    scipy.io.savemat(os.path.join(d, f'mstr_area_{dataset_num}.mat'),
                     {'mstr_area': norm_area})
    scipy.io.savemat(os.path.join(d, f'mstr_perim_{dataset_num}.mat'),
                     {'mstr_perim': norm_perim})
    scipy.io.savemat(os.path.join(d, f'homogen_data_{dataset_num}.mat'), {
        'mstr': np.arange(1, n_kept + 1).reshape(-1, 1).astype(float),
        'c00': results[:, 0:1], 'c11': results[:, 1:2],
        'c01': results[:, 2:3], 'c10': results[:, 3:4]})
    if args.save_images:
        scipy.io.savemat(os.path.join(d, f'mstr_images_{dataset_num}.mat'),
                         {'mstr_images': images})
    print(f"Saved dataset #{dataset_num} ({n_kept} samples) to {d}")


if __name__ == "__main__":
    main()
