"""Run the Python Stokes-Brinkman homogenizer over the generated dataset.

Mirrors ``dataset/generate_homogenized_data.m``: reads
``dataset/mstr_images_<n>.mat`` and writes ``dataset/homogen_data_<n>.mat``
with the same fields (mstr, c00, c11, c01, c10) so downstream Python code
(``scripts/train_vae.py``) sees an identical interface.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dataset"))

from fluid_homogenization import fluid_homogenization  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=int,
        default=1,
        help="dataset number used in mstr_images_<n>.mat (default: 1)",
    )
    parser.add_argument(
        "--solid-zeta",
        type=float,
        default=1e6,
        help="inverse permeability of solid pixels (default: 1e6, matches MATLAB)",
    )
    parser.add_argument(
        "--fluid-zeta",
        type=float,
        default=0.0,
        help="inverse permeability of fluid pixels (default: 0.0)",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=90.0,
        help="cell skew angle in degrees (default: 90)",
    )
    parser.add_argument(
        "--lx", type=float, default=1.0, help="cell length in x (default: 1.0)"
    )
    parser.add_argument(
        "--ly", type=float, default=1.0, help="cell length in y (default: 1.0)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = os.path.join(ROOT, "dataset", f"mstr_images_{args.dataset}.mat")
    out_path = os.path.join(ROOT, "dataset", f"homogen_data_{args.dataset}.mat")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(
            f"{in_path} not found.  Run scripts/run_generate_data.py first."
        )

    data = scipy.io.loadmat(in_path)
    images = np.asarray(data["mstr_images"])  # (num_samples, nelx, nely)
    num_samples, nelx, nely = images.shape
    print(
        f"Loaded {in_path} -> {num_samples} micro-cells of {nelx}x{nely}"
    )

    zeta = (args.solid_zeta, args.fluid_zeta)

    mstr = np.zeros((num_samples, 1))
    c00 = np.zeros((num_samples, 1))
    c11 = np.zeros((num_samples, 1))
    c01 = np.zeros((num_samples, 1))
    c10 = np.zeros((num_samples, 1))

    t_start = time.time()
    for i in range(num_samples):
        cell = images[i]  # shape (nelx, nely)

        # MATLAB version replaces a single pixel with solid when the cell is
        # completely fluid -- otherwise the periodic-Stokes problem has a null
        # space (rigid-body translation) and the solve becomes singular.
        if np.all(cell == 1):
            cell = cell.copy()
            cell[nelx // 2, nely // 2] = 0

        # MATLAB reshape(data(i,:,:), nelx, nely) then transpose -> (nely, nelx)
        x = cell.T.astype(np.float64)

        ch = fluid_homogenization(args.lx, args.ly, zeta, args.phi, x)

        mstr[i, 0] = i + 1  # match MATLAB 1-based numbering
        c00[i, 0] = ch[0, 0]
        c11[i, 0] = ch[1, 1]
        c01[i, 0] = ch[0, 1]
        c10[i, 0] = ch[1, 0]

        if (i + 1) % 10 == 0 or i == num_samples - 1:
            elapsed = time.time() - t_start
            print(
                f"  cell {i + 1:3d}/{num_samples}  c00={ch[0,0]:.3e}  "
                f"c11={ch[1,1]:.3e}  c01={ch[0,1]:+.2e}   "
                f"elapsed {elapsed:.1f}s"
            )

    scipy.io.savemat(
        out_path,
        {
            "mstr": mstr,
            "c00": c00,
            "c11": c11,
            "c01": c01,
            "c10": c10,
        },
        do_compression=True,
    )
    print(
        f"\nWrote {out_path}\n"
        f"  c00 range: {c00.min():.3e} .. {c00.max():.3e}\n"
        f"  c11 range: {c11.min():.3e} .. {c11.max():.3e}\n"
        f"  |c01| max: {np.abs(c01).max():.3e} (should be small for symmetric shapes)"
    )


if __name__ == "__main__":
    main()
