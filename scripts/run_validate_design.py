"""True-FEA validation of an optimized design (paper §3.3, §3.5).

The decoder predicts each micro-structure's permeability / contact area. To
check accuracy, the paper re-computes the "true" values by reconstructing the
actual super-shapes of the final design, homogenizing them (here with the
ported parallel solver) and re-solving the global Stokes problem. This script
reproduces that: it loads a ``design.npz`` produced by run_to.py and the same
config, and reports decoder-vs-true dissipated power and contact area.
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys

import numpy as np
import torch
import yaml
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "TOMAS")
for _p in ("fluid_TO", "vae", "dataset", ""):
    sys.path.insert(0, os.path.join(REPO, _p))
sys.path.insert(0, os.path.join(HERE, "..", "dataset_py"))

import dataset.supershape as ss                # noqa: E402
import material_constants, fluid_mesher, fluid_bcs, fluid_material, fluid_fe  # noqa: E402
from fluid_homogenization import fluid_homogenization  # noqa: E402


def _true_props(params8, nel=150):
    sp = ss.SuperShapes(*[np.array([params8[i]]) for i in range(8)])
    polys, pruned = ss.super_shape_to_shapely_polygon(sp)
    if len(polys) == 0:
        return 1e-6, 1e-6, 0.0, 0.0
    img = ss.project_shapely_polygons_to_density(polys, nel, nel, True)[0]
    if np.all(img == 1):
        img = img.copy(); img[nel // 2, nel // 2] = 0.0
    CH = fluid_homogenization(img.T)
    area = polys[0].area / (pruned.domain_length_x * pruned.domain_length_y)
    perim = polys[0].length / (pruned.domain_length_x + pruned.domain_length_y)
    return float(CH[0, 0]), float(CH[1, 1]), float(area), float(perim)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--design", required=True, help="path to design.npz")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    d = np.load(args.design, allow_pickle=True)
    sp = d["shape_params"]            # (N, 8)
    theta = d["theta"]                # (N,)
    elem_dx = float(d["elem_dx"]); perim_scale = float(d["perim_scale"])
    N = sp.shape[0]
    print(f"Re-homogenizing {N} micro-structures of the final design ...")

    res = Parallel(n_jobs=args.n_jobs, verbose=5, batch_size=8)(
        delayed(_true_props)(sp[i]) for i in range(N))
    res = np.asarray(res)
    true_C00, true_C11, true_area, true_perim = res[:, 0], res[:, 1], res[:, 2], res[:, 3]

    # True contact area (same scaling as the optimization constraint field).
    true_contact_area = float(np.sum(elem_dx * perim_scale * true_perim))
    decoder_contact_area = float(np.sum(d["constraint_field"])) \
        if str(d["constraint_type"]) == "PERIMETER" else float("nan")

    # Rebuild the solver and compute the true dissipated power.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    bb = fluid_mesher.BoundingBox(**{k: cfg["BOUNDING_BOX"][k]
                                     for k in ("x_min", "x_max", "y_min", "y_max")})
    mesh = fluid_mesher.fluid_mesher(cfg["MESH"]["nelx"], cfg["MESH"]["nely"], bb)
    bc = fluid_bcs.get_dirichlet_bc_and_fixed_dofs(
        mesh, cfg["BOUNDARY_CONDITIONS"]["char_velocity"],
        fluid_bcs.FluidSampleProblems[cfg["BOUNDARY_CONDITIONS"]["example"]])
    mat = material_constants.MaterialConstants(
        kinematic_viscosity=cfg["MATERIAL_CONSTANTS"]["kinematic_viscosity"])
    fmat = fluid_material.FluidMaterial(mesh.elem_dx, mesh.elem_dy, mat)
    solver = fluid_fe.FluidSolver(mesh, bc, fixture_const=cfg["BOUNDARY_CONDITIONS"]["fixture_const"])

    with torch.no_grad():
        true_power, _ = solver.fluid_objective_function(
            fmat, torch.tensor(true_C00 + 1e-6), torch.tensor(true_C11 + 1e-6),
            torch.tensor(theta))
        dec_power, _ = solver.fluid_objective_function(
            fmat, torch.tensor(d["C00"] + 1e-6), torch.tensor(d["C11"] + 1e-6),
            torch.tensor(theta))
    true_power = float(true_power); dec_power = float(dec_power)

    print("\n==== Decoder vs True (global FEA / homogenization) ====")
    print(f"  dissipated power : decoder={dec_power:.3f}   true={true_power:.3f}   "
          f"err={100*abs(dec_power-true_power)/true_power:.2f}%")
    if not np.isnan(decoder_contact_area):
        print(f"  contact area     : decoder={decoder_contact_area:.3f}   "
              f"true={true_contact_area:.3f}   "
              f"err={100*abs(decoder_contact_area-true_contact_area)/true_contact_area:.2f}%")
    out = os.path.join(os.path.dirname(args.design), "validation.json")
    import json
    json.dump({"decoder_power": dec_power, "true_power": true_power,
               "decoder_contact_area": decoder_contact_area,
               "true_contact_area": true_contact_area}, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
