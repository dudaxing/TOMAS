"""Latent-space experiments: ideal micro-structure (3.1, Fig 10), latent
density (Fig 7) and reconstruction accuracy (Table 1).

* 3.1 Ideal micro-structure: sample the decoder on a 200x200 latent grid,
  keep micro-structures with solid volume fraction 0.25 +/- 0.001 and pick the
  one with the highest trace(C) = C00 + C11. Prints its latent coordinate
  (feed to run_to.py --fix-latent for the bent-pipe experiment 3.2) and shape
  parameters. Saves Fig 10.
* Table 1: percentage reconstruction error for points inside the dataset
  (encode->decode) and for new decoder-generated points (decode->reconstruct
  the actual super-shape->homogenize with the ported solver->compare).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.io
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "TOMAS")
for _p in ("vae", "dataset", ""):
    sys.path.insert(0, os.path.join(REPO, _p))
sys.path.insert(0, os.path.join(HERE, "..", "dataset_py"))

import network, data_preprocess          # noqa: E402
import supershape as ss                  # noqa: E402
from fluid_homogenization import fluid_homogenization  # noqa: E402

NORM = ([data_preprocess.NomalizationType.LINEAR] * 8 +
        [data_preprocess.NomalizationType.LOG] * 2 +
        [data_preprocess.NomalizationType.LINEAR] * 2)
F = data_preprocess.VAE_Fields
CXY_CONST = 1e-5


def load_vae(vae_dir):
    p = network.VAE_Params(input_dim=10, encoder_hidden_dim=600, latent_dim=2,
                           decoder_hidden_dim=600)
    vae = network.VariationalAutoencoder(vae_params=p)
    vae.encoder.is_training = False
    vae.load_state_dict(torch.load(os.path.join(vae_dir, "vae_net.pt")))
    vae.eval()
    n = torch.load(os.path.join(vae_dir, "nomalization.pt"))
    return vae, n["max_feature"], n["min_feature"]


def decode(vae, z, mx, mn):
    out = data_preprocess.stack_vae_output(vae.decoder(z), mx, mn, NORM)
    return out


def normalize_with(data, mx, mn):
    """Normalize using the *training* feature min/max (from nomalization.pt) so
    encodings stay in-distribution. LOG features are log10-scaled first."""
    out = data.clone()
    for i, nt in enumerate(NORM):
        col = torch.log10(data[:, i]) if nt == data_preprocess.NomalizationType.LOG else data[:, i]
        out[:, i] = (col - mn[i]) / (mx[i] - mn[i])
    return out


def homogenize_shape(params8, nel=150):
    """Reconstruct a single super-shape from its 8 params and homogenize it."""
    sp = ss.SuperShapes(*[np.array([params8[i]]) for i in range(8)])
    polys, pruned = ss.super_shape_to_shapely_polygon(sp)
    if len(polys) == 0:
        return None
    img = ss.project_shapely_polygons_to_density(polys, nel, nel, True)[0]
    if np.all(img == 1):
        img = img.copy(); img[nel // 2, nel // 2] = 0.0
    CH = fluid_homogenization(img.T)
    area = polys[0].area / (pruned.domain_length_x * pruned.domain_length_y)
    perim = polys[0].length / (pruned.domain_length_x + pruned.domain_length_y)
    return CH[0, 0], CH[1, 1], area, perim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vae-dir", default=os.path.join(REPO, "vae"))
    ap.add_argument("--data-dir", default=os.path.join(REPO, "dataset"))
    ap.add_argument("--datagen-config", default=os.path.join(REPO, "notebooks", "datagen.yaml"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "latent"))
    ap.add_argument("--grid", type=int, default=200)
    ap.add_argument("--target-vf", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vae, mx, mn = load_vae(args.vae_dir)

    # ---------- 3.1 Ideal micro-structure selection (Fig 10) ----------
    g = np.linspace(-3, 3, args.grid)
    Z1, Z2 = np.meshgrid(g, g)
    Z = torch.tensor(np.stack([Z1.ravel(), Z2.ravel()], axis=1)).double()
    with torch.no_grad():
        out = decode(vae, Z, mx, mn).numpy()
    vf = out[:, F.shape_area.value]
    trace = out[:, F.homog_c00.value] + out[:, F.homog_c11.value]
    mask = np.abs(vf - args.target_vf) <= 0.001
    print(f"[3.1] {mask.sum()} latent points with vf={args.target_vf}+/-0.001")
    idx_pool = np.where(mask)[0]
    best = idx_pool[np.argmax(trace[idx_pool])]
    z_star = Z[best].numpy()
    mstr_star = out[best, :6]
    print(f"  M* latent z=({z_star[0]:.4f}, {z_star[1]:.4f})")
    print(f"  M* shape params a={mstr_star[0]:.4f} b={mstr_star[1]:.4f} m={mstr_star[2]:.4f} "
          f"n1={mstr_star[3]:.4f} n2={mstr_star[4]:.4f} n3={mstr_star[5]:.4f}")
    print(f"  M* C00={out[best,F.homog_c00.value]:.4e} C11={out[best,F.homog_c11.value]:.4e} "
          f"vf={vf[best]:.4f}")
    np.save(os.path.join(args.out_dir, "M_star.npy"),
            {"z": z_star, "shape_params": mstr_star})

    plt.figure(figsize=(5, 4))
    plt.scatter(Z1.ravel()[mask], Z2.ravel()[mask], c=trace[mask], cmap="viridis", s=8)
    plt.colorbar(label="trace(C) = C00+C11")
    plt.scatter([z_star[0]], [z_star[1]], marker="*", s=300, c="red", label="M*")
    plt.xlabel("$z_1$"); plt.ylabel("$z_2$"); plt.legend()
    plt.title(f"Latent coords with solid vf $\\approx$ {args.target_vf} (Fig 10)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "fig10_ideal_mstr.png"), dpi=200)
    plt.close()

    # ---------- Fig 7: latent density of the training data ----------
    dnum = yaml.safe_load(open(args.datagen_config))["DATASET"]["dataset_num"]
    sp = scipy.io.loadmat(os.path.join(args.data_dir, f"mstr_shape_parameters_{dnum}.mat"))["mstr_shape_parameters"]
    h = scipy.io.loadmat(os.path.join(args.data_dir, f"homogen_data_{dnum}.mat"))
    area = scipy.io.loadmat(os.path.join(args.data_dir, f"mstr_area_{dnum}.mat"))["mstr_area"]
    perim = scipy.io.loadmat(os.path.join(args.data_dir, f"mstr_perim_{dnum}.mat"))["mstr_perim"]
    # Match the training filter + normalization so the encodings are in-distribution.
    c00v, c11v = h["c00"].reshape(-1, 1), h["c11"].reshape(-1, 1)
    areav, perimv = area.reshape(-1, 1), perim.reshape(-1, 1)
    keep = ((areav[:, 0] >= 0.02) & (perimv[:, 0] >= 0.05) &
            (c00v[:, 0] > 0) & (c11v[:, 0] > 0))
    sp, c00v, c11v, areav, perimv = sp[keep], c00v[keep], c11v[keep], areav[keep], perimv[keep]
    data = torch.tensor(np.hstack((sp, c00v, c11v, perimv, areav))).double()
    normalized = normalize_with(data, mx, mn)
    with torch.no_grad():
        zdata = vae.encoder(normalized).numpy()
    plt.figure(figsize=(5, 4))
    plt.hist2d(zdata[:, 0], zdata[:, 1], bins=60, cmap="magma")
    plt.colorbar(label="density"); plt.xlabel("$z_1$"); plt.ylabel("$z_2$")
    plt.title("Latent space density (Fig 7)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "fig7_latent_density.png"), dpi=200)
    plt.close()

    # ---------- Table 1: reconstruction accuracy ----------
    rng = np.random.default_rng(0)
    print("\n[Table 1] Reconstruction % error")
    print(f"{'point':>6} {'type':>10} {'dC00%':>8} {'dC11%':>8} {'dvf%':>8} {'dperim%':>8}")
    rows = []
    # in-dataset points (encode -> decode), compare to dataset truth
    for k, i in enumerate(rng.choice(len(sp), 4, replace=False)):
        with torch.no_grad():
            rec = decode(vae, vae.encoder(normalized[i:i + 1]), mx, mn).numpy()[0]
        t = data[i].numpy()
        e = lambda a, b: 100.0 * abs(a - b) / (abs(b) + 1e-12)
        row = [e(rec[F.homog_c00.value], t[F.homog_c00.value]),
               e(rec[F.homog_c11.value], t[F.homog_c11.value]),
               e(rec[F.shape_area.value], t[F.shape_area.value]),
               e(rec[F.shape_perim.value], t[F.shape_perim.value])]
        rows.append(("in", row))
        print(f"{chr(65+k):>6} {'in-data':>10} {row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f} {row[3]:8.2f}")
    # new decoder-generated points: decode -> rebuild shape -> homogenize -> compare
    for k in range(4):
        z = torch.tensor(rng.uniform(-2, 2, (1, 2))).double()
        with torch.no_grad():
            rec = decode(vae, z, mx, mn).numpy()[0]
        truth = homogenize_shape(np.concatenate([rec[:6], [CXY_CONST, CXY_CONST]]))
        if truth is None:
            continue
        tc00, tc11, tarea, tperim = truth
        e = lambda a, b: 100.0 * abs(a - b) / (abs(b) + 1e-12)
        row = [e(rec[F.homog_c00.value], tc00), e(rec[F.homog_c11.value], tc11),
               e(rec[F.shape_area.value], tarea), e(rec[F.shape_perim.value], tperim)]
        rows.append(("new", row))
        print(f"{chr(69+k):>6} {'new':>10} {row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f} {row[3]:8.2f}")
    np.save(os.path.join(args.out_dir, "table1_errors.npy"),
            np.array(rows, dtype=object), allow_pickle=True)
    print(f"\nSaved latent-space results to {args.out_dir}")


if __name__ == "__main__":
    main()
