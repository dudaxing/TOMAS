"""Headless VAE training (Algorithm 2) — scripted from notebooks/train_vae_main.ipynb.

Trains the variational auto-encoder on the paper's 10-feature micro-structure
dataset (6 shape params + C00 + C11 + perimeter + area) and saves:
  * ``TOMAS/vae/vae_net.pt``       - trained VAE weights
  * ``TOMAS/vae/nomalization.pt``  - {'max_feature','min_feature'} used to
                                     de-normalize the decoder output.

NOTE: the original notebook never persisted ``nomalization.pt`` even though the
multiscale-TO notebook loads it; we add that here so the downstream optimization
can run unattended.
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

import network          # noqa: E402  (TOMAS/vae/network.py)
import data_preprocess  # noqa: E402
import train_vae        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datagen-config", default=os.path.join(REPO, "notebooks", "datagen.yaml"))
    ap.add_argument("--vae-config", default=os.path.join(REPO, "notebooks", "vae_config.yaml"))
    ap.add_argument("--data-dir", default=os.path.join(REPO, "dataset"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "vae"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "..", "results", "vae"))
    ap.add_argument("--force", action="store_true", help="retrain even if weights exist")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs (smoke test)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--min-vf", type=float, default=0.02,
                    help="drop training samples with solid volume fraction below this")
    ap.add_argument("--min-perim", type=float, default=0.05,
                    help="drop training samples with perimeter below this")
    ap.add_argument("--lr-min", type=float, default=None,
                    help="cosine LR floor; set equal to --lr/config lr for a FLAT schedule "
                         "(overrides vae_config lr_min without editing it)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="mini-batch size for SGD (default None = full-batch). A finite value "
                         "(e.g. 512) converges to a much lower reconstruction error.")
    args = ap.parse_args()

    with open(args.datagen_config) as f:
        data_cfg = yaml.safe_load(f)
    with open(args.vae_config) as f:
        vae_cfg = yaml.safe_load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    dnum = data_cfg["DATASET"]["dataset_num"]
    d = args.data_dir
    shape_params = scipy.io.loadmat(os.path.join(d, f"mstr_shape_parameters_{dnum}.mat"))["mstr_shape_parameters"]
    homog = scipy.io.loadmat(os.path.join(d, f"homogen_data_{dnum}.mat"))
    area = scipy.io.loadmat(os.path.join(d, f"mstr_area_{dnum}.mat"))["mstr_area"]
    perim = scipy.io.loadmat(os.path.join(d, f"mstr_perim_{dnum}.mat"))["mstr_perim"]
    # scipy.io.savemat stores 1-D arrays as (1, N); normalize everything to (N, 1).
    c00, c11 = homog["c00"].reshape(-1, 1), homog["c11"].reshape(-1, 1)
    area = area.reshape(-1, 1)
    perim = perim.reshape(-1, 1)
    n_raw = shape_params.shape[0]

    # Drop degenerate micro-structures (near-empty / collapsed cells) that
    # otherwise dominate the high-permeability extreme and distort the 2-D
    # latent space. Only affects VAE training data (TO uses the decoder).
    keep = ((area[:, 0] >= args.min_vf) & (perim[:, 0] >= args.min_perim) &
            np.isfinite(c00[:, 0]) & np.isfinite(c11[:, 0]) &
            (c00[:, 0] > 0) & (c11[:, 0] > 0))
    shape_params, c00, c11, area, perim = (shape_params[keep], c00[keep],
                                           c11[keep], area[keep], perim[keep])
    print(f"Loaded dataset #{dnum}: kept {int(keep.sum())}/{n_raw} samples "
          f"(dropped {n_raw - int(keep.sum())} with vf<{args.min_vf} or perim<{args.min_perim})")

    # 10 features (paper): [a,b,m,n1,n2,n3] + C00 + C11 + perim + area.
    # The near-constant center coords (cx,cy = shape_params[:,6:8]) are dropped.
    mstr_data = torch.tensor(np.hstack((
        shape_params, c00, c11, perim, area))).double()
    normalization_types = ([data_preprocess.NomalizationType.LINEAR] * 8 +
                           [data_preprocess.NomalizationType.LOG] * 2 +
                           [data_preprocess.NomalizationType.LINEAR] * 2)
    normalized, max_feature, min_feature = data_preprocess.stack_train_data(
        mstr_data, normalization_types)
    num_samples, num_features = normalized.shape
    assert num_features == 12, num_features

    net_cfg = vae_cfg["NETWORK"]
    opt_cfg = vae_cfg["OPTIMIZATION"]
    vae_params = network.VAE_Params(input_dim=num_features,
                                    encoder_hidden_dim=net_cfg["encoder_hidden_dim"],
                                    latent_dim=net_cfg["latent_dim"],
                                    decoder_hidden_dim=net_cfg["decoder_hidden_dim"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Training device: {device}")
    vae_net = network.VariationalAutoencoder(vae_params=vae_params).to(device)

    num_epochs = args.epochs or opt_cfg["num_epochs"]
    weights_file = os.path.join(args.out_dir, "vae_net.pt")
    if args.force or not os.path.isfile(weights_file):
        print(f"Training VAE: {num_epochs} epochs, lr={opt_cfg['lr']}, "
              f"kl={opt_cfg['kl_factor']} ...", flush=True)
        train_vae.train_autoencoder(
            vae=vae_net, train_data=normalized.to(device),
            num_epochs=num_epochs, kl_factor=opt_cfg["kl_factor"],
            lr=opt_cfg["lr"], save_file=weights_file,
            print_every=max(1, num_epochs // 20),
            lr_min=(args.lr_min if args.lr_min is not None else opt_cfg.get("lr_min")),
            batch_size=args.batch_size)
        # Re-save a CPU state_dict so the (CPU-based) TO step loads it portably.
        vae_net.to("cpu")
        torch.save(vae_net.state_dict(), weights_file)
    else:
        print("Weights exist; loading (use --force to retrain).")
    vae_net.to("cpu")
    vae_net.encoder.is_training = False
    vae_net.load_state_dict(torch.load(weights_file, map_location="cpu"))
    vae_net.eval()

    # Persist normalization for the downstream TO step.
    torch.save({"max_feature": max_feature, "min_feature": min_feature},
               os.path.join(args.out_dir, "nomalization.pt"))
    print(f"Saved {weights_file} and nomalization.pt")

    # Latent-space scatter of the training data (companion to Fig 7).
    z = vae_net.encoder(normalized).detach().numpy()
    plt.figure(figsize=(5, 4))
    plt.scatter(z[:, 0], z[:, 1], s=4, alpha=0.4)
    plt.xlabel("$z_1$"); plt.ylabel("$z_2$")
    plt.title(f"VAE latent space ({num_samples} micro-structures)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, "latent_space_scatter.png"), dpi=200)

    # Overall reconstruction error on the training set (sanity / Table-1 flavour).
    recon = data_preprocess.stack_vae_output(vae_net(normalized), max_feature,
                                             min_feature, normalization_types)
    recon = recon.detach().numpy()
    true = mstr_data.numpy()
    names = ["a", "b", "m", "n1", "n2", "n3", "cx", "cy", "C00", "C11", "perim", "area"]
    print("Mean abs %% reconstruction error per feature (over samples with |true|>1e-3):")
    pct_means = {}
    for i, nm in enumerate(names):
        m = np.abs(true[:, i]) > 1e-3
        e = 100.0 * np.abs(recon[m, i] - true[m, i]) / np.abs(true[m, i])
        pct_means[nm] = float(np.mean(e))
        print(f"  {nm:>5}: {pct_means[nm]:8.3f}%  (n={int(m.sum())})")
    np.save(os.path.join(args.results_dir, "recon_pct_error.npy"), pct_means)
    print("Done.")


if __name__ == "__main__":
    main()
