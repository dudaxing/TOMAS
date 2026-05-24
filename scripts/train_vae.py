"""Runs the steps of notebooks/train_vae_main.ipynb as a plain script.

Reads the dataset under ``/workspace/dataset/`` and trains the
``VariationalAutoencoder`` defined in ``vae/network.py``.  The number of
epochs is configurable via ``--epochs``, allowing both a quick smoke
test (e.g. ``--epochs 200``) and a full training run that matches the
notebook defaults (``--epochs 17000``).

The script also persists the normalisation min/max tensors to
``vae/nomalization.pt``, which ``notebooks/multiscale_TO_main.ipynb``
expects to load before the topology-optimization stage.
"""

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset"))
sys.path.insert(0, os.path.join(ROOT, "vae"))

import network  # noqa: E402
import data_preprocess  # noqa: E402
import train_vae  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override number of epochs (defaults to vae_config.yaml)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="train even if a saved vae_net.pt already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(os.path.join(ROOT, "notebooks", "vae_config.yaml"), "r") as fh:
        vae_config = yaml.safe_load(fh)
    with open(os.path.join(ROOT, "notebooks", "datagen.yaml"), "r") as fh:
        data_config = yaml.safe_load(fh)

    dataset_num = data_config["DATASET"]["dataset_num"]
    dataset_dir = os.path.join(ROOT, "dataset")
    mstr_shape_params = scipy.io.loadmat(
        os.path.join(dataset_dir, f"mstr_shape_parameters_{dataset_num}.mat")
    )["mstr_shape_parameters"]
    homog = scipy.io.loadmat(
        os.path.join(dataset_dir, f"homogen_data_{dataset_num}.mat")
    )
    mstr_area = scipy.io.loadmat(
        os.path.join(dataset_dir, f"mstr_area_{dataset_num}.mat")
    )["mstr_area"]
    mstr_perim = scipy.io.loadmat(
        os.path.join(dataset_dir, f"mstr_perim_{dataset_num}.mat")
    )["mstr_perim"]
    c00 = np.asarray(homog["c00"]).reshape(-1, 1)
    c11 = np.asarray(homog["c11"]).reshape(-1, 1)
    print(
        f"shape_params: {mstr_shape_params.shape}, c00/c11: {c00.shape}, "
        f"area: {mstr_area.shape}, perim: {mstr_perim.shape}"
    )

    mstr_data = torch.tensor(
        np.hstack(
            (
                mstr_shape_params,
                c00,
                c11,
                np.asarray(mstr_perim).reshape(-1, 1),
                np.asarray(mstr_area).reshape(-1, 1),
            )
        )
    ).double()

    normalization_types = (
        [data_preprocess.NomalizationType.LINEAR] * 8
        + [data_preprocess.NomalizationType.LOG] * 2
        + [data_preprocess.NomalizationType.LINEAR] * 2
    )
    normalized_train_data, max_feature, min_feature = data_preprocess.stack_train_data(
        mstr_data, normalization_types
    )
    num_samples, num_features = normalized_train_data.shape
    print(f"normalized data: {num_samples} samples, {num_features} features")

    vae_yaml = vae_config["NETWORK"]
    vae_params = network.VAE_Params(
        input_dim=num_features,
        encoder_hidden_dim=vae_yaml["encoder_hidden_dim"],
        latent_dim=vae_yaml["latent_dim"],
        decoder_hidden_dim=vae_yaml["decoder_hidden_dim"],
    )
    vae_net = network.VariationalAutoencoder(vae_params=vae_params)

    vae_dir = os.path.join(ROOT, "vae")
    os.makedirs(vae_dir, exist_ok=True)
    weight_path = os.path.join(vae_dir, "vae_net.pt")

    if (not os.path.isfile(weight_path)) or args.retrain:
        opt_yaml = vae_config["OPTIMIZATION"]
        num_epochs = args.epochs if args.epochs is not None else opt_yaml["num_epochs"]
        print(f"Training VAE for {num_epochs} epochs (lr={opt_yaml['lr']}, kl={opt_yaml['kl_factor']}) ...")
        convg_history = train_vae.train_autoencoder(
            vae=vae_net,
            train_data=normalized_train_data,
            num_epochs=num_epochs,
            kl_factor=opt_yaml["kl_factor"],
            lr=opt_yaml["lr"],
            save_file=weight_path,
            print_every=max(1, num_epochs // 10),
        )
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(convg_history["recon_loss"], label="recon")
        ax.plot(convg_history["kl_loss"], label="kl")
        ax.plot(convg_history["loss"], label="total")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(ROOT, "output", "vae_loss_history.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"Saved {fig_path}")
    else:
        print(f"Found existing {weight_path}, skipping training (pass --retrain to override)")

    vae_net.encoder.is_training = False
    vae_net.load_state_dict(torch.load(weight_path))
    vae_net.eval()

    norm_path = os.path.join(vae_dir, "nomalization.pt")
    torch.save(
        {"max_feature": max_feature, "min_feature": min_feature},
        norm_path,
    )
    print(f"Saved normalisation info -> {norm_path}")

    with torch.no_grad():
        vae_latent_encoding = vae_net.encoder(normalized_train_data).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(vae_latent_encoding[:, 0], vae_latent_encoding[:, 1], s=20)
    ax.set_title("Latent space")
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.tight_layout()
    fig_path = os.path.join(ROOT, "output", "vae_latent_space.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
