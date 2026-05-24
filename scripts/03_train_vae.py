#!/usr/bin/env python3
"""Step 3: Train VAE on homogenized super-shape database."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset"))
sys.path.insert(0, os.path.join(ROOT, "vae"))

import data_preprocess
import network
import train_vae

os.chdir(os.path.join(ROOT, "notebooks"))

with open("vae_config.yaml") as f:
    vae_config = yaml.safe_load(f)
with open("datagen.yaml") as f:
    data_config = yaml.safe_load(f)

dataset_num = data_config["DATASET"]["dataset_num"]
dataset_dir = os.path.join(ROOT, "dataset")

mstr_shape_params = scipy.io.loadmat(
    os.path.join(dataset_dir, f"mstr_shape_parameters_{dataset_num}.mat")
)["mstr_shape_parameters"]
mstr_homog_data = scipy.io.loadmat(
    os.path.join(dataset_dir, f"homogen_data_{dataset_num}.mat")
)
mstr_area = scipy.io.loadmat(
    os.path.join(dataset_dir, f"mstr_area_{dataset_num}.mat")
)["mstr_area"].reshape(-1, 1)
mstr_perim = scipy.io.loadmat(
    os.path.join(dataset_dir, f"mstr_perim_{dataset_num}.mat")
)["mstr_perim"].reshape(-1, 1)
c00, c11 = mstr_homog_data["c00"], mstr_homog_data["c11"]

mstr_data = torch.tensor(
    np.hstack(
        (
            mstr_shape_params,
            c00,
            c11,
            mstr_perim.reshape((-1, 1)),
            mstr_area,
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

out_dir = os.path.join(ROOT, "results", "step3_vae")
os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots()
ax.hist(c00.flatten(), bins=20, edgecolor="black")
ax.set_xlabel("C_00")
ax.set_ylabel("Frequency")
ax.set_title("Homogenized permeability C_00 distribution")
fig.savefig(os.path.join(out_dir, "c00_histogram.png"), dpi=150)
plt.close()

num_samples, num_features = normalized_train_data.shape
vae_yaml = vae_config["NETWORK"]
vae_params = network.VAE_Params(
    input_dim=num_features,
    encoder_hidden_dim=vae_yaml["encoder_hidden_dim"],
    latent_dim=vae_yaml["latent_dim"],
    decoder_hidden_dim=vae_yaml["decoder_hidden_dim"],
)
vae_net = network.VariationalAutoencoder(vae_params=vae_params)

vae_dir = os.path.join(ROOT, "vae")
file_path = os.path.join(vae_dir, "vae_net.pt")
opt_yaml = vae_config["OPTIMIZATION"]

convg_history = train_vae.train_autoencoder(
    vae=vae_net,
    train_data=normalized_train_data,
    num_epochs=opt_yaml["num_epochs"],
    kl_factor=opt_yaml["kl_factor"],
    lr=opt_yaml["lr"],
    save_file=file_path,
    print_every=max(1, opt_yaml["num_epochs"] // 20),
)

torch.save(
    {"max_feature": max_feature, "min_feature": min_feature},
    os.path.join(vae_dir, "nomalization.pt"),
)

vae_net.encoder.is_training = False
vae_net.load_state_dict(torch.load(file_path))
vae_net.eval()
vae_latent = vae_net.encoder(normalized_train_data).detach().numpy()

fig, ax = plt.subplots()
ax.scatter(vae_latent[:, 0], vae_latent[:, 1], s=12, alpha=0.7)
ax.set_xlabel("z1")
ax.set_ylabel("z2")
ax.set_title("VAE latent space (training set)")
fig.savefig(os.path.join(out_dir, "latent_scatter.png"), dpi=150)
plt.close()

fig, ax = plt.subplots()
ax.semilogy(convg_history["recon_loss"], label="reconstruction")
ax.semilogy(convg_history["kl_loss"], label="KL")
ax.set_xlabel("epoch")
ax.legend()
fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
plt.close()

print(f"VAE trained for {opt_yaml['num_epochs']} epochs; weights at {file_path}")
