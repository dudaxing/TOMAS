#!/usr/bin/env python3
"""Step 1: Generate super-shape microstructure dataset (Python part of Algorithm 1)."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "dataset"))

import mesher  # noqa: F401
import supershape as ss

os.chdir(os.path.join(ROOT, "notebooks"))

with open("datagen.yaml", "r") as f:
    config_data = yaml.safe_load(f)

shape_yaml = config_data["SUPERSHAPE"]
shape_extents = ss.SuperShapeExtents(
    a=ss.Extents(shape_yaml["min_a"], shape_yaml["max_a"]),
    b=ss.Extents(shape_yaml["min_b"], shape_yaml["max_b"]),
    m=ss.Extents(shape_yaml["min_m"], shape_yaml["max_m"]),
    n1=ss.Extents(shape_yaml["min_n1"], shape_yaml["max_n1"]),
    n2=ss.Extents(shape_yaml["min_n2"], shape_yaml["max_n2"]),
    n3=ss.Extents(shape_yaml["min_n3"], shape_yaml["max_n3"]),
    center_x=ss.Extents(shape_yaml["min_cx"], shape_yaml["max_cx"]),
    center_y=ss.Extents(shape_yaml["min_cy"], shape_yaml["max_cy"]),
)

random_shape_param = ss.generate_random_super_shapes(
    config_data["DATASET"]["num_samples"], shape_extents
)

mesh_yaml = config_data["MESH"]
polygons, pruned_shape_parameters = ss.super_shape_to_shapely_polygon(
    random_shape_param
)
shape_density = ss.project_shapely_polygons_to_density(
    polygons, mesh_yaml["nelx"], mesh_yaml["nely"], True
)

shape_perim = ss.compute_shapely_polygon_perimeter(polygons)
shape_area = ss.compute_shapely_polygon_area(polygons)
normalized_shape_area = shape_area / (
    random_shape_param.domain_length_x * random_shape_param.domain_length_y
)
normalized_shape_perim = shape_perim / (
    random_shape_param.domain_length_x + random_shape_param.domain_length_y
)

out_dir = os.path.join(ROOT, "results", "step1_datagen")
os.makedirs(out_dir, exist_ok=True)

n_imgs_x, n_imgs_y = 4, 4
rng = np.random.default_rng(config_data["SUPERSHAPE"]["shape_seed"])
rand_img_idxs = rng.integers(0, shape_density.shape[0], n_imgs_x * n_imgs_y)
_, axs = plt.subplots(n_imgs_x, n_imgs_y, figsize=(8, 8))
for i, ax in enumerate(axs.flatten()):
    im = ax.imshow(
        shape_density[rand_img_idxs[i], :].reshape(mesh_yaml["nelx"], mesh_yaml["nely"]).T,
        cmap="coolwarm",
    )
    ax.set_title(f"P={normalized_shape_perim[rand_img_idxs[i]]:.2f}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "sample_microstructures.png"), dpi=150)
plt.close()

shape_params = pruned_shape_parameters.to_stacked_array()
dataset_num = config_data["DATASET"]["dataset_num"]
dataset_dir = os.path.join(ROOT, "dataset")

scipy.io.savemat(
    os.path.join(dataset_dir, f"mstr_shape_parameters_{dataset_num}.mat"),
    {"mstr_shape_parameters": shape_params},
)
scipy.io.savemat(
    os.path.join(dataset_dir, f"mstr_images_{dataset_num}.mat"),
    {"mstr_images": shape_density},
)
scipy.io.savemat(
    os.path.join(dataset_dir, f"mstr_area_{dataset_num}.mat"),
    {"mstr_area": normalized_shape_area},
)
scipy.io.savemat(
    os.path.join(dataset_dir, f"mstr_perim_{dataset_num}.mat"),
    {"mstr_perim": normalized_shape_perim},
)

# Input expected by MATLAB homogenization script
scipy.io.savemat(
    os.path.join(dataset_dir, "recons_shapes.mat"),
    {"mstr_images": shape_density},
)

print(f"Generated {shape_density.shape[0]} microstructures on {mesh_yaml['nelx']}x{mesh_yaml['nely']} grid.")
print(f"Saved dataset #{dataset_num} under {dataset_dir}")
