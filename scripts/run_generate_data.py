"""Executes the steps of notebooks/generate_data.ipynb as a plain Python script.

Generates super-shape micro-structure samples and saves them as `.mat`
files in ``/workspace/dataset`` so they can be fed into the MATLAB
homogenisation script (``mstr_data_gen_main.m``).
"""

import os
import sys
import yaml
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset"))

import supershape as ss  # noqa: E402
import mesher  # noqa: E402  (re-exported via supershape, but ensures importable)


def main() -> None:
    config_path = os.path.join(ROOT, "notebooks", "datagen.yaml")
    with open(config_path, "r") as fh:
        config_data = yaml.safe_load(fh)

    shape_yaml = config_data["SUPERSHAPE"]
    shape_extents = ss.SuperShapeExtents(
        a=ss.Extents(shape_yaml["min_a"], shape_yaml["max_a"]),
        b=ss.Extents(shape_yaml["min_b"], shape_yaml["max_b"]),
        m=ss.Extents(shape_yaml["min_m"], shape_yaml["max_m"]),
        n1=ss.Extents(shape_yaml["min_n1"], shape_yaml["max_n1"]),
        n2=ss.Extents(shape_yaml["min_n2"], shape_yaml["max_n2"]),
        n3=ss.Extents(shape_yaml["min_n3"], shape_yaml["max_n3"]),
        center_x=ss.Extents(shape_yaml["min_cx"], shape_yaml["max_cy"]),
        center_y=ss.Extents(shape_yaml["min_cy"], shape_yaml["max_cy"]),
    )

    num_samples = config_data["DATASET"]["num_samples"]
    print(f"Generating {num_samples} random super-shapes ...")
    random_shape_param = ss.generate_random_super_shapes(num_samples, shape_extents)

    mesh_yaml = config_data["MESH"]
    print(
        f"Projecting on {mesh_yaml['nelx']}x{mesh_yaml['nely']} grid and pruning "
        "shapes that intersect the unit cell boundary ..."
    )
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

    print(
        f"After pruning: {shape_density.shape[0]} samples, "
        f"density tensor shape = {shape_density.shape}"
    )

    out_dir = os.path.join(ROOT, "dataset")
    out_fig_dir = os.path.join(ROOT, "output")
    os.makedirs(out_fig_dir, exist_ok=True)

    # Sample plot of 16 polygons - saved to file because we're running headless
    rng = np.random.default_rng(42)
    n_imgs_x, n_imgs_y = 4, 4
    rand_img_idxs = rng.integers(0, shape_density.shape[0], size=n_imgs_x * n_imgs_y)
    rand_mstr_imgs = shape_density[rand_img_idxs, :]
    rand_shape_perim = normalized_shape_perim[rand_img_idxs]

    fig, axs = plt.subplots(n_imgs_x, n_imgs_y)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        im = ax.imshow(rand_mstr_imgs[i, :].T, cmap="coolwarm")
        ax.set_title(f"{rand_shape_perim[i]:.2F}")
        ax.axis("off")
    cbar = plt.colorbar(im, ax=axs)
    cbar.set_label("density (0=solid, 1=fluid)")
    fig.suptitle("Random super-shape microstructures")
    sample_path = os.path.join(out_fig_dir, "sample_microstructures.png")
    fig.savefig(sample_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample figure -> {sample_path}")

    shape_params = pruned_shape_parameters.to_stacked_array()
    dataset_num = config_data["DATASET"]["dataset_num"]

    files = {
        f"mstr_shape_parameters_{dataset_num}.mat": {
            "mstr_shape_parameters": shape_params
        },
        f"mstr_images_{dataset_num}.mat": {"mstr_images": shape_density},
        f"mstr_area_{dataset_num}.mat": {"mstr_area": normalized_shape_area},
        f"mstr_perim_{dataset_num}.mat": {"mstr_perim": normalized_shape_perim},
    }
    for fname, payload in files.items():
        path = os.path.join(out_dir, fname)
        scipy.io.savemat(path, payload, do_compression=True)
        print(f"Saved {path}  (keys: {list(payload.keys())})")

    print("\nDataset summary:")
    print(f"  shape_params:           {shape_params.shape}")
    print(f"  shape_density:          {shape_density.shape}")
    print(f"  normalized_shape_area:  {normalized_shape_area.shape}")
    print(f"  normalized_shape_perim: {normalized_shape_perim.shape}")


if __name__ == "__main__":
    main()
