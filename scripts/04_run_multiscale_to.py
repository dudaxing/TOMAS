#!/usr/bin/env python3
"""Step 4: Multiscale fluid topology optimization (paper Section 4 examples)."""
import argparse
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vae"))
sys.path.insert(0, os.path.join(ROOT, "fluid_TO"))
sys.path.insert(0, os.path.join(ROOT, "dataset"))

import dataset.supershape as supershape
import fluid_bcs
import fluid_fe
import fluid_material
import fluid_mesher
import loss
import material_constants
import neural_network
import opt_constraints
import plot
import projection
import utils
import vae.data_preprocess as vae_data_prep
import vae.network as vae_network


class Optimizer(Enum):
    ADAM = auto()
    LBFGS = auto()


class GradClipActivation(Enum):
    GRAD_CLIP_ON = auto()
    GRAD_CLIP_OFF = auto()


@dataclass
class OptimizationSettings:
    plot_interval: int
    lr: float
    num_epochs: int
    method: Optimizer
    grad_clip_activation: GradClipActivation
    grad_clip_norm: float
    constraints: List = None


@dataclass
class OptimizationInitials:
    J0: float
    constraints: np.ndarray


def run_multiscale_to(config_name: str) -> None:
    os.chdir(os.path.join(ROOT, "notebooks"))
    with open(config_name) as f:
        config_data = yaml.safe_load(f)

    example = config_data["BOUNDARY_CONDITIONS"]["example"].lower()
    out_dir = os.path.join(ROOT, "results", f"step4_to_{example}")
    os.makedirs(out_dir, exist_ok=True)

    plot_epoch = [0]

    _plot_microstructures_orig = plot.plot_microstructures_in_macro_mesh

    def save_microstructure_plot(mstr_params, mstr_rotation, nelx, nely, epoch):
        _plot_microstructures_orig(mstr_params, mstr_rotation, nelx, nely, epoch)
        fig = plt.gcf()
        fig.savefig(
            os.path.join(out_dir, f"microstructures_epoch_{epoch:04d}.png"),
            dpi=120,
            bbox_inches="tight",
        )
        plt.close(fig)

    bounding_box = fluid_mesher.BoundingBox(
        x_min=config_data["BOUNDING_BOX"]["x_min"],
        x_max=config_data["BOUNDING_BOX"]["x_max"],
        y_min=config_data["BOUNDING_BOX"]["y_min"],
        y_max=config_data["BOUNDING_BOX"]["y_max"],
    )
    fluid_mesh = fluid_mesher.fluid_mesher(
        nelx=config_data["MESH"]["nelx"],
        nely=config_data["MESH"]["nely"],
        bounding_box=bounding_box,
    )

    char_velocity = config_data["BOUNDARY_CONDITIONS"]["char_velocity"]
    fluid_bc = fluid_bcs.get_dirichlet_bc_and_fixed_dofs(
        fluid_mesh,
        char_velocity,
        fluid_bcs.FluidSampleProblems[config_data["BOUNDARY_CONDITIONS"]["example"]],
    )

    fluid_mat_constants = material_constants.MaterialConstants(
        kinematic_viscosity=config_data["MATERIAL_CONSTANTS"]["kinematic_viscosity"]
    )
    fluid_mat = fluid_material.FluidMaterial(
        fluid_mesh.elem_dx, fluid_mesh.elem_dy, fluid_mat_constants
    )

    fluid_solver = fluid_fe.FluidSolver(
        fluid_mesh,
        fluid_bc,
        fixture_const=config_data["BOUNDARY_CONDITIONS"]["fixture_const"],
    )

    fourier_map = projection.FourierMap(
        fluid_mesh,
        fourier_map_activation=projection.FourierActivation[
            config_data["FOURIER_MAP_PARAMS"]["fourier_map_activation"]
        ],
        num_fourier_terms=config_data["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
        max_radius=config_data["FOURIER_MAP_PARAMS"]["max_radius"],
        min_radius=config_data["FOURIER_MAP_PARAMS"]["min_radius"],
    )

    symmetry_activation_x_axis = projection.SymmetryActivation[
        config_data["FOURIER_MAP_PARAMS"]["symmetry_activation_x_axis"]
    ]
    symmetry_activation_y_axis = projection.SymmetryActivation[
        config_data["FOURIER_MAP_PARAMS"]["symmetry_activation_y_axis"]
    ]

    constraint_type = opt_constraints.ConstraintType[
        config_data["OPTIMIZATION"]["constraint_type"]
    ]
    nn_settings = neural_network.NeuralNetworkParameters(
        input_dim=2 * config_data["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
        output_dim=config_data["NEURAL_NETWORK_PARAMS"]["output_dim"],
        num_layers=config_data["NEURAL_NETWORK_PARAMS"]["num_layers"],
        num_neurons_per_layer=config_data["NEURAL_NETWORK_PARAMS"]["num_neurons_per_layer"],
    )
    neural_net = neural_network.TopOptNet(nn_params=nn_settings)

    vae_yaml_path = os.path.join(ROOT, "notebooks", "vae_config.yaml")
    with open(vae_yaml_path) as f:
        vae_config_data = yaml.safe_load(f)
    vae_yaml = vae_config_data["NETWORK"]
    vae_params = vae_network.VAE_Params(
        input_dim=12,
        encoder_hidden_dim=vae_yaml["encoder_hidden_dim"],
        latent_dim=vae_yaml["latent_dim"],
        decoder_hidden_dim=vae_yaml["decoder_hidden_dim"],
    )
    vae_net = vae_network.VariationalAutoencoder(vae_params=vae_params)
    file_path = os.path.join(ROOT, "vae", "vae_net.pt")
    vae_net.encoder.is_training = False
    vae_net.load_state_dict(torch.load(file_path, map_location="cpu"))
    vae_net.eval()
    nomalization = torch.load(os.path.join(ROOT, "vae", "nomalization.pt"))
    max_feature = nomalization["max_feature"]
    min_feature = nomalization["min_feature"]
    normalization_types = (
        [vae_data_prep.NomalizationType.LINEAR] * 8
        + [vae_data_prep.NomalizationType.LOG] * 2
        + [vae_data_prep.NomalizationType.LINEAR] * 2
    )

    sym_params = projection.SymParams(
        sym_x_axis_mid_pt=0.5 * config_data["BOUNDING_BOX"]["y_max"],
        sym_y_axis_mid_pt=0.5 * config_data["BOUNDING_BOX"]["x_max"],
    )

    def calculate_projected_coordinates(res: float) -> Tuple[torch.Tensor, dict]:
        if res == 1:
            fluid_xy = torch.tensor(fluid_solver.mesh.elem_centers, requires_grad=True).double()
        else:
            elem_centers = utils.generate_points_in_domain(
                fluid_mesh.nelx,
                fluid_mesh.nely,
                fluid_mesh.elem_dx,
                fluid_mesh.elem_dy,
                fluid_mesh.num_dim,
                res,
            )
            fluid_xy = torch.tensor(elem_centers, requires_grad=True).double()
        fluid_xy_r, fluid_signs_reflection = projection.apply_reflection(
            fluid_xy,
            symmetry_activation_y_axis,
            symmetry_activation_x_axis,
            sym_params,
        )
        if fourier_map.fourier_map_activation == projection.FourierActivation.FOURIER_ON:
            fluid_xy_f = fourier_map.apply_fourier_map(fluid_xy_r)
        else:
            fluid_xy_f = fluid_xy_r
        return fluid_xy_f, fluid_signs_reflection

    desired_vol_frac = config_data["OPTIMIZATION"]["desired_vol_frac"]
    desired_perimeter = config_data["OPTIMIZATION"]["desired_perimeter"]
    loss_type = loss.LossTypes[config_data["LOSS"]["method"]]
    loss_params = loss.PenaltyLossParameters(
        alpha0=config_data["LOSS"]["alpha0"],
        del_alpha=config_data["LOSS"]["del_alpha"],
    )
    opt_initials = OptimizationInitials(
        J0=config_data["OPTIMIZATION"]["init_objective"],
        constraints=np.zeros((config_data["OPTIMIZATION"]["num_constraints"],)),
    )
    opt_params = OptimizationSettings(
        plot_interval=config_data["OPTIMIZATION"]["plot_interval"],
        lr=config_data["OPTIMIZATION"]["learning_rate"],
        num_epochs=config_data["OPTIMIZATION"]["num_epochs"],
        method=Optimizer[config_data["OPTIMIZATION"]["method"]],
        grad_clip_activation=GradClipActivation[
            config_data["OPTIMIZATION"]["grad_clip_activation"]
        ],
        grad_clip_norm=config_data["OPTIMIZATION"]["grad_clip_norm"],
    )
    perim_scale = 2.0

    fluid_xy_f, fluid_signs_reflection = calculate_projected_coordinates(res=1.0)
    if opt_params.method == Optimizer.ADAM:
        optimizer = optim.Adam(neural_net.parameters(), amsgrad=True, lr=opt_params.lr)
    else:
        optimizer = optim.LBFGS(
            neural_net.parameters(), line_search_fn="strong_wolfe"
        )

    convergence_history = {"epoch": [], "perimeter": [], "fluid_loss": []}
    epoch = 0

    def loss_wrapper(eps=1e-6):
        nonlocal epoch
        optimizer.zero_grad()
        latent_space, theta = neural_net(fluid_xy_f)
        theta = torch.einsum("i,i->i", theta, fluid_signs_reflection["X"])
        theta = torch.einsum("i,i->i", theta, fluid_signs_reflection["Y"])
        decoded_latent_pt = vae_net.decoder(latent_space)
        vae_output_renormalized = vae_data_prep.stack_vae_output(
            decoded_latent_pt, max_feature, min_feature, normalization_types
        )
        if constraint_type == opt_constraints.ConstraintType.VOLUME:
            constraint_field = vae_output_renormalized[
                :, vae_data_prep.VAE_Fields.shape_area.value
            ]
        else:
            constraint_field = (
                fluid_solver.mesh.elem_dx
                * perim_scale
                * vae_output_renormalized[:, vae_data_prep.VAE_Fields.shape_perim.value]
            )
        C_00 = vae_output_renormalized[:, vae_data_prep.VAE_Fields.homog_c00.value] + eps
        C_11 = vae_output_renormalized[:, vae_data_prep.VAE_Fields.homog_c11.value] + eps
        mstr_data = (
            vae_output_renormalized[:, : vae_data_prep.VAE_Fields.homog_c00.value]
            .clone()
            .detach()
            .numpy()
        )
        mstr_params = supershape.SuperShapes(
            mstr_data[:, 0],
            mstr_data[:, 1],
            mstr_data[:, 2],
            mstr_data[:, 3],
            mstr_data[:, 4],
            mstr_data[:, 5],
            mstr_data[:, 6],
            mstr_data[:, 7],
        )
        fluid_loss, _ = fluid_solver.fluid_objective_function(
            fluid_mat, C_00, C_11, theta
        )
        field_cons_value = opt_constraints.constraint_function(
            constraint_type, constraint_field, desired_vol_frac, desired_perimeter
        )
        if epoch == 0 or epoch == 20:
            opt_initials.J0 = fluid_loss.item()
        opt_params.constraints = [field_cons_value]
        net_loss = loss.combined_loss(
            objective=fluid_loss / opt_initials.J0,
            constraints=opt_params.constraints,
            loss_type=loss_type,
            loss_params=loss_params,
            epoch=epoch,
        )
        return (
            net_loss,
            fluid_loss,
            opt_params.constraints,
            constraint_field,
            mstr_params,
            C_00,
            C_11,
            theta,
        )

    def closure():
        net_loss, *_ = loss_wrapper()
        net_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(neural_net.parameters(), opt_params.grad_clip_norm)
        return net_loss

    t0 = time.time()
    for epoch in range(opt_params.num_epochs):
        if opt_params.method == Optimizer.ADAM:
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)

        loss.update_loss_parameters(epoch, loss_type, loss_params, opt_params.constraints)
        (
            net_loss,
            fluid_loss,
            constraint,
            constraint_field,
            mstr_params,
            C_00,
            C_11,
            theta,
        ) = loss_wrapper()
        net_loss.backward(retain_graph=True)
        convergence_history["epoch"].append(epoch)
        convergence_history["perimeter"].append(torch.sum(constraint_field).item())
        convergence_history["fluid_loss"].append(fluid_loss.item())
        print(
            f"Iter {epoch:3d} J: {fluid_loss.item():.2E}; perimeter: {torch.sum(constraint_field).item():.2f}"
        )
        if epoch % opt_params.plot_interval == 0:
            theta_np = theta.clone().detach().numpy()
            save_microstructure_plot(
                mstr_params,
                theta_np,
                fluid_solver.mesh.nelx,
                fluid_solver.mesh.nely,
                epoch,
            )

    fig, ax1 = plt.subplots()
    ax1.semilogy(convergence_history["fluid_loss"], "b-", label="dissipated power J")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("J")
    ax2 = ax1.twinx()
    ax2.plot(convergence_history["perimeter"], "r--", label="perimeter")
    ax2.set_ylabel("perimeter")
    fig.legend(loc="upper right")
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=150)
    plt.close()

    np.savez(
        os.path.join(out_dir, "convergence_history.npz"),
        **convergence_history,
    )
    print(f"Finished {example} in {time.time() - t0:.1f}s; outputs in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="config_diffuser.yaml",
        help="YAML config in notebooks/",
    )
    args = parser.parse_args()
    run_multiscale_to(args.config)
