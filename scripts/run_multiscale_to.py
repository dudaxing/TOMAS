"""Runs the multiscale topology optimization pipeline.

This script replicates the steps of ``notebooks/multiscale_TO_main.ipynb``
without requiring an interactive notebook.  Use ``--config`` to switch
between the bundled examples (diffuser / bent_pipe / biffurcated_pipe)
and ``--epochs`` to override the iteration count for smoke tests.
"""

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vae"))
sys.path.insert(0, os.path.join(ROOT, "fluid_TO"))
sys.path.insert(0, os.path.join(ROOT, "dataset"))

import dataset.supershape as supershape  # noqa: E402
import material_constants  # noqa: E402
import fluid_mesher  # noqa: E402
import fluid_bcs  # noqa: E402
import fluid_material as fluid_material_mod  # noqa: E402  (avoid name clash)
import fluid_fe  # noqa: E402
import projection  # noqa: E402
import neural_network  # noqa: E402
import loss as loss_mod  # noqa: E402
import utils  # noqa: E402
import plot as plot_mod  # noqa: E402
import opt_constraints  # noqa: E402

import vae.network as vae_network  # noqa: E402
import vae.data_preprocess as vae_data_prep  # noqa: E402


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


@dataclass
class OptimizationInitials:
    J0: float
    constraints: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_diffuser.yaml",
        help="config file name inside notebooks/ (default: config_diffuser.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override OPTIMIZATION.num_epochs from the config file",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="tag appended to output filenames (defaults to config stem)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = os.path.join(ROOT, "notebooks", args.config)
    with open(config_path, "r") as fh:
        config_data = yaml.safe_load(fh)

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
        fluid_bcs.FluidSampleProblems[
            config_data["BOUNDARY_CONDITIONS"]["example"]
        ],
    )

    fluid_mat_constants = material_constants.MaterialConstants(
        kinematic_viscosity=config_data["MATERIAL_CONSTANTS"][
            "kinematic_viscosity"
        ]
    )
    fluid_material = fluid_material_mod.FluidMaterial(
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
    symmetry_activation_x_axis = projection.SymmetryActivation.SYM_X_AXIS_ON
    symmetry_activation_y_axis = projection.SymmetryActivation.SYM_Y_AXIS_OFF

    constraint_type = opt_constraints.ConstraintType[
        config_data["OPTIMIZATION"]["constraint_type"]
    ]
    nn_settings = neural_network.NeuralNetworkParameters(
        input_dim=2 * config_data["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
        output_dim=config_data["NEURAL_NETWORK_PARAMS"]["output_dim"],
        num_layers=config_data["NEURAL_NETWORK_PARAMS"]["num_layers"],
        num_neurons_per_layer=config_data["NEURAL_NETWORK_PARAMS"][
            "num_neurons_per_layer"
        ],
    )
    neural_net = neural_network.TopOptNet(nn_params=nn_settings)

    with open(os.path.join(ROOT, "notebooks", "vae_config.yaml"), "r") as fh:
        vae_config_data = yaml.safe_load(fh)
    vae_yaml = vae_config_data["NETWORK"]
    vae_params = vae_network.VAE_Params(
        input_dim=12,
        encoder_hidden_dim=vae_yaml["encoder_hidden_dim"],
        latent_dim=vae_yaml["latent_dim"],
        decoder_hidden_dim=vae_yaml["decoder_hidden_dim"],
    )
    vae_net = vae_network.VariationalAutoencoder(vae_params=vae_params)
    vae_net.encoder.is_training = False
    vae_net.load_state_dict(torch.load(os.path.join(ROOT, "vae", "vae_net.pt")))
    vae_net.eval()
    nomalization = torch.load(os.path.join(ROOT, "vae", "nomalization.pt"))
    max_feature = nomalization["max_feature"]
    min_feature = nomalization["min_feature"]
    print("Loaded VAE + normalisation tensors.")
    normalization_types = (
        [vae_data_prep.NomalizationType.LINEAR] * 8
        + [vae_data_prep.NomalizationType.LOG] * 2
        + [vae_data_prep.NomalizationType.LINEAR] * 2
    )

    sym_params = projection.SymParams(
        sym_x_axis_mid_pt=0.5 * config_data["BOUNDING_BOX"]["y_max"],
        sym_y_axis_mid_pt=0.5 * config_data["BOUNDING_BOX"]["x_max"],
    )
    num_constraints = config_data["OPTIMIZATION"]["num_constraints"]
    desired_vol_frac = config_data["OPTIMIZATION"]["desired_vol_frac"]
    desired_perimeter = config_data["OPTIMIZATION"]["desired_perimeter"]
    loss_type = loss_mod.LossTypes[config_data["LOSS"]["method"]]
    loss_params = loss_mod.PenaltyLossParameters(
        alpha0=config_data["LOSS"]["alpha0"],
        del_alpha=config_data["LOSS"]["del_alpha"],
    )
    num_epochs = (
        args.epochs
        if args.epochs is not None
        else config_data["OPTIMIZATION"]["num_epochs"]
    )
    opt_initials = OptimizationInitials(
        J0=config_data["OPTIMIZATION"]["init_objective"],
        constraints=np.zeros((num_constraints,)),
    )
    opt_params = OptimizationSettings(
        plot_interval=config_data["OPTIMIZATION"]["plot_interval"],
        lr=config_data["OPTIMIZATION"]["learning_rate"],
        num_epochs=num_epochs,
        method=Optimizer[config_data["OPTIMIZATION"]["method"]],
        grad_clip_activation=GradClipActivation[
            config_data["OPTIMIZATION"]["grad_clip_activation"]
        ],
        grad_clip_norm=config_data["OPTIMIZATION"]["grad_clip_norm"],
    )

    perim_scale = 2.0
    perimeter_history = []
    fluid_loss_history = []
    tag = args.tag or os.path.splitext(args.config)[0]
    out_dir = os.path.join(ROOT, "output", tag)
    os.makedirs(out_dir, exist_ok=True)

    fluid_xy = torch.tensor(fluid_solver.mesh.elem_centers, requires_grad=True).double()
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

    if opt_params.method == Optimizer.ADAM:
        print("Adam optimizer")
        optimizer = optim.Adam(neural_net.parameters(), amsgrad=True, lr=opt_params.lr)
    else:
        print("LBFGS optimizer")
        optimizer = optim.LBFGS(neural_net.parameters(), line_search_fn="strong_wolfe")

    def loss_wrapper(epoch: int, eps: float = 1e-6):
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
        else:  # PERIMETER
            constraint_field = (
                fluid_solver.mesh.elem_dx
                * perim_scale
                * vae_output_renormalized[
                    :, vae_data_prep.VAE_Fields.shape_perim.value
                ]
            )

        C_00 = vae_output_renormalized[:, vae_data_prep.VAE_Fields.homog_c00.value] + eps
        C_11 = vae_output_renormalized[:, vae_data_prep.VAE_Fields.homog_c11.value] + eps
        mstr_data = vae_output_renormalized[
            :, : vae_data_prep.VAE_Fields.homog_c00.value
        ].detach().numpy()
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

        fluid_loss, _vp = fluid_solver.fluid_objective_function(
            fluid_material, C_00, C_11, theta
        )
        field_cons_value = opt_constraints.constraint_function(
            constraint_type, constraint_field, desired_vol_frac, desired_perimeter
        )
        if epoch == 0 or epoch == 20:
            opt_initials.J0 = fluid_loss.item()

        opt_params.constraints = [field_cons_value]
        net_loss = loss_mod.combined_loss(
            objective=fluid_loss / opt_initials.J0,
            constraints=opt_params.constraints,
            loss_type=loss_type,
            loss_params=loss_params,
            epoch=epoch,
        )
        return net_loss, fluid_loss, opt_params.constraints, constraint_field, mstr_params, C_00, C_11, theta

    def make_closure(epoch_holder):
        def closure():
            optimizer.zero_grad()
            net_loss, *_ = loss_wrapper(epoch_holder[0])
            net_loss.backward(retain_graph=True)
            if opt_params.grad_clip_activation == GradClipActivation.GRAD_CLIP_ON:
                torch.nn.utils.clip_grad_norm_(
                    neural_net.parameters(), opt_params.grad_clip_norm
                )
            return net_loss

        return closure

    epoch_holder = [0]
    closure = make_closure(epoch_holder)

    start_time = time.time()
    for epoch in range(opt_params.num_epochs):
        epoch_holder[0] = epoch
        if opt_params.method == Optimizer.ADAM:
            closure()
            optimizer.step()
        else:
            optimizer.step(closure)

        loss_mod.update_loss_parameters(
            epoch, loss_type, loss_params, opt_params.constraints
        )
        (net_loss, fluid_loss, constraint, constraint_field, mstr_params, C_00, C_11, theta) = (
            loss_wrapper(epoch)
        )
        net_loss.backward(retain_graph=True)
        perimeter_history.append(torch.sum(constraint_field).item())
        fluid_loss_history.append(fluid_loss.item())
        print(
            f"Iter {epoch:3d} J: {fluid_loss.item():.2E} "
            f"perimeter: {torch.sum(constraint_field).item():.2F}"
        )

        if epoch % opt_params.plot_interval == 0 or epoch == opt_params.num_epochs - 1:
            theta_np = theta.detach().numpy()
            plt.close("all")
            try:
                plot_mod.plot_microstructures_in_macro_mesh(
                    mstr_params,
                    theta_np,
                    fluid_solver.mesh.nelx,
                    fluid_solver.mesh.nely,
                    epoch,
                )
                fig = plt.gcf()
                fig_path = os.path.join(out_dir, f"design_epoch_{epoch:04d}.png")
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {fig_path}")
            except Exception as exc:
                print(f"WARNING: plotting failed at epoch {epoch}: {exc}")

    elapsed = time.time() - start_time
    print(f"\nOptimization done in {elapsed:.1f} s")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fluid_loss_history)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("fluid loss")
    axes[1].plot(perimeter_history)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("perimeter")
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "convergence.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
