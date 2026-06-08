"""Headless multiscale fluid TO runner (Algorithm 3) — from multiscale_TO_main.ipynb.

Runs one optimization for a given config and saves the final micro-structure
layout, the convergence curve, the velocity-magnitude field, and a metrics JSON
(final dissipated power and contact area / volume).

Supports the paper's experiment variants via flags:
  --constraint {PERIMETER,VOLUME}   override the constraint type (Fig 12a vs 12b)
  --desired-perim / --desired-vol   override constraint targets (Pareto sweep)
  --fix-latent "z1,z2"              orientation-only mode (experiment 3.2):
                                    pin the latent code (a chosen micro-structure)
                                    and optimise orientation only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "TOMAS")
for _p in ("fluid_TO", "vae", "dataset", ""):
    sys.path.insert(0, os.path.join(REPO, _p))

import dataset.supershape as supershape          # noqa: E402
import material_constants, fluid_mesher, fluid_bcs, fluid_material  # noqa: E402
import fluid_fe, projection, neural_network, loss, opt_constraints  # noqa: E402
import vae.network as vae_network                # noqa: E402
import vae.data_preprocess as vae_data_prep      # noqa: E402

NORM_TYPES = ([vae_data_prep.NomalizationType.LINEAR] * 8 +
              [vae_data_prep.NomalizationType.LOG] * 2 +
              [vae_data_prep.NomalizationType.LINEAR] * 2)
CXY_CONST = 1e-5  # near-constant super-shape center coords (dropped from the 10-D VAE)


def build(config_data, vae_dir, seed=77):
    bb = fluid_mesher.BoundingBox(**{k: config_data["BOUNDING_BOX"][k]
                                     for k in ("x_min", "x_max", "y_min", "y_max")})
    mesh = fluid_mesher.fluid_mesher(config_data["MESH"]["nelx"],
                                     config_data["MESH"]["nely"], bb)
    bc = fluid_bcs.get_dirichlet_bc_and_fixed_dofs(
        mesh, config_data["BOUNDARY_CONDITIONS"]["char_velocity"],
        fluid_bcs.FluidSampleProblems[config_data["BOUNDARY_CONDITIONS"]["example"]])
    mat_const = material_constants.MaterialConstants(
        kinematic_viscosity=config_data["MATERIAL_CONSTANTS"]["kinematic_viscosity"])
    fmat = fluid_material.FluidMaterial(mesh.elem_dx, mesh.elem_dy, mat_const)
    solver = fluid_fe.FluidSolver(mesh, bc,
                                  fixture_const=config_data["BOUNDARY_CONDITIONS"]["fixture_const"])

    fmap = projection.FourierMap(
        mesh,
        fourier_map_activation=projection.FourierActivation[
            config_data["FOURIER_MAP_PARAMS"]["fourier_map_activation"]],
        num_fourier_terms=config_data["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
        max_radius=config_data["FOURIER_MAP_PARAMS"]["max_radius"],
        min_radius=config_data["FOURIER_MAP_PARAMS"]["min_radius"])

    nn_params = neural_network.NeuralNetworkParameters(
        input_dim=2 * config_data["FOURIER_MAP_PARAMS"]["num_fourier_terms"],
        output_dim=config_data["NEURAL_NETWORK_PARAMS"]["output_dim"],
        num_layers=config_data["NEURAL_NETWORK_PARAMS"]["num_layers"],
        num_neurons_per_layer=config_data["NEURAL_NETWORK_PARAMS"]["num_neurons_per_layer"])
    net = neural_network.TopOptNet(nn_params=nn_params, seed=seed)

    vae_params = vae_network.VAE_Params(input_dim=12, encoder_hidden_dim=600,
                                        latent_dim=2, decoder_hidden_dim=600)
    vae = vae_network.VariationalAutoencoder(vae_params=vae_params)
    vae.encoder.is_training = False
    vae.load_state_dict(torch.load(os.path.join(vae_dir, "vae_net.pt")))
    vae.eval()
    norm = torch.load(os.path.join(vae_dir, "nomalization.pt"))
    return mesh, solver, fmat, fmap, net, vae, norm["max_feature"], norm["min_feature"]


def save_microstructures(mstr_params, theta, nelx, nely, path):
    """Paper-faithful micro-structure plot (cf. Figs 11/16).

    Each super-shape is drawn at its TRUE outline (no clip-to-square, which the
    earlier version used and which squared off circles/leaves). Its size grows
    with the solid volume fraction (sqrt, with a visible floor) so dense cells
    read as large packed shapes and high-permeability cells as small ones --
    reproducing the size variation of the paper's designs -- on the paper's
    pink-on-light-blue palette.
    """
    x, y = supershape.get_euclidean_coords_of_points_on_surf_super_shape(mstr_params, theta)
    area = np.abs(0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1))
    rmax = np.max(np.hypot(x, y), axis=1) + 1e-12
    vf = np.sqrt(area / (np.max(area) + 1e-12))
    scale = 0.92 * np.clip(vf, 0.18, 1.0) / rmax        # half-cell == 0.5
    fig, ax = plt.subplots(figsize=(nelx / 4 + 1.5, nely / 4 + 1.5))
    ax.set_facecolor("#DAE8FC")
    ctr = 0
    for rw in range(nelx):
        for col in range(nely):
            ax.fill((rw + 0.5) + x[ctr] * scale[ctr],
                    (col + 0.5) + y[ctr] * scale[ctr],
                    facecolor="#F4B6B6", edgecolor="#222222", linewidth=0.45)
            ctr += 1
    ax.set_xlim(0, nelx); ax.set_ylim(0, nely)
    for s in ax.spines.values():
        s.set_color("#222222"); s.set_linewidth(1.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout(pad=0.2); fig.savefig(path, dpi=220); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vae-dir", default=os.path.join(REPO, "vae"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "..", "results", "to"))
    ap.add_argument("--tag", default=None, help="output subfolder name")
    ap.add_argument("--constraint", choices=["PERIMETER", "VOLUME"], default=None)
    ap.add_argument("--desired-perim", type=float, default=None)
    ap.add_argument("--desired-vol", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--fix-latent", default=None, help='"z1,z2" -> orientation-only')
    ap.add_argument("--fix-latent-file", default=None,
                    help="M_star.npy from run_latent_space.py -> orientation-only")
    ap.add_argument("--seed", type=int, default=77, help="NN init seed (multi-seed best-of)")
    ap.add_argument("--init-net", default=None,
                    help="warm-start NN weights (.pt) from a previous run (Pareto continuation)")
    ap.add_argument("--with-volume", action="store_true",
                    help="add a solid-volume-fraction constraint (>= desired_vol) ALONGSIDE "
                         "the perimeter/contact-area constraint, so contact area cannot be "
                         "satisfied by piling thin perimeter on the no-flow walls "
                         "(reproduces Fig 14's rising Pareto trend)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(REPO, "notebooks", "vae_config.yaml")) as f:
        _ = yaml.safe_load(f)

    tag = args.tag or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    mesh, solver, fmat, fmap, net, vae, max_feature, min_feature = build(cfg, args.vae_dir, seed=args.seed)
    if args.init_net:
        net.load_state_dict(torch.load(args.init_net))
        print(f"warm-started NN from {args.init_net}")

    # constraint / target overrides
    constraint_type = opt_constraints.ConstraintType[
        args.constraint or cfg["OPTIMIZATION"]["constraint_type"]]
    desired_perim = args.desired_perim if args.desired_perim is not None else cfg["OPTIMIZATION"]["desired_perimeter"]
    desired_vol = args.desired_vol if args.desired_vol is not None else cfg["OPTIMIZATION"]["desired_vol_frac"]
    num_epochs = args.epochs or cfg["OPTIMIZATION"]["num_epochs"]
    lr = args.lr if args.lr is not None else cfg["OPTIMIZATION"]["learning_rate"]
    perim_scale = 2.0

    fix_latent = None
    if args.fix_latent:
        z1, z2 = [float(v) for v in args.fix_latent.split(",")]
        fix_latent = torch.tensor([z1, z2]).double()
    elif args.fix_latent_file:
        ms = np.load(args.fix_latent_file, allow_pickle=True).item()
        fix_latent = torch.tensor(np.asarray(ms["z"], dtype=float)).double()
        print(f"orientation-only: fixed latent z = {fix_latent.tolist()}")

    # symmetry / reflection setup (mirrors the notebook)
    sym_params = projection.SymParams(sym_x_axis_mid_pt=0.5 * cfg["BOUNDING_BOX"]["y_max"],
                                      sym_y_axis_mid_pt=0.5 * cfg["BOUNDING_BOX"]["x_max"])
    sym_x = projection.SymmetryActivation[cfg["FOURIER_MAP_PARAMS"]["symmetry_activation_x_axis"]]
    sym_y = projection.SymmetryActivation[cfg["FOURIER_MAP_PARAMS"]["symmetry_activation_y_axis"]]

    # Element-center coordinates are *inputs*, not design variables; detach the
    # projected features and reflection signs so each epoch builds a fresh graph
    # (the NN weights are the only leaves). Avoids the notebook's retain_graph.
    xy = torch.tensor(solver.mesh.elem_centers).double()
    xy_r, signs = projection.apply_reflection(xy, sym_y, sym_x, sym_params)
    xy_f = fmap.apply_fourier_map(xy_r).detach()
    signs = {k: v.detach() for k, v in signs.items()}
    num_elems = mesh.num_elems

    loss_type = loss.LossTypes[cfg["LOSS"]["method"]]
    loss_params = loss.PenaltyLossParameters(alpha0=cfg["LOSS"]["alpha0"],
                                             del_alpha=cfg["LOSS"]["del_alpha"])
    optimizer = torch.optim.Adam(net.parameters(), amsgrad=True, lr=lr)
    grad_clip = cfg["OPTIMIZATION"]["grad_clip_norm"]

    history = {"epoch": [], "dissipated_power": [], "contact_area": []}
    J0 = {"val": cfg["OPTIMIZATION"]["init_objective"]}
    eps = 1e-6
    last = {}

    def forward(epoch):
        latent, theta = net(xy_f)
        theta = theta * signs["X"] * signs["Y"]
        if fix_latent is not None:
            latent = fix_latent.unsqueeze(0).repeat(num_elems, 1)
        decoded = vae.decoder(latent)
        out = vae_data_prep.stack_vae_output(decoded, max_feature, min_feature, NORM_TYPES)
        if constraint_type == opt_constraints.ConstraintType.VOLUME:
            constraint_field = out[:, vae_data_prep.VAE_Fields.shape_area.value]
        else:
            constraint_field = (solver.mesh.elem_dx * perim_scale *
                                out[:, vae_data_prep.VAE_Fields.shape_perim.value])
        C00 = out[:, vae_data_prep.VAE_Fields.homog_c00.value] + eps
        C11 = out[:, vae_data_prep.VAE_Fields.homog_c11.value] + eps
        sp_data = out[:, :vae_data_prep.VAE_Fields.homog_c00.value].clone().detach().numpy()
        mstr = supershape.SuperShapes(*[sp_data[:, i] for i in range(8)])
        fluid_loss, vp_field = solver.fluid_objective_function(fmat, C00, C11, theta)
        cons_list = [opt_constraints.constraint_function(constraint_type, constraint_field,
                                                         desired_vol, desired_perim)]
        if args.with_volume and constraint_type == opt_constraints.ConstraintType.PERIMETER:
            # Secondary solid-volume-fraction constraint (mean solid >= desired_vol).
            # Without it, the contact-area target can be met by piling thin
            # high-perimeter micro-structures on the no-flow top/bottom walls, leaving
            # the flow path open -> dissipated power barely rises with contact area
            # (a flat Pareto front). Forcing >= desired_vol solid distributed couples
            # contact area to permeability (more area -> smaller pores -> higher power).
            vol_field = out[:, vae_data_prep.VAE_Fields.shape_area.value]
            cons_list.append(opt_constraints.constraint_function(
                opt_constraints.ConstraintType.VOLUME, vol_field, desired_vol, desired_perim))
        if epoch in (0, 20):
            J0["val"] = fluid_loss.item()
        net_loss = loss.combined_loss(fluid_loss / J0["val"], cons_list, loss_type,
                                      loss_params, epoch)
        return net_loss, fluid_loss, constraint_field, mstr, theta, vp_field

    t0 = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        net_loss, fluid_loss, cfield, mstr, theta, vp_field = forward(epoch)
        net_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        loss.update_loss_parameters(epoch, loss_type, loss_params, [None])
        ca = torch.sum(cfield).item()
        history["epoch"].append(epoch)
        history["dissipated_power"].append(fluid_loss.item())
        history["contact_area"].append(ca)
        if epoch % 25 == 0 or epoch == num_epochs - 1:
            print(f"  iter {epoch:3d}  J={fluid_loss.item():.4f}  contact_area={ca:.3f}",
                  flush=True)
    # Final design from the converged NN parameters (no-grad forward).
    with torch.no_grad():
        net_loss, fluid_loss, cfield, mstr, theta, vp_field = forward(num_epochs - 1)
        decoded = vae_data_prep.stack_vae_output(
            vae.decoder(fix_latent.unsqueeze(0).repeat(num_elems, 1)) if fix_latent is not None
            else vae.decoder(net(xy_f)[0]), max_feature, min_feature, NORM_TYPES)
        C00 = decoded[:, vae_data_prep.VAE_Fields.homog_c00.value].numpy()
        C11 = decoded[:, vae_data_prep.VAE_Fields.homog_c11.value].numpy()
    theta_np = theta.detach().numpy()
    last = dict(mstr=mstr, theta=theta_np, vp=vp_field,
                J=fluid_loss.item(), ca=torch.sum(cfield).item())
    # Save the design so the true-FEA validation (run_validate_design.py) can
    # re-homogenize it with the ported solver.
    np.savez(os.path.join(out_dir, "design.npz"),
             shape_params=mstr.to_stacked_array(), theta=theta_np,
             C00=C00, C11=C11, constraint_field=cfield.detach().numpy(),
             nelx=mesh.nelx, nely=mesh.nely, elem_dx=mesh.elem_dx,
             constraint_type=constraint_type.name, perim_scale=perim_scale)
    torch.save(net.state_dict(), os.path.join(out_dir, "net.pt"))  # for Pareto warm-start
    dt = time.time() - t0

    # ---- outputs
    save_microstructures(last["mstr"], last["theta"], mesh.nelx, mesh.nely,
                         os.path.join(out_dir, "design.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["dissipated_power"], label="dissipated power")
    plt.plot(history["epoch"], history["contact_area"], label="contact area")
    plt.xlabel("iteration"); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=200); plt.close()

    u, v, _ = solver.get_element_velocity_pressure(last["vp"].flatten())
    vmag = torch.sqrt(u ** 2 + v ** 2).detach().numpy().reshape(mesh.nelx, mesh.nely).T
    plt.figure(figsize=(5, 5))
    im = plt.imshow(vmag, origin="lower", cmap="jet"); plt.colorbar(im)
    plt.title(f"velocity magnitude (max={vmag.max():.3f})"); plt.axis("equal"); plt.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "velocity_mag.png"), dpi=200); plt.close()

    metrics = {"tag": tag, "constraint_type": constraint_type.name,
               "desired_perimeter": desired_perim, "desired_vol_frac": desired_vol,
               "num_epochs": num_epochs, "lr": lr,
               "final_dissipated_power": last["J"], "final_contact_area": last["ca"],
               "max_velocity_magnitude": float(vmag.max()),
               "orientation_only": fix_latent is not None, "runtime_sec": dt}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{tag}] final power={last['J']:.3f} contact_area={last['ca']:.3f} "
          f"({dt:.1f}s)  -> {out_dir}")


if __name__ == "__main__":
    main()
