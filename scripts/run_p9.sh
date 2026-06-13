#!/usr/bin/env bash
# Branch p9 (2026-06-13): 3-D latent VAE. The A-deepening showed all three
# flow-physics methods are bounded by the 2-D VAE's permeability<->contact-area
# Pareto. A 3-D latent should reconstruct the 6-D super-shape manifold better
# (lower error) -> a better Pareto that lifts all methods toward the paper's
# power (~25). Reuses the on-disk author-exact dataset (m in [0.5,11]); only the
# VAE is retrained (latent_dim=3, + anisotropy stratification to keep an
# anisotropic M* for the bent pipe). Compare recon error & power to P6b (2-D).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p9

echo "########## p9 STEP 1: train 3-D VAE (aniso-stratify, author dataset) ##########"; date
mkdir -p $V
stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
  --min-vf 0 --min-perim 0 --aniso-stratify --aniso-power 0.5 \
  --out-dir $V --results-dir results/vae_p9 2>&1 \
  | stdbuf -oL grep -E "aniso-stratify|Training|iter 16150|Saved|: +[0-9]+\.[0-9]+%|Error|Traceback"

echo "########## p9 STEP 2: M* (3-D grid, default max-trace) ##########"; date
mkdir -p results/latent_p9
python -u scripts/run_latent_space.py --vae-dir $V --out-dir results/latent_p9 2>&1 \
  | grep -E "\[3.1\]|M. latent|M. TRUE|self-consistent"

echo "########## p9 STEP 3: diffuser (perim 60) ##########"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --tag diffuser_p9 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p9/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p9/design.npz \
  --out results/to/diffuser_p9/design_pp05.png --min-vf 0.05 2>&1 | tail -1

echo "########## p9 STEP 4: bifurcated (perim 70 + per-cell 0.10) ##########"; date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --vae-dir $V \
  --desired-perim 70 --min-cell-vf 0.10 --tag bifurcated_p9 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p9/design.npz 2>&1 | tail -3

echo "########## p9 STEP 5: bent orientation-only (default M*) ##########"; date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --vae-dir $V \
  --fix-latent-file results/latent_p9/M_star.npy --tag bent_orient_p9 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p9/design.npz 2>&1 | tail -3
date; echo P9_DONE
