#!/usr/bin/env bash
# P6b (2026-06-13): MILD anisotropy stratification (power=0.5) -> partial
# flattening that keeps BOTH the anisotropic tail (bent-pipe M*) AND the
# round/isotropic bulk (bifurcated walls). Goal: a single VAE serving all three
# figures. P6 (power=1) gave bent 14.15 but lozenge walls; P0 gave star walls.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## P6b train VAE (aniso-stratify power=0.5) ##########"
date
mkdir -p TOMAS/vae_p6b
stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
  --min-vf 0 --min-perim 0 --aniso-stratify --aniso-power 0.5 \
  --out-dir TOMAS/vae_p6b --results-dir results/vae_p6b 2>&1 \
  | stdbuf -oL grep -E "aniso-stratify|Training|iter 0 |iter 16150|Saved|: +[0-9]+\.[0-9]+%|Error|Traceback|CUDA"

echo "---------- P6b latent diagnostics ----------"
python -u tools/diag_latent.py --vae-dir TOMAS/vae_p6b 2>&1 | grep -vE "^\s*$" | head -16

echo "---------- P6b M* (default max-trace) ----------"
mkdir -p results/latent_p6b
python -u scripts/run_latent_space.py --vae-dir TOMAS/vae_p6b \
  --out-dir results/latent_p6b 2>&1 | grep -E "\[3.1\]|M. latent|M. TRUE|self-consistent"

echo "########## P6b bifurcated visual test (pure perim 70) ##########"
date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
  --vae-dir TOMAS/vae_p6b --desired-perim 70 --tag bifurcated_p6b 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p6b/design.npz 2>&1 | tail -3

echo "########## P6b bifurcated + per-cell vf (clean walls) ##########"
date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
  --vae-dir TOMAS/vae_p6b --desired-perim 70 --min-cell-vf 0.10 --tag bifurcated_p6b_pc 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p6b_pc/design.npz 2>&1 | tail -3

echo "########## P6b bent-pipe ultimate gate (default M*) ##########"
date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
  --vae-dir TOMAS/vae_p6b --fix-latent-file results/latent_p6b/M_star.npy \
  --tag bent_orient_p6b 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p6b/design.npz 2>&1 | tail -3
date
echo P6B_DONE
