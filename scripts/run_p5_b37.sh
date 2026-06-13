#!/usr/bin/env bash
# P5 stage 2 (2026-06-12): beta = 3e-7 sweet-spot probe, bracketed by
# 1e-7 (wiggly latent, aniso M* alive) < ??? < 1e-6 (smooth, aniso dead)
# < 1e-5 (posterior collapse). Full protocol: train -> diag -> M* (default +
# directional) -> bifurcated visual test -> bent-pipe ultimate gate.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## P5 [b37] train VAE kl=3e-7 ##########"
date
mkdir -p TOMAS/vae_b37
stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
  --min-vf 0 --min-perim 0 --kl-factor 3e-7 --out-dir TOMAS/vae_b37 \
  --results-dir results/vae_b37 2>&1 \
  | stdbuf -oL grep -E "Training|iter 0 |iter 16150|Saved|: +[0-9]+\.[0-9]+%|Error|Traceback|CUDA"

echo "---------- P5 [b37] latent diagnostics ----------"
python -u tools/diag_latent.py --vae-dir TOMAS/vae_b37 2>&1 | grep -vE "^\s*$" | head -16

echo "---------- P5 [b37] M* (directional mode; falls back to max-trace) ----------"
mkdir -p results/latent_b37
python -u scripts/run_latent_space.py --vae-dir TOMAS/vae_b37 \
  --out-dir results/latent_b37 --m-star-aniso 2>&1 | grep -E "\[3.1\]|M\*|directional|self-consistent"

echo "########## P5 [b37] bifurcated visual test (pure perim 70) ##########"
date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
  --vae-dir TOMAS/vae_b37 --desired-perim 70 --tag bifurcated_b37 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_b37/design.npz 2>&1 | tail -3

echo "########## P5 [b37] bent-pipe ultimate gate (orientation-only) ##########"
date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
  --vae-dir TOMAS/vae_b37 --fix-latent-file results/latent_b37/M_star.npy \
  --tag bent_orient_b37 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_b37/design.npz 2>&1 | tail -3
date
echo P5_B37_DONE
