#!/usr/bin/env bash
# P6 (2026-06-12): anisotropy-stratified resampling. Hypothesis (archive Sec 12):
# the VAE under-represents the rare anisotropic tail (5% of cells >=4x); flatten
# the anisotropy histogram in training so the decoder PRESERVES anisotropy
# without raising KL (which killed it in P5). Same author-exact data, beta=1e-7.
# Full protocol: train -> diag -> M* (directional+default) -> bifurcated visual
# -> bent-pipe ultimate gate. Compare to P0 (M* true 4.1x, bent 15.86).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## P6 train VAE (aniso-stratified, beta=1e-7) ##########"
date
mkdir -p TOMAS/vae_p6
stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
  --min-vf 0 --min-perim 0 --aniso-stratify --out-dir TOMAS/vae_p6 \
  --results-dir results/vae_p6 2>&1 \
  | stdbuf -oL grep -E "aniso-stratify|Training|iter 0 |iter 16150|Saved|: +[0-9]+\.[0-9]+%|Error|Traceback|CUDA"

echo "---------- P6 latent diagnostics ----------"
python -u tools/diag_latent.py --vae-dir TOMAS/vae_p6 2>&1 | grep -vE "^\s*$" | head -16

echo "---------- P6 M* (default max-trace) ----------"
mkdir -p results/latent_p6
python -u scripts/run_latent_space.py --vae-dir TOMAS/vae_p6 \
  --out-dir results/latent_p6 2>&1 | grep -E "\[3.1\]|M\*|self-consistent"
echo "---------- P6 M* (directional) ----------"
mkdir -p results/latent_p6_aniso
python -u scripts/run_latent_space.py --vae-dir TOMAS/vae_p6 \
  --out-dir results/latent_p6_aniso --m-star-aniso 2>&1 | grep -E "M\*|directional"

echo "########## P6 bifurcated visual test (pure perim 70) ##########"
date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
  --vae-dir TOMAS/vae_p6 --desired-perim 70 --tag bifurcated_p6 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p6/design.npz 2>&1 | tail -3

echo "########## P6 bent-pipe ultimate gate (default M*) ##########"
date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
  --vae-dir TOMAS/vae_p6 --fix-latent-file results/latent_p6/M_star.npy \
  --tag bent_orient_p6 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p6/design.npz 2>&1 | tail -3
date
echo P6_DONE
