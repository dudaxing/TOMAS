#!/usr/bin/env bash
# P5 stage 1 (2026-06-12): beta (kl_factor) sweep on the author-exact dataset.
# Hypothesis (archive doc Sec 11.8): stronger KL -> smoother/better-organized
# latent map -> per-cell perimeter capped like the paper's decoder -> thick
# circle walls instead of star walls. Stage 1 = train + latent diagnostics +
# M* gate per beta; the winning beta goes to stage 2 (bent + bifurcated).
# Baseline (beta=1e-7, P0): m-wiggle 0.2394/step, star share 39.0%,
# perim p97 4.114, M* true aniso 4.1x.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1

for B in b6:1e-6 b5:1e-5; do
  NAME="${B%%:*}"; KL="${B##*:}"
  echo "########## P5 [$NAME] train VAE kl=$KL (17000 ep, const lr, no filter) ##########"
  date
  mkdir -p TOMAS/vae_$NAME
  stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
    --min-vf 0 --min-perim 0 --kl-factor $KL --out-dir TOMAS/vae_$NAME \
    --results-dir results/vae_$NAME 2>&1 \
    | stdbuf -oL grep -E "Training|iter 0 |iter 16150|Saved|: +[0-9]+\.[0-9]+%|Error|Traceback|CUDA"
  echo "---------- P5 [$NAME] latent diagnostics ----------"
  python -u tools/diag_latent.py --vae-dir TOMAS/vae_$NAME 2>&1 | grep -vE "^\s*$"
  echo "---------- P5 [$NAME] M* gate (true anisotropy must survive) ----------"
  mkdir -p results/latent_$NAME
  python -u scripts/run_latent_space.py --vae-dir TOMAS/vae_$NAME \
    --out-dir results/latent_$NAME 2>&1 | grep -E "\[3.1\]|M\*|self-consistent"
done
date
echo P5_STAGE1_DONE
