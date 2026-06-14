#!/usr/bin/env bash
# p11: push 3-D VAE reconstruction toward the paper's Table-1 (0.4-4%). Baseline
# p9 (3-D, kl 1e-7, 17000 ep, full-batch, stratified): C00 24% / C11 20%.
# Levers tested (all 3-D, author m<=11 dataset, anisotropy-stratified):
#   kl8 : kl_factor 1e-8 (10x less latent regularization -> freer decoder)
#   kl9 : kl_factor 1e-9 (100x less)
#   mb  : minibatch 1024 + cosine LR + 30000 epochs (train harder)
# Reports per-feature reconstruction for each; the winner gets the TO runs (p11b).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1

train () { # name extra-args...
  local name=$1; shift
  echo "########## p11 [$name] train ($*) ##########"; date
  mkdir -p TOMAS/vae_p11_$name
  stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
    --min-vf 0 --min-perim 0 --aniso-stratify --aniso-power 0.5 \
    --out-dir TOMAS/vae_p11_$name --results-dir results/vae_p11_$name "$@" 2>&1 \
    | stdbuf -oL grep -E "Training|Saved|C00:|C11:|^\s+(a|b|m|n1|n2|n3|perim|area):|Error|Traceback"
}

train kl8 --kl-factor 1e-8
train kl9 --kl-factor 1e-9
train mb  --batch-size 1024 --epochs 30000 --lr-min 1e-4
date; echo P11_DONE
