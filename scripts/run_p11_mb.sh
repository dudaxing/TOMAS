#!/usr/bin/env bash
# p11 mb verify: minibatch + cosine LR alternative to the kl8 (KL-knob) winner.
# Confirms minibatch (batch 1024, kl 1e-7) reaches the SAME ~12% C00 recon floor
# as kl8 (full-batch, kl 1e-8) -- minibatch shrinks the effective KL (batch/N),
# so it is equivalent to lowering kl_factor. Both hit the 3-D author-library
# reconstruction floor (~12%, still above the paper's Table-1 0.4-4%).
# Result: m 7.2% / C00 12.3% / C11 11.0% (vs kl8 8.5% / 13.2% / 10.7%).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
mkdir -p TOMAS/vae_p11_mb results/vae_p11_mb
python -u scripts/run_train_vae.py --force --device cuda \
  --min-vf 0 --min-perim 0 --aniso-stratify --aniso-power 0.5 \
  --batch-size 1024 --epochs 4000 --lr-min 1e-4 \
  --out-dir TOMAS/vae_p11_mb --results-dir results/vae_p11_mb
echo P11_MB_DONE
