#!/usr/bin/env bash
# p12 FINAL: stack all best levers -> clean good-flow shapes AT paper-level power.
#   3-D latent (p9: better Pareto)  +  KL 1e-8 sweet spot (p11: recon 13%)
#   +  m<=4 good-flow library (p8b/p10: no stars)  +  anisotropy stratification.
# Target: the paper's "clean circles/lenses + power ~25" point.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p12

echo "########## p12 STEP 1: train 3-D VAE, kl 1e-8, m<=4 lib, stratified ##########"; date
mkdir -p $V
stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
  --min-vf 0 --min-perim 0 --aniso-stratify --aniso-power 0.5 --kl-factor 1e-8 \
  --out-dir $V --results-dir results/vae_p12 2>&1 \
  | stdbuf -oL grep -E "aniso-stratify|Training|Saved|C00:|C11:|^\s+(m|area|perim):|Error|Traceback"

echo "########## p12 STEP 2: M* (3-D, default max-trace) ##########"; date
mkdir -p results/latent_p12
python -u scripts/run_latent_space.py --vae-dir $V --out-dir results/latent_p12 2>&1 \
  | grep -E "\[3.1\]|M. latent|M. TRUE|self-consistent"

echo "########## p12 STEP 3: diffuser (perim 60) ##########"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --tag diffuser_p12 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p12/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p12/design.npz \
  --out results/to/diffuser_p12/design_pp05.png --min-vf 0.05 2>&1 | tail -1

echo "########## p12 STEP 4: bifurcated (perim 70 + per-cell 0.10) ##########"; date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --vae-dir $V \
  --desired-perim 70 --min-cell-vf 0.10 --tag bifurcated_p12 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p12/design.npz 2>&1 | tail -3

echo "########## p12 STEP 5: bent orientation-only (default M*) ##########"; date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --vae-dir $V \
  --fix-latent-file results/latent_p12/M_star.npy --tag bent_orient_p12 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p12/design.npz 2>&1 | tail -3

echo "########## p12 shape check ##########"
python -c "
import numpy as np
for t in ['diffuser_p12','bifurcated_p12']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]
    print(f'  {t}: m mean={m.mean():.2f} frac(m>4)={(m>4).mean():.1%}')
"
date; echo P12_DONE
