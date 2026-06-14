#!/usr/bin/env bash
# p11b: TO experiments on the best-reconstruction VAE (kl8 = 3-D, kl 1e-8,
# stratified, author m<=11). Reconstruction C00 13% / C11 11% (vs p9 24%/20%,
# 2-D 45%/42%). Tests whether better reconstruction lowers power further.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p11_kl8

echo "########## p11b STEP 1: M* (3-D, default max-trace) ##########"; date
mkdir -p results/latent_p11kl8
python -u scripts/run_latent_space.py --vae-dir $V --out-dir results/latent_p11kl8 2>&1 \
  | grep -E "\[3.1\]|M. latent|M. TRUE|self-consistent"

echo "########## p11b STEP 2: diffuser (perim 60) ##########"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --tag diffuser_p11kl8 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p11kl8/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p11kl8/design.npz \
  --out results/to/diffuser_p11kl8/design_pp05.png --min-vf 0.05 2>&1 | tail -1

echo "########## p11b STEP 3: bifurcated (perim 70 + per-cell 0.10) ##########"; date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --vae-dir $V \
  --desired-perim 70 --min-cell-vf 0.10 --tag bifurcated_p11kl8 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p11kl8/design.npz 2>&1 | tail -3

echo "########## p11b STEP 4: bent orientation-only (default M*) ##########"; date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --vae-dir $V \
  --fix-latent-file results/latent_p11kl8/M_star.npy --tag bent_orient_p11kl8 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p11kl8/design.npz 2>&1 | tail -3

echo "########## p11b shape check ##########"
python -c "
import numpy as np
for t in ['diffuser_p11kl8','bifurcated_p11kl8']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]
    print(f'  {t}: m mean={m.mean():.2f} frac(m>4)={(m>4).mean():.1%}')
"
date; echo P11B_DONE
