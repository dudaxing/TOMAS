#!/usr/bin/env bash
# Branch A (p8a): EFFECTIVE contact-area metric on the P6b VAE. Geometric
# perimeter x convexity (clamp(area/perim^2 / 0.3136,0,1)) so flow-dead star
# crevices don't count -> optimiser must use compact good-flow shapes to satisfy
# the contact-area target. Most principled fix (faithful to author library;
# corrects the flawed proxy at the source). Compare to P6b (stars, 22.89@CA48).
# NOTE: effective<=geometric, so true (geometric) CA of the result will exceed
# the effective target; run_validate_design reports the true geometric CA.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p6b

echo "########## p8a diffuser (effective-area, target 60) ##########"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --effective-area --tag diffuser_p8a 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p8a/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p8a/design.npz \
  --out results/to/diffuser_p8a/design_pp05.png --min-vf 0.05 2>&1 | tail -1

echo "########## p8a bifurcated (effective-area, target 70) ##########"; date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --vae-dir $V \
  --desired-perim 70 --effective-area --tag bifurcated_p8a 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p8a/design.npz 2>&1 | tail -3

echo "########## p8a bent orientation-only (M* unaffected by CA metric) ##########"; date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --vae-dir $V \
  --fix-latent-file results/latent_p6b/M_star.npy --tag bent_orient_p8a 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p8a/design.npz 2>&1 | tail -3

echo "########## p8a shape check ##########"
python -c "
import numpy as np
for t in ['diffuser_p8a','bifurcated_p8a']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']
    m=sp[:,2]
    print(f'  {t}: m mean={m.mean():.2f} frac(m>4)={(m>4).mean():.1%}')
"
date; echo P8A_DONE
