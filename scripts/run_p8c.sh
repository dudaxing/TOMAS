#!/usr/bin/env bash
# Branch C (p8c): FLOW-QUALITY penalty on the P6b VAE. Penalise high-m/low-n
# (star/gear/concave) cells -> push toward leaf/lens/circle without punishing
# elongation. Test on all three examples + true-FEA + post-process render.
# Baseline to beat: diffuser P6b true 22.89@CA48 (under) / P7 35@CA60.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p6b
W=20.0   # flow-quality weight

echo "########## p8c diffuser (perim 60 + flow-quality) ##########"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --flow-quality-weight $W --tag diffuser_p8c 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p8c/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p8c/design.npz \
  --out results/to/diffuser_p8c/design_pp05.png --min-vf 0.05 2>&1 | tail -1

echo "########## p8c bifurcated (perim 70 + flow-quality) ##########"; date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --vae-dir $V \
  --desired-perim 70 --flow-quality-weight $W --tag bifurcated_p8c 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p8c/design.npz 2>&1 | tail -3

echo "########## p8c bent orientation-only (flow-quality OFF: fixed M*, only theta) ##########"; date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --vae-dir $V \
  --fix-latent-file results/latent_p6b/M_star.npy --tag bent_orient_p8c 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p8c/design.npz 2>&1 | tail -3

echo "########## p8c m/n distribution check (did the penalty move shapes?) ##########"
python -c "
import numpy as np
for t in ['diffuser_p8c','diffuser_p6b','bifurcated_p8c']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']
    m=sp[:,2]; nmin=np.minimum.reduce([sp[:,3],sp[:,4],sp[:,5]])
    print(f'  {t}: m mean={m.mean():.2f} frac(m>4)={ (m>4).mean():.1%}  nmin mean={nmin.mean():.2f} frac(n<1)={(nmin<1).mean():.1%}')
"
date; echo P8C_DONE
