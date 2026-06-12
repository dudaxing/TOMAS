#!/usr/bin/env bash
# P4a (2026-06-12): PER-CELL minimum volume fraction constraint (paper Sec 3.7:
# "we impose a minimum volume constraint on each microstructure"). Contrast vs
# *_p0 (pure contact area) and *_p3a (global volume): expect full tiling (no
# empty dot cells), star walls replaced by distributed compact shapes.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## P4a STEP 1: bifurcated, PERIMETER 70 + per-cell vf>=0.10 ##########"
date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
  --desired-perim 70 --min-cell-vf 0.10 --tag bifurcated_p4a 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p4a/design.npz 2>&1 | tail -3

echo "########## P4a STEP 2: diffuser, PERIMETER 60 + per-cell vf>=0.10 ##########"
date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml \
  --desired-perim 60 --min-cell-vf 0.10 --tag diffuser_p4a 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p4a/design.npz 2>&1 | tail -3

echo "########## P4a STEP 3: min-vf coverage + smoothness ##########"
python -c "
import numpy as np
for t in ['bifurcated_p4a','diffuser_p4a','bifurcated_p0','diffuser_p0']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True)
    sp=d['shape_params']; nx,ny=int(d['nelx']),int(d['nely'])
    m=sp[:,2].reshape(nx,ny)
    r=(np.abs(np.diff(m,axis=0)).mean()+np.abs(np.diff(m,axis=1)).mean())/2
    vf=d['volume_fraction'] if 'volume_fraction' in d.files else None
    extra=''
    if vf is not None:
        extra=f' empty(vf<0.05)={float((vf<0.05).mean()):.1%} vf<0.10={float((vf<0.10).mean()):.1%}'
    print(f'  {t}: m[{m.min():.2f},{m.max():.2f}] adj|dm|={r:.2f}{extra}')
"
date
echo P4A_DONE
