#!/usr/bin/env bash
# P3a (2026-06-12): dual-constraint (contact area + volume 0.5) reruns of the
# bifurcated pipe and diffuser on the P0 author-exact VAE. Single-variable
# contrast vs *_p0 (pure PERIMETER): does the volume constraint replace the
# high-perimeter star walls with the paper's circle+lens vocabulary?
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## P3a STEP 1: bifurcated, PERIMETER 70 + VOLUME 0.5 ##########"
date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
  --desired-perim 70 --with-volume --desired-vol 0.5 --tag bifurcated_p3a 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p3a/design.npz 2>&1 | tail -3

echo "########## P3a STEP 2: diffuser, PERIMETER 60 + VOLUME 0.5 ##########"
date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml \
  --desired-perim 60 --with-volume --desired-vol 0.5 --tag diffuser_p3a 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p3a/design.npz 2>&1 | tail -3

echo "########## P3a STEP 3: smoothness ##########"
python -c "
import numpy as np
for t in ['bifurcated_p3a','diffuser_p3a']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True)
    sp=d['shape_params']; nx,ny=int(d['nelx']),int(d['nely'])
    m=sp[:,2].reshape(nx,ny)
    r=(np.abs(np.diff(m,axis=0)).mean()+np.abs(np.diff(m,axis=1)).mean())/2
    print(f'  {t}: m[{m.min():.2f},{m.max():.2f}] adj|dm|={r:.2f}')
"
date
echo P3A_DONE
