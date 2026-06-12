#!/usr/bin/env bash
# P0 resume (2026-06-12): STEP 1 (dataset) already done; resume from STEP 2.
# The first attempt died when the WSL VM terminated mid-VAE-training.
# Output is line-buffered (stdbuf) so progress is visible in the log live.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## P0 STEP 2: retrain VAE (17000 ep, CONSTANT lr 8e-3, no filtering) ##########"
date
stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda --min-vf 0 --min-perim 0 2>&1 \
  | stdbuf -oL grep -E "Loaded dataset|Training|iter [0-9]+|Saved|: +[0-9]+\.[0-9]+%|reconstruction error|Error|Traceback|CUDA"

echo "########## P0 STEP 3: M* selection (max TRUE trace, self-consistency filter) ##########"
date
stdbuf -oL -eL python -u scripts/run_latent_space.py 2>&1 | stdbuf -oL grep -vE "^\s*$"

echo "########## P0 STEP 4: bent pipe orientation-only (Fig 11) ##########"
date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
  --fix-latent-file results/latent/M_star.npy --tag bent_orient_p0 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p0/design.npz 2>&1 | tail -3

echo "########## P0 STEP 5: bifurcated pipe, PERIMETER 70 (Fig 16, author-style) ##########"
date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
  --desired-perim 70 --tag bifurcated_p0 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p0/design.npz 2>&1 | tail -3

echo "########## P0 STEP 6: diffuser, PERIMETER 60 (Fig 13) ##########"
date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml \
  --desired-perim 60 --tag diffuser_p0 2>&1 | tail -2
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p0/design.npz 2>&1 | tail -3

echo "########## P0 STEP 7: smoothness check (adj|dm|) ##########"
python -c "
import numpy as np
for t in ['bent_orient_p0','bifurcated_p0','diffuser_p0']:
    try:
        d=np.load(f'results/to/{t}/design.npz',allow_pickle=True)
        sp=d['shape_params']; nx,ny=int(d['nelx']),int(d['nely'])
        m=sp[:,2].reshape(nx,ny)
        r=(np.abs(np.diff(m,axis=0)).mean()+np.abs(np.diff(m,axis=1)).mean())/2
        print(f'  {t}: m[{m.min():.2f},{m.max():.2f}] adj|dm|={r:.2f}')
    except Exception as e:
        print(f'  {t}: {e}')
"
date
echo P0_DONE
