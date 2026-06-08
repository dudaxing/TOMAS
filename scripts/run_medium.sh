#!/usr/bin/env bash
# MEDIUM-library pipeline (round 6): m in [1,6], n in [0.5,4] independent.
# Goal: a shape vocabulary the 2-D VAE can reconstruct smoothly (no spiky
# high-m stars, no cell-to-cell noise) -> clean circle/lens designs like the
# paper's Figs 11/16, while still reaching the contact-area targets.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## STEP 1: regenerate dataset (m in [1,6], n in [0.5,4]) ##########"
python -u dataset_py/run_data_generation.py --config $NB/datagen.yaml \
  --num-samples 7000 --data-dir TOMAS/dataset --n-jobs -1 2>&1 | grep -E "kept|homogenized [0-9]|Saved dataset"
python -c "
import scipy.io,numpy as np
sp=scipy.io.loadmat('TOMAS/dataset/mstr_shape_parameters_1.mat')['mstr_shape_parameters']
print(f'  m[{sp[:,2].min():.2f},{sp[:,2].max():.2f}] mean={sp[:,2].mean():.2f}  n[{sp[:,3].min():.2f},{sp[:,3].max():.2f}]')
"
echo "########## STEP 2: retrain VAE ##########"
python -u scripts/run_train_vae.py --force --device cuda --epochs 17000 --lr-min 8e-3 \
  --min-vf 0 --min-perim 0 2>&1 | grep -E "Saved|C00:|C11:|area:"

echo "########## STEP 3: experiments (improved renderer) ##########"
echo "----- 3.1 M* -----"
python -u scripts/run_latent_space.py 2>&1 | grep -E "M\*|Saved"
echo "----- 3.2 bent orientation-only (Fig 11) -----"
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --fix-latent-file results/latent/M_star.npy --tag bent_orient_med 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml --design results/to/bent_orient_med/design.npz 2>&1 | tail -3
echo "----- 3.7 bifurcated + volume (Fig 16) -----"
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --desired-perim 70 --with-volume --desired-vol 0.5 --tag bifurcated_med 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml --design results/to/bifurcated_med/design.npz 2>&1 | tail -3
echo "----- 3.4 diffuser + volume (Fig 13) -----"
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --desired-perim 60 --with-volume --desired-vol 0.5 --tag diffuser_med 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml --design results/to/diffuser_med/design.npz 2>&1 | tail -3
echo "----- smoothness check (adjacent |dm|) -----"
python -c "
import numpy as np
for t in ['bifurcated_med','diffuser_med','bent_orient_med']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']; nx,ny=int(d['nelx']),int(d['nely'])
    m=sp[:,2].reshape(nx,ny); r=(np.abs(np.diff(m,0)).mean()+np.abs(np.diff(m,1)).mean())/2
    print(f'  {t}: m[{m.min():.1f},{m.max():.1f}] adj|dm|={r:.2f}')
"
echo MEDIUM_DONE
