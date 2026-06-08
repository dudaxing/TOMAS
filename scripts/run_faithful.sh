#!/usr/bin/env bash
# Paper-faithful pipeline: dataset with the paper's exact super-shape ranges
# (m in [1,22], independent n in [0.5,10]) -> VAE -> key experiments, rendered
# with the improved paper-style micro-structure renderer.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## STEP 1: regenerate dataset (paper ranges) ##########"
python -u dataset_py/run_data_generation.py --config $NB/datagen.yaml \
  --num-samples 7000 --data-dir TOMAS/dataset --n-jobs -1 2>&1 | grep -E "kept|homogenized [0-9]|Saved dataset"
python -c "
import scipy.io,numpy as np
sp=scipy.io.loadmat('TOMAS/dataset/mstr_shape_parameters_1.mat')['mstr_shape_parameters']
print(f'  m[{sp[:,2].min():.2f},{sp[:,2].max():.2f}] mean={sp[:,2].mean():.2f}  n indep={not np.allclose(sp[:,3],sp[:,4])}')
"
echo "########## STEP 2: retrain VAE ##########"
python -u scripts/run_train_vae.py --force --device cuda --epochs 17000 --lr-min 8e-3 \
  --min-vf 0 --min-perim 0 2>&1 | grep -E "Saved|C00:|C11:|area:"

echo "########## STEP 3: experiments (improved renderer) ##########"
echo "----- 3.1 M* -----"
python -u scripts/run_latent_space.py 2>&1 | grep -E "M\*|Saved"
echo "----- 3.2 bent orientation-only (Fig 11) -----"
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --fix-latent-file results/latent/M_star.npy --tag bent_orient_faithful 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml --design results/to/bent_orient_faithful/design.npz 2>&1 | tail -3
echo "----- 3.7 bifurcated + volume (Fig 16) -----"
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --desired-perim 70 --with-volume --desired-vol 0.5 --tag bifurcated_faithful 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml --design results/to/bifurcated_faithful/design.npz 2>&1 | tail -3
echo "----- 3.4 diffuser -----"
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --desired-perim 60 --tag diffuser_faithful 2>&1 | tail -1
echo FAITHFUL_DONE
