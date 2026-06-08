#!/usr/bin/env bash
# MEDIUM library + CORRELATED n (round 6b): m in [1,6], n1=n2=n3 in [0.5,4].
# Aim: smooth 2-D-VAE reconstruction (low error -> smooth decoder -> smooth
# designs) AND the paper's circle (m~4-6, high n) + lens (m~2, low n) palette.
# Diffuser uses NO volume constraint (paper Fig 13/14 is contact-area only).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## STEP 1: regenerate dataset (m in [1,6], correlated n) ##########"
python -u dataset_py/run_data_generation.py --config $NB/datagen.yaml \
  --num-samples 7000 --data-dir TOMAS/dataset --n-jobs -1 2>&1 | grep -E "kept|homogenized [0-9]|Saved dataset"
python -c "
import scipy.io,numpy as np
sp=scipy.io.loadmat('TOMAS/dataset/mstr_shape_parameters_1.mat')['mstr_shape_parameters']
print(f'  m[{sp[:,2].min():.2f},{sp[:,2].max():.2f}] mean={sp[:,2].mean():.2f}  n corr={np.allclose(sp[:,3],sp[:,4])}')
"
echo "########## STEP 2: retrain VAE ##########"
python -u scripts/run_train_vae.py --force --device cuda --epochs 17000 --lr-min 8e-3 \
  --min-vf 0 --min-perim 0 2>&1 | grep -E "Saved|C00:|C11:|area:"

echo "########## STEP 3: experiments ##########"
echo "----- 3.1 M* -----"
python -u scripts/run_latent_space.py 2>&1 | grep -E "M\*|Saved"
echo "----- 3.2 bent orientation-only (Fig 11) -----"
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --fix-latent-file results/latent/M_star.npy --tag bent_orient_med2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml --design results/to/bent_orient_med2/design.npz 2>&1 | tail -3
echo "----- 3.7 bifurcated + volume (Fig 16) -----"
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --desired-perim 70 --with-volume --desired-vol 0.5 --tag bifurcated_med2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml --design results/to/bifurcated_med2/design.npz 2>&1 | tail -3
echo "----- 3.4 diffuser (NO volume, paper Fig 13) -----"
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --desired-perim 60 --tag diffuser_med2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml --design results/to/diffuser_med2/design.npz 2>&1 | tail -3
echo "----- smoothness check (adjacent |dm|; lower=smoother) -----"
python -c "
import numpy as np
for t in ['bifurcated_med2','diffuser_med2']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']; nx,ny=int(d['nelx']),int(d['nely'])
    m=sp[:,2].reshape(nx,ny); r=(np.abs(np.diff(m,axis=0)).mean()+np.abs(np.diff(m,axis=1)).mean())/2
    print(f'  {t}: m[{m.min():.1f},{m.max():.1f}] adj|dm|={r:.2f}')
"
echo MED2_DONE
