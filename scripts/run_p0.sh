#!/usr/bin/env bash
# P0 (2026-06-12): UPSTREAM-AUTHOR-EXACT end-to-end rerun.
# Rationale: the paper's figures were produced by the authors' CODE, whose
# datagen.yaml uses m in [0.5,11] + independent n in [0.5,10] — NOT the paper
# text's m in [1,22]. Evidence: Fig-10 M* has m=0.604 (inside [0.5,11], outside
# [1,22]). This config was never run in rounds 1-6. VAE: 17000 epochs, constant
# lr 8e-3, NO sample filtering (upstream notebook trains on all 7000).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## P0 STEP 1: regenerate dataset (author ranges m=[0.5,11], indep n=[0.5,10]) ##########"
date
python -u dataset_py/run_data_generation.py --config $NB/datagen.yaml \
  --num-samples 7000 --data-dir TOMAS/dataset --n-jobs -1 2>&1 | grep -E "kept|homogenized [0-9]|Saved dataset|Elapsed|sec"
python -c "
import scipy.io,numpy as np
sp=scipy.io.loadmat('TOMAS/dataset/mstr_shape_parameters_1.mat')['mstr_shape_parameters']
h=scipy.io.loadmat('TOMAS/dataset/homogen_data_1.mat')
print(f'  [check] m range [{sp[:,2].min():.2f},{sp[:,2].max():.2f}] mean={sp[:,2].mean():.2f}')
print(f'  [check] n independent: {not np.allclose(sp[:,3],sp[:,4])}')
print(f'  [check] frac m<1: {(sp[:,2]<1).mean():.3f}  (fish/leaf region present)')
print(f'  [check] C00 range [{h[\"c00\"].min():.2e},{h[\"c00\"].max():.2e}]')
"

echo "########## P0 STEP 2: retrain VAE (17000 ep, CONSTANT lr 8e-3, no filtering) ##########"
date
python -u scripts/run_train_vae.py --force --device cuda --min-vf 0 --min-perim 0 \
  2>&1 | grep -E "Loaded dataset|Training|iter [0-9]+|Saved|: +[0-9]+\.[0-9]+%|reconstruction error"

echo "########## P0 STEP 3: M* selection (max TRUE trace, self-consistency filter) ##########"
date
python -u scripts/run_latent_space.py 2>&1 | grep -vE "^\s*$"

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
