#!/usr/bin/env bash
# Full pipeline for the leaf/almond-DOMINATED micro-structure library
# (datagen.yaml now: m in [1.8,2.6], correlated n in [0.5,3.5]).
#   1. regenerate the 7000-sample dataset
#   2. retrain the VAE (17k epochs, flat LR, no filter)
#   3. re-run the key experiments + true-FEA validation
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "########## STEP 1: regenerate dataset ##########"
python -u dataset_py/run_data_generation.py --config $NB/datagen.yaml \
  --num-samples 7000 --data-dir TOMAS/dataset --n-jobs -1 2>&1 | grep -E "kept|rasteriz|homogenized [0-9]|Saved dataset"
python -c "
import scipy.io,numpy as np,yaml
sp=scipy.io.loadmat('TOMAS/dataset/mstr_shape_parameters_1.mat')['mstr_shape_parameters']
n1,m=sp[:,3],sp[:,2]
print(f'  dataset: m mean={m.mean():.2f}  n mean={n1.mean():.2f}  leaf(m~2&n<1)={(((m>1.5)&(m<3))&(n1<1)).sum()}  n<1={ (n1<1).sum() }')
"
echo "########## STEP 2: retrain VAE ##########"
python -u scripts/run_train_vae.py --force --device cuda --epochs 17000 --lr-min 8e-3 \
  --min-vf 0 --min-perim 0 2>&1 | grep -E "Loaded|Training device|Saved|C00:|C11:|area:"

echo "########## STEP 3: experiments ##########"
echo "----- 3.1 latent / M* -----"
python -u scripts/run_latent_space.py 2>&1 | grep -E "M\*|Saved"
echo "----- 3.4 diffuser -----"
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --desired-perim 60 --tag diffuser_leaf2 2>&1 | tail -1
echo "----- 3.7 bifurcated + volume -----"
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --desired-perim 70 --with-volume --desired-vol 0.5 --tag bifurcated_leaf2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml --design results/to/bifurcated_leaf2/design.npz 2>&1 | tail -4
echo "----- 3.2 bent pipe orientation-only (all-leaf, like Fig 11) -----"
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --fix-latent-file results/latent/M_star.npy --tag bent_orient_leaf2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml --design results/to/bent_orient_leaf2/design.npz 2>&1 | tail -4
echo "----- 3.5 Pareto + volume -----"
python -u scripts/run_pareto.py --with-volume --desired-vol 0.75 --perims 50,60,70,80 --out-dir results/pareto_leaf2 2>&1 | grep -E "contact_area=.*true_power|Monotonic"
echo LEAF_DOMINATED_DONE
