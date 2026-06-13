#!/usr/bin/env bash
# Branch B (p8b): FLUID-PRINCIPLED library (m in [0.5,4], good-flow families only)
# + anisotropy-stratified VAE. Tests whether giving the VAE a vocabulary of only
# leaf/lens/fish/circle (no scallop/star) lets a single VAE reach the paper's
# CA60@low-power point with clean good-flow shapes -- closing the gap A/C revealed.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p8b

echo "########## p8b STEP 1: regenerate dataset (m in [0.5,4], good-flow library) ##########"; date
python -u dataset_py/run_data_generation.py --config $NB/datagen.yaml \
  --num-samples 7000 --data-dir TOMAS/dataset --n-jobs -1 2>&1 | grep -E "kept|homogenized [0-9]|Saved dataset"
python -c "
import scipy.io,numpy as np
sp=scipy.io.loadmat('TOMAS/dataset/mstr_shape_parameters_1.mat')['mstr_shape_parameters']
print(f'  [check] m range [{sp[:,2].min():.2f},{sp[:,2].max():.2f}] mean={sp[:,2].mean():.2f}  frac(m<1)={(sp[:,2]<1).mean():.3f}')
"

echo "########## p8b STEP 2: train VAE (aniso-stratify power=0.5) ##########"; date
mkdir -p $V
stdbuf -oL -eL python -u scripts/run_train_vae.py --force --device cuda \
  --min-vf 0 --min-perim 0 --aniso-stratify --aniso-power 0.5 \
  --out-dir $V --results-dir results/vae_p8b 2>&1 \
  | stdbuf -oL grep -E "aniso-stratify|Training|iter 16150|Saved|: +[0-9]+\.[0-9]+%|Error|Traceback"

echo "########## p8b STEP 3: M* (default max-trace) ##########"; date
mkdir -p results/latent_p8b
python -u scripts/run_latent_space.py --vae-dir $V --out-dir results/latent_p8b 2>&1 \
  | grep -E "\[3.1\]|M. latent|M. TRUE|self-consistent"

echo "########## p8b STEP 4: diffuser (perim 60) ##########"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --tag diffuser_p8b 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p8b/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p8b/design.npz \
  --out results/to/diffuser_p8b/design_pp05.png --min-vf 0.05 2>&1 | tail -1

echo "########## p8b STEP 5: bifurcated (perim 70 + per-cell vf>=0.10) ##########"; date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --vae-dir $V \
  --desired-perim 70 --min-cell-vf 0.10 --tag bifurcated_p8b 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p8b/design.npz 2>&1 | tail -3

echo "########## p8b STEP 6: bent orientation-only (default M*) ##########"; date
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --vae-dir $V \
  --fix-latent-file results/latent_p8b/M_star.npy --tag bent_orient_p8b 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p8b/design.npz 2>&1 | tail -3

echo "########## p8b shape check ##########"
python -c "
import numpy as np
for t in ['diffuser_p8b','bifurcated_p8b']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]
    print(f'  {t}: m mean={m.mean():.2f} frac(m>4)={(m>4).mean():.1%}')
"
date; echo P8B_DONE
