#!/usr/bin/env bash
# p9b: (1) recover the bent pipe with a DIRECTIONAL M* in the 3-D latent (the
# default max-trace M* was near-isotropic 1.68x -> bent 16.64); (2) demonstrate
# the SYNTHESIS 3-D latent + method A (effective area): clean good-flow shapes
# AT the improved 3-D Pareto (diffuser was 20.6 but star-walled w/o a method).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p9

echo "########## p9b STEP 1: directional M* (3-D) + bent ##########"; date
mkdir -p results/latent_p9_aniso
python -u scripts/run_latent_space.py --vae-dir $V --out-dir results/latent_p9_aniso \
  --m-star-aniso 2>&1 | grep -E "M. latent|M. TRUE|directional"
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --vae-dir $V \
  --fix-latent-file results/latent_p9_aniso/M_star.npy --tag bent_orient_p9aniso 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
  --design results/to/bent_orient_p9aniso/design.npz 2>&1 | tail -3

echo "########## p9b STEP 2: 3-D + method A (effective area) diffuser ##########"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --effective-area --tag diffuser_p9A 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p9A/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p9A/design.npz \
  --out results/to/diffuser_p9A/design_pp05.png --min-vf 0.05 2>&1 | tail -1
python -c "import numpy as np; d=np.load('results/to/diffuser_p9A/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]; print('  diffuser_p9A shapes: m mean=%.2f frac(m>4)=%.0f%%'%(m.mean(),100*(m>4).mean()))"
date; echo P9B_DONE
