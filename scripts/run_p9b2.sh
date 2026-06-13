#!/usr/bin/env bash
# p9b2: SYNTHESIS 3-D latent + method A (effective area) + per-cell, on the
# diffuser AND bifurcated. Goal: clean good-flow shapes AT the improved 3-D
# Pareto (3-D diffuser was 20.6 but star-walled; A should clean it).
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore MKL_NUM_THREADS=1
NB=TOMAS/notebooks; V=TOMAS/vae_p9

echo "### 3D + A: diffuser effective-area gamma=1 target80 ###"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 80 --effective-area --conv-power 1.0 --tag diffuser_p9A 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p9A/design.npz 2>&1 | grep -E "dissipated|contact area"
python tools/rerender_design.py --design results/to/diffuser_p9A/design.npz \
  --out results/to/diffuser_p9A/design_pp05.png --min-vf 0.05 2>&1 | tail -1
python -c "import numpy as np; d=np.load('results/to/diffuser_p9A/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]; print('  shapes: m mean=%.2f frac(m>4)=%.0f%%'%(m.mean(),100*(m>4).mean()))"

echo "### 3D + A: bifurcated effective-area target90 ###"; date
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --vae-dir $V \
  --desired-perim 90 --effective-area --conv-power 1.0 --tag bifurcated_p9A 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml \
  --design results/to/bifurcated_p9A/design.npz 2>&1 | grep -E "dissipated|contact area"
python -c "import numpy as np; d=np.load('results/to/bifurcated_p9A/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]; print('  shapes: m mean=%.2f frac(m>4)=%.0f%%'%(m.mean(),100*(m>4).mean()))"
date; echo P9B2_DONE
