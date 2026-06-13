#!/usr/bin/env bash
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore MKL_NUM_THREADS=1
NB=TOMAS/notebooks; V=TOMAS/vae_p6b
echo "## diffuser effective-area + per-cell vf>=0.10 (target 60) ##"; date
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
  --desired-perim 60 --effective-area --min-cell-vf 0.10 --tag diffuser_p8a_pc 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
  --design results/to/diffuser_p8a_pc/design.npz 2>&1 | tail -3
python tools/rerender_design.py --design results/to/diffuser_p8a_pc/design.npz \
  --out results/to/diffuser_p8a_pc/design_pp05.png --min-vf 0.05 2>&1 | tail -1
python -c "import numpy as np; d=np.load('results/to/diffuser_p8a_pc/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]; print('m mean=%.2f frac(m>4)=%.1f%%'%(m.mean(),100*(m>4).mean()))"
echo DONE
