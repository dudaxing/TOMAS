#!/usr/bin/env bash
# Branch A deepening (p8a2): soften convexity (conv^gamma) + calibrate effective
# target so geometric CA hits ~60 while keeping clean ovals at lower power.
# Baseline A (gamma=1, target60): geom CA 46, power 52.6. Sweep gamma/target.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore MKL_NUM_THREADS=1
NB=TOMAS/notebooks; V=TOMAS/vae_p6b

run () { # tag gamma target
  local tag=$1 g=$2 t=$3
  python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
    --desired-perim $t --effective-area --conv-power $g --tag $tag 2>&1 | tail -1
  python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
    --design results/to/$tag/design.npz 2>&1 | grep -E "dissipated|contact area"
  python -c "import numpy as np; d=np.load('results/to/$tag/design.npz',allow_pickle=True); sp=d['shape_params']; m=sp[:,2]; print('  shapes: m mean=%.2f frac(m>4)=%.0f%%'%(m.mean(),100*(m>4).mean()))"
}

echo "### A2-1: gamma=0.5 target=60 ###"; date; run diffuser_p8a2_g05t60 0.5 60
echo "### A2-2: gamma=0.5 target=80 ###"; run diffuser_p8a2_g05t80 0.5 80
echo "### A2-3: gamma=0.5 target=100 ###"; run diffuser_p8a2_g05t100 0.5 100
echo "### A2-4: gamma=1.0 target=100 ###"; run diffuser_p8a2_g10t100 1.0 100
date; echo P8A2_DONE
