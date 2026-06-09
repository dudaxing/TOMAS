#!/usr/bin/env bash
# Approach 1 calibration (v2): sweep the DECODED-attribute smoothness weight on
# the bifurcated case, reporting decoded power, contact area and roughness adj|dm|.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
CFG=TOMAS/notebooks/config_biffurcated_pipe.yaml

run() {  # $1=smooth_weight  $2=tag
  python -u scripts/run_to.py --config $CFG --desired-perim 70 --with-volume --desired-vol 0.5 \
    --smooth-weight "$1" --tag "$2" 2>&1 | tail -1
  python -c "
import numpy as np
d=np.load('results/to/$2/design.npz',allow_pickle=True); sp=d['shape_params']; nx,ny=int(d['nelx']),int(d['nely'])
m=sp[:,2].reshape(nx,ny); r=(np.abs(np.diff(m,0)).mean()+np.abs(np.diff(m,1)).mean())/2
print(f'   --> $2: m[{m.min():.1f},{m.max():.1f}] adj|dm|={r:.2f}')
"
}

echo "===== sw=1  (decoded-TV) ====="; run 1.0  bifur_d1
echo "===== sw=4  (decoded-TV) ====="; run 4.0  bifur_d4
echo "===== sw=12 (decoded-TV) ====="; run 12.0 bifur_d12
echo SWEEP2_DONE
