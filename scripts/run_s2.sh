#!/usr/bin/env bash
# Approach 2: medium library (m in [1,6], correlated n) -> clean circle+lens
# vocabulary, with a DIRECTIONAL M* (run_latent_space --m-star-aniso, already
# done) for the bent pipe. Bent pipe uses no smoothness (it hurts orientation-
# only); bifurcated/diffuser are already smooth in the medium library.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "----- 3.2 bent orientation-only (directional M*, Fig 11) -----"
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --fix-latent-file results/latent/M_star.npy --tag bent_orient_s2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml --design results/to/bent_orient_s2/design.npz 2>&1 | tail -3
echo "----- 3.7 bifurcated + volume (Fig 16) -----"
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --desired-perim 70 --with-volume --desired-vol 0.5 --tag bifurcated_s2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml --design results/to/bifurcated_s2/design.npz 2>&1 | tail -3
echo "----- 3.4 diffuser + volume (Fig 13) -----"
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --desired-perim 60 --with-volume --desired-vol 0.5 --tag diffuser_s2 2>&1 | tail -1
python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml --design results/to/diffuser_s2/design.npz 2>&1 | tail -3
echo "----- smoothness check -----"
python -c "
import numpy as np
for t in ['bifurcated_s2','diffuser_s2']:
    d=np.load(f'results/to/{t}/design.npz',allow_pickle=True); sp=d['shape_params']; nx,ny=int(d['nelx']),int(d['nely'])
    m=sp[:,2].reshape(nx,ny); r=(np.abs(np.diff(m,0)).mean()+np.abs(np.diff(m,1)).mean())/2
    print(f'  {t}: m[{m.min():.1f},{m.max():.1f}] adj|dm|={r:.2f}')
"
echo S2_DONE
