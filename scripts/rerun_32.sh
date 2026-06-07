#!/usr/bin/env bash
# Re-run experiment 3.2 (bent pipe, orientation-only) with the corrected,
# self-consistent M* (run_latent_space.py now loads the 12-D VAE and selects by
# true re-homogenized trace(C)). Then validate with true FEA.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1

echo "===== RUN 3.2 orientation-only (new M*) ====="
python -u scripts/run_to.py --config TOMAS/notebooks/config_bent_pipe.yaml \
  --fix-latent-file results/latent/M_star.npy --tag bent_orient_3p2v2 2>&1 | tail -3
echo "----- VALIDATE 3.2 (new M*) -----"
python -u scripts/run_validate_design.py --config TOMAS/notebooks/config_bent_pipe.yaml \
  --design results/to/bent_orient_3p2v2/design.npz 2>&1 | tail -6
echo RERUN32_ALLDONE
