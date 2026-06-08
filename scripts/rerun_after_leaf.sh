#!/usr/bin/env bash
# Re-run the key experiments after regenerating the dataset with correlated
# n1=n2=n3 sampling (which restores the anisotropic leaf/almond micro-structures)
# and retraining the VAE. Validates the headline designs with true FEA.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks

echo "===== 3.1 latent space (M*, Fig7, Fig10, Table1) ====="
python -u scripts/run_latent_space.py 2>&1 | grep -E "3.1|M\*|Saved" | head -8

echo "===== 3.4 diffuser (single contact-area constraint) ====="
python -u scripts/run_to.py --config $NB/config_diffuser.yaml --desired-perim 60 --tag diffuser_leaf 2>&1 | tail -2

echo "===== 3.3b bent pipe, contact area 75.69 ====="
python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml --constraint PERIMETER --desired-perim 75.69 --tag bent_perim_leaf 2>&1 | tail -2

echo "===== 3.7 bifurcated pipe, contact area 70 + volume constraint (distributed) ====="
python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml --desired-perim 70 --with-volume --desired-vol 0.5 --tag bifurcated_leaf 2>&1 | tail -2
echo "----- validate 3.7 -----"
python -u scripts/run_validate_design.py --config $NB/config_biffurcated_pipe.yaml --design results/to/bifurcated_leaf/design.npz 2>&1 | tail -5

echo "===== 3.5 Pareto, warm-start + volume constraint (rising front) ====="
python -u scripts/run_pareto.py --with-volume --desired-vol 0.75 --perims 50,60,70,80 --out-dir results/pareto_leaf 2>&1 | grep -E "contact_area=.*true_power|Monotonic"

echo RERUN_AFTER_LEAF_DONE
