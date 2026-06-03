#!/usr/bin/env bash
# Run every TOMAS numerical experiment (§3.1-§3.7) after the VAE is trained.
# Each step is independent; we keep going on failure so all results are attempted.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
NB=TOMAS/notebooks
mkdir -p results/logs

run() { echo "===== $1 ====="; shift; "$@" 2>&1 | tail -8; echo; }

# 3.1 / Fig 7 / Fig 10 / Table 1  (also produces results/latent/M_star.npy)
run "3.1 latent space (M*, Fig7, Fig10, Table1)" \
    python -u scripts/run_latent_space.py

# 3.4 diffuser convergence (Fig 13), contact area 60
run "3.4 diffuser (Fig13)" \
    python -u scripts/run_to.py --config $NB/config_diffuser.yaml --tag diffuser_3p4

# 3.3b bent pipe, perimeter constraint 75.69 (Fig 12b, paper P=7.56)
run "3.3b bent pipe perimeter=75.69 (Fig12b)" \
    python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
    --constraint PERIMETER --desired-perim 75.69 --tag bent_perim_3p3b

# 3.3a bent pipe, volume constraint 0.75 (Fig 12a, paper P=9.61)
run "3.3a bent pipe volume=0.75 (Fig12a)" \
    python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
    --constraint VOLUME --desired-vol 0.75 --tag bent_vol_3p3a

# 3.2 bent pipe, orientation only with M* (Fig 11, paper P=15.1)
run "3.2 bent pipe orientation-only (Fig11)" \
    python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
    --fix-latent-file results/latent/M_star.npy --tag bent_orient_3p2

# 3.7 bifurcated pipe, contact area 70 (Fig 16)
run "3.7 bifurcated pipe perim=70 (Fig16)" \
    python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
    --desired-perim 70 --tag bifurcated_3p7

# 3.5 Pareto front for diffuser (Fig 14)
run "3.5 Pareto sweep (Fig14)" \
    python -u scripts/run_pareto.py --perims 40,50,60,70,80

# 3.3 / 3.5 validation: true FEA re-homogenization of final designs
run "validate bent-pipe perimeter design (3.3)" \
    python -u scripts/run_validate_design.py --config $NB/config_bent_pipe.yaml \
    --design results/to/bent_perim_3p3b/design.npz
run "validate diffuser pareto-50 design (3.5)" \
    python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
    --design results/pareto/pareto_50/design.npz

echo "ALL EXPERIMENTS DONE"
