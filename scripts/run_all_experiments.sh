#!/usr/bin/env bash
# Run every TOMAS numerical experiment (§3.1-§3.7) after the VAE is trained.
# Improved version: headline cases use multi-seed best-of (true-FEA validated),
# the Pareto sweep uses warm-start continuation, and every design is validated.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
NB=TOMAS/notebooks
SEEDS=77,1,2
mkdir -p results/logs

run() { echo "===== $1 ====="; shift; "$@" 2>&1 | tail -10; echo; }

# 3.1 / Fig 7 / Fig 10 / Table 1  (produces results/latent/M_star.npy)
run "3.1 latent space (M*, Fig7, Fig10, Table1)" \
    python -u scripts/run_latent_space.py

# 3.4 diffuser (Fig13), contact area 60 -- best-of seeds
run "3.4 diffuser (Fig13) best-of" \
    python -u scripts/run_best_of.py --config $NB/config_diffuser.yaml \
    --desired-perim 60 --seeds $SEEDS --tag diffuser_3p4

# 3.3b bent pipe, perimeter constraint 75.69 (Fig 12b) -- best-of seeds
run "3.3b bent pipe perimeter=75.69 (Fig12b) best-of" \
    python -u scripts/run_best_of.py --config $NB/config_bent_pipe.yaml \
    --constraint PERIMETER --desired-perim 75.69 --seeds $SEEDS --tag bent_perim_3p3b

# 3.3a bent pipe, volume constraint 0.75 (Fig 12a)
run "3.3a bent pipe volume=0.75 (Fig12a)" \
    python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
    --constraint VOLUME --desired-vol 0.75 --tag bent_vol_3p3a
run "  validate 3.3a" python -u scripts/run_validate_design.py \
    --config $NB/config_bent_pipe.yaml --design results/to/bent_vol_3p3a/design.npz

# 3.2 bent pipe, orientation only with M* (Fig 11)
run "3.2 bent pipe orientation-only (Fig11)" \
    python -u scripts/run_to.py --config $NB/config_bent_pipe.yaml \
    --fix-latent-file results/latent/M_star.npy --tag bent_orient_3p2
run "  validate 3.2" python -u scripts/run_validate_design.py \
    --config $NB/config_bent_pipe.yaml --design results/to/bent_orient_3p2/design.npz

# 3.7 bifurcated pipe, contact area 70 (Fig 16)
run "3.7 bifurcated pipe perim=70 (Fig16)" \
    python -u scripts/run_to.py --config $NB/config_biffurcated_pipe.yaml \
    --desired-perim 70 --tag bifurcated_3p7
run "  validate 3.7" python -u scripts/run_validate_design.py \
    --config $NB/config_biffurcated_pipe.yaml --design results/to/bifurcated_3p7/design.npz

# 3.5 Pareto front (Fig 14) -- warm-start continuation + per-point true-FEA validation
run "3.5 Pareto sweep (Fig14) warm-start" \
    python -u scripts/run_pareto.py --perims 40,50,60,70,80

echo "ALL EXPERIMENTS DONE"
