#!/usr/bin/env bash
# Fair-comparison experiment for Fig 12b: sweep the (decoded) contact-area target
# so the TRUE (re-homogenized) contact area brackets the paper's 75.69, then
# validate each design with true FEA. Lets us compare dissipated power at a
# matched TRUE contact area instead of a matched decoder target.
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1

for P in 79 82; do
  echo "===== RUN target_perim=$P ====="
  python -u scripts/run_to.py --config TOMAS/notebooks/config_bent_pipe.yaml \
    --constraint PERIMETER --desired-perim "$P" --seed 77 --tag "bent_perim_ca$P" 2>&1 | tail -3
  echo "----- VALIDATE target_perim=$P -----"
  python -u scripts/run_validate_design.py --config TOMAS/notebooks/config_bent_pipe.yaml \
    --design "results/to/bent_perim_ca$P/design.npz" 2>&1 | tail -6
done
echo FAIRCOMPARE_ALLDONE
