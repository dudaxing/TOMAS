#!/usr/bin/env bash
# P7 (2026-06-13): close the diffuser on the P6b single VAE. P6b diffuser (pure
# perim 60) gave true power 22.89 (closest to paper ~25) but CA only 48.2/60
# (-23%, constraint NOT met) -- the anisotropy-rich VAE prefers permeable low-
# perimeter cells. Try three remedies and pick the one that hits CA~60 while
# keeping power near the paper:
#   A: raise the perimeter target (75) so the satisfied CA lands near 60
#   B: per-cell vf>=0.10 (forbids empty cells -> more boundary, fuller tiling)
#   C: per-cell vf>=0.05 (milder)
set -u
source ~/tomas-venv/bin/activate
cd /mnt/e/Working/reproduceTOMAS
export PYTHONWARNINGS=ignore
export MKL_NUM_THREADS=1
NB=TOMAS/notebooks
V=TOMAS/vae_p6b

run () { # tag, extra-args
  local tag="$1"; shift
  python -u scripts/run_to.py --config $NB/config_diffuser.yaml --vae-dir $V \
    --tag "$tag" "$@" 2>&1 | tail -1
  python -u scripts/run_validate_design.py --config $NB/config_diffuser.yaml \
    --design results/to/$tag/design.npz 2>&1 | tail -3
  python tools/rerender_design.py --design results/to/$tag/design.npz \
    --out results/to/$tag/design_pp05.png --min-vf 0.05 2>&1 | tail -1
}

echo "########## P7-A diffuser perim 75 ##########"; date
run diffuser_p7a --desired-perim 75
echo "########## P7-B diffuser perim 60 + per-cell vf>=0.10 ##########"; date
run diffuser_p7b --desired-perim 60 --min-cell-vf 0.10
echo "########## P7-C diffuser perim 60 + per-cell vf>=0.05 ##########"; date
run diffuser_p7c --desired-perim 60 --min-cell-vf 0.05
date; echo P7_DONE
