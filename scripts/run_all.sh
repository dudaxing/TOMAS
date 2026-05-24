#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
export MPLBACKEND=Agg

echo "=== Step 1: Super-shape dataset ==="
"$PY" "$ROOT/scripts/01_generate_supershape_data.py"

echo "=== Step 2: MATLAB/Octave homogenization ==="
cd "$ROOT/dataset"
octave --no-gui --quiet "$ROOT/scripts/02_run_homogenization.m" 2>&1 | tee "$ROOT/results/step2_homogenization.log"

echo "=== Step 3: VAE training ==="
"$PY" "$ROOT/scripts/03_train_vae.py" 2>&1 | tee "$ROOT/results/step3_vae_training.log"

echo "=== Step 4: Multiscale TO examples ==="
for cfg in config_diffuser.yaml config_bent_pipe.yaml config_biffurcated_pipe.yaml; do
  echo "--- Running $cfg ---"
  "$PY" "$ROOT/scripts/04_run_multiscale_to.py" "$cfg" 2>&1 | tee "$ROOT/results/step4_${cfg%.yaml}.log"
done

echo "=== Pipeline complete ==="
