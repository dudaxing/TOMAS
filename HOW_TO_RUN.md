# How to run TOMAS (pure Python)

This repository combines super-shape micro-structure generation, a
Stokes-Brinkman homogenizer, a VAE trained on shape parameters + homogenized
properties, and a neural-network based multiscale fluid topology
optimizer.  All four stages now have headless Python entry-points and
run end-to-end on Linux without MATLAB.

A faithful translation of the original MATLAB homogenizer lives in
``dataset/fluid_homogenization.py``; the original ``.m`` files are kept
under ``dataset/`` for reference / cross-checking.

## 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# scientific stack
pip install numpy scipy pandas matplotlib pyyaml shapely geopandas \
            ipywidgets jupyter notebook nbformat nbclient

# PyTorch (CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# sparse solver for the multiscale TO step (needs SuiteSparse headers)
sudo apt-get install -y libsuitesparse-dev python3.12-dev
CFLAGS="-I/usr/include/suitesparse" CPPFLAGS="-I/usr/include/suitesparse" \
    pip install --no-build-isolation torch-sparse-solve
```

## 2. Generate the super-shape dataset (Python)

```bash
source .venv/bin/activate
python scripts/run_generate_data.py
```

Outputs (compressed `.mat`) go to `dataset/`:

- `mstr_shape_parameters_1.mat`  — `(N, 8)` super-shape parameters
- `mstr_images_1.mat`            — `(N, nelx, nely)` binary microstructures
- `mstr_area_1.mat`              — `(N,)` normalized solid area
- `mstr_perim_1.mat`             — `(N,)` normalized perimeter

You can change `num_samples`, mesh resolution, parameter ranges, and the
dataset index in `notebooks/datagen.yaml`.

## 3. Homogenize the microstructures (Python)

```bash
source .venv/bin/activate
python scripts/test_homogenization.py     # quick sanity tests (~1 s)
python scripts/run_homogenization.py      # full dataset (~9 min for 100x150^2)
```

The Python implementation lives in
``dataset/fluid_homogenization.py``.  It is a direct translation of
``dataset/fluidHomogenization.m`` (same Q2-P1 element pair, same
Brinkman penalty, same periodic-BC remapping, same pressure
stabilisation), validated against analytic limit cases in
``scripts/test_homogenization.py``.

Output is ``dataset/homogen_data_<n>.mat`` containing ``mstr, c00, c01,
c10, c11`` -- the same keys the MATLAB version wrote.

If you prefer to run the original MATLAB code instead:

1. Copy ``dataset/mstr_images_1.mat`` and the ``.m`` files to MATLAB.
2. From inside ``dataset/``, run ``run_homogenization_dataset``.

## 4. Train the VAE (Python)

```bash
source .venv/bin/activate
python scripts/train_vae.py            # 17000 epochs by default
python scripts/train_vae.py --epochs 300 --retrain    # quick smoke test
```

Outputs in `vae/`:

- `vae_net.pt`         — trained weights
- `nomalization.pt`    — per-feature min/max tensors (needed by stage 5)

## 5. Run multiscale topology optimization (Python)

```bash
source .venv/bin/activate
python scripts/run_multiscale_to.py --config config_diffuser.yaml
python scripts/run_multiscale_to.py --config config_bent_pipe.yaml
python scripts/run_multiscale_to.py --config config_biffurcated_pipe.yaml
```

Use `--epochs N` to cap iterations.  Designs and convergence curves are
written to `output/<config_stem>/`.

## Notebook versions

The original Jupyter notebooks (`notebooks/generate_data.ipynb`,
`notebooks/train_vae_main.ipynb`, `notebooks/multiscale_TO_main.ipynb`)
remain available.  The scripts under `scripts/` are equivalent, headless
versions and are easier to drive from a cloud agent.
