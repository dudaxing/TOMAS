# How to run TOMAS (Linux Python + Windows MATLAB)

This repository combines Python (data generation, VAE training, neural-net
based topology optimization) with a MATLAB Stokes-Brinkman homogenizer.
The Python parts run in the cloud agent VM (Linux); the MATLAB part runs
on the user's local Windows machine.

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

## 3. Homogenize the microstructures (MATLAB, Windows)

1. Copy `dataset/mstr_images_1.mat` (and the `.m` files in `dataset/`) to
   your local MATLAB session.
2. From inside `dataset/`, run:

   ```matlab
   run_homogenization_dataset
   ```

   That wrapper just calls `generate_homogenized_data('mstr_images_1.mat',
   'homogen_data_1.mat')`, which in turn calls `fluidHomogenization` on
   each of the 100 micro-cells.
3. The script writes `homogen_data_1.mat` containing `mstr, c00, c01, c10,
   c11` (each `N x 1`).
4. Place `homogen_data_1.mat` back in `dataset/`.

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
