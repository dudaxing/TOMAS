# Smoke test artifacts (synthetic, do NOT use for publication)

This folder contains files produced by running the full TOMAS pipeline on
**synthetic** homogenization data so we could validate every Python module
before the real MATLAB output is ready.

| file | meaning |
| --- | --- |
| `homogen_data_1_synthetic.mat` | placeholder produced by `scripts/make_synthetic_homogen.py` (rough scaling law, NOT a real Stokes-Brinkman homogenization) |
| `vae_net.pt` | VAE weights after only 300 epochs on the synthetic data |
| `nomalization.pt` | min/max feature tensors that go with the smoke-test VAE |

To produce real results:

1. Copy `dataset/mstr_images_1.mat` to your Windows MATLAB machine.
2. Run `dataset/run_homogenization_dataset.m` -> produces `homogen_data_1.mat`.
3. Place that file back in `dataset/`.
4. Run `python scripts/train_vae.py --retrain` (full 17000 epochs by default).
5. Run `python scripts/run_multiscale_to.py --config config_diffuser.yaml` (or another config).
