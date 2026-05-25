# Comparison: N=100 vs N=500 microstructure dataset

All other settings unchanged (`datagen.yaml` mesh 150×150, VAE 17000 epochs, same TO configs).

## Dataset

| | N=100 | N=500 |
|---|-------|-------|
| Samples | 100 | 500 |
| Homogenization time | ~9 min | ~47 min |
| `homogen_data_1.mat` size | ~4 KB | ~16 KB |

## VAE training

| | N=100 (lr=8e-3) | N=500 (lr=8e-3) | N=500 (lr=2e-3, used for TO) |
|---|-----------------|-----------------|------------------------------|
| Final recon loss | **3.5e-5** | **0.24** (diverged ~iter 6800) | **1.7e-3** |
| Notes | Stable | Unstable with default lr | Stable; ~50× worse recon than N=100 |

At iter 5100 with N=500 and lr=8e-3, recon was ~1.5e-3 before divergence — early stopping could be tried.

## Multiscale TO (final dissipated power J, perimeter target in parentheses)

| Case | Target perim | N=100 J (iter 0 → final) | N=500 J (iter 0 → final) |
|------|--------------|--------------------------|--------------------------|
| Diffuser | 60 | 151 → **17.1** | 92.4 → **30.0** (worse) |
| Bent pipe | 75.69 | 26.7 → **6.27** | 22.8 → **7.15** (slightly worse) |
| Bifurcated | 75 | 48.5 → **21.7** | 38.6 → **23.0** (slightly worse) |

Perimeter constraints are still satisfied for N=500 runs.

## Conclusion

Increasing data from 100 to 500 **did not improve** macro TO with the current VAE recipe:

1. Default lr=8e-3 **fails** on 500 samples (training blow-up).
2. lr=2e-3 trains but **reconstruction error** remains much higher than N=100.
3. Final **J is higher** (worse) on all three benchmarks, especially diffuser (~76% worse vs ~89% reduction).

Next steps to try: early stop VAE at ~5100 epochs (lr=8e-3), lower lr schedule, more epochs with smaller lr, or larger network / adjusted kl_factor.
