"""Experiment B helper: dataset health check.

Reads the current dataset/*.mat files and reports:
  - actual sample count vs configured num_samples (prune ratio)
  - per-feature stats (min, max, median, IQR, skewness)
  - long-tail diagnostic for c00 / c11 (max/median, log10 spread)
Also writes a four-panel histogram (linear + log) to output/dist_n500.png.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, "notebooks", "datagen.yaml"), "r") as fh:
    cfg = yaml.safe_load(fh)
num_samples_cfg = cfg["DATASET"]["num_samples"]
dataset_num = cfg["DATASET"]["dataset_num"]

ddir = os.path.join(ROOT, "dataset")
sp = scipy.io.loadmat(os.path.join(ddir, f"mstr_shape_parameters_{dataset_num}.mat"))[
    "mstr_shape_parameters"
]
h = scipy.io.loadmat(os.path.join(ddir, f"homogen_data_{dataset_num}.mat"))
area = np.asarray(
    scipy.io.loadmat(os.path.join(ddir, f"mstr_area_{dataset_num}.mat"))["mstr_area"]
).reshape(-1)
perim = np.asarray(
    scipy.io.loadmat(os.path.join(ddir, f"mstr_perim_{dataset_num}.mat"))["mstr_perim"]
).reshape(-1)

n_actual = sp.shape[0]
prune_ratio = 1.0 - n_actual / num_samples_cfg
print(f"num_samples (config) = {num_samples_cfg}")
print(f"actual rows in mstr_shape_parameters = {n_actual}")
print(f"prune ratio = {prune_ratio:.3f}")
print()

c00 = np.asarray(h["c00"]).reshape(-1)
c11 = np.asarray(h["c11"]).reshape(-1)


def stats(name, x):
    pos = x[x > 0]
    median = np.median(x)
    p1, p5, p95, p99 = np.percentile(x, [1, 5, 95, 99])
    spread = (np.max(pos) / np.min(pos)) if pos.size else float("nan")
    print(
        f"{name:>6}: n={x.size}  "
        f"min={x.min():.3e}  p1={p1:.3e}  p5={p5:.3e}  "
        f"med={median:.3e}  p95={p95:.3e}  p99={p99:.3e}  max={x.max():.3e}  "
        f"max/min(>0)={spread:.2e}"
    )


stats("c00", c00)
stats("c11", c11)
stats("perim", perim)
stats("area", area)
print()
print("ratio max/median (long-tail proxy):")
print(f"  c00: {c00.max() / np.median(c00):.2f}")
print(f"  c11: {c11.max() / np.median(c11):.2f}")
print(f"  perim: {perim.max() / np.median(perim):.2f}")
print(f"  area: {area.max() / np.median(area):.2f}")

# four histograms: c00 (log), c11 (log), perim, area
fig, axs = plt.subplots(2, 2, figsize=(9, 6))
axs = axs.flatten()
for ax, (name, x, log) in zip(
    axs,
    [
        ("c00", c00, True),
        ("c11", c11, True),
        ("perim", perim, False),
        ("area", area, False),
    ],
):
    if log:
        ax.hist(np.log10(x[x > 0]), bins=40, color="C0", alpha=0.85)
        ax.set_xlabel(f"log10({name})")
    else:
        ax.hist(x, bins=40, color="C0", alpha=0.85)
        ax.set_xlabel(name)
    ax.set_ylabel("count")
    ax.set_title(f"{name}: n={x.size}")
fig.suptitle(f"Dataset distribution (N={n_actual})")
fig.tight_layout()
out_path = os.path.join(ROOT, "output", f"dist_n{n_actual}.png")
fig.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")
