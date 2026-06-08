"""Render a grid of random super-shape outlines for a candidate (m, n) range, so
we can VISUALLY check whether a sampling config is leaf/almond-dominated (like the
paper) before committing to a full dataset regeneration. n1=n2=n3 (correlated)."""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TOMAS", "dataset"))
import supershape as ss  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-m", type=float, default=1.5)
    ap.add_argument("--max-m", type=float, default=6.0)
    ap.add_argument("--min-n", type=float, default=0.5)
    ap.add_argument("--max-n", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/shape_samples.png")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    N = a.grid * a.grid
    av = rng.uniform(0.05, 0.75, N)
    bv = rng.uniform(0.05, 0.75, N)
    mv = rng.uniform(a.min_m, a.max_m, N)
    nv = rng.uniform(a.min_n, a.max_n, N)          # correlated n1=n2=n3
    cc = np.full(N, 1e-5)
    sp = ss.SuperShapes(av, bv, mv, nv, nv.copy(), nv.copy(), cc, cc)
    theta = np.zeros(N)
    x, y = ss.get_euclidean_coords_of_points_on_surf_super_shape(sp, theta)
    fig, axs = plt.subplots(a.grid, a.grid, figsize=(a.grid * 1.5, a.grid * 1.5))
    for i, ax in enumerate(axs.ravel()):
        ax.fill(x[i], y[i], facecolor="#f4c2c2", edgecolor="k", lw=1.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"m={mv[i]:.1f} n={nv[i]:.1f}", fontsize=6)
    fig.suptitle(f"super-shapes  m∈[{a.min_m},{a.max_m}]  n∈[{a.min_n},{a.max_n}] (n1=n2=n3)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    plt.savefig(a.out, dpi=130); print("saved", a.out)
    # leaf stats
    leaf = ((mv > 1.5) & (mv < 3) & (nv < 1)).mean()
    twofold = ((mv > 1.5) & (mv < 3)).mean()
    print(f"leaf(m≈2 & n<1) frac={leaf*100:.1f}%  2-fold(m≈2) frac={twofold*100:.1f}%  n<1 frac={(nv<1).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
