"""Re-render a saved design (design.npz) with a paper-faithful micro-structure
renderer: each super-shape is drawn at its TRUE outline (no hard clip-to-square),
scaled to sit cleanly inside its cell, on the paper's pink/blue palette.

--size {fill,vf}:
  fill : every shape scaled to ~fill its cell (uniform size) -> best for the
         orientation-only / single-M* designs (paper Fig 11d).
  vf   : shape size grows with solid volume fraction (sqrt) -> reproduces the
         size variation of the paper's Fig 16b (dense circles + small leaves).
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "TOMAS", "dataset"))
import supershape as ss  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", choices=["fill", "vf"], default="fill")
    ap.add_argument("--margin", type=float, default=0.92, help="fraction of half-cell a shape may fill")
    a = ap.parse_args()

    d = np.load(a.design, allow_pickle=True)
    sp = d["shape_params"]; theta = d["theta"]
    nelx, nely = int(d["nelx"]), int(d["nely"])
    mstr = ss.SuperShapes(*[sp[:, i] for i in range(8)])
    x, y = ss.get_euclidean_coords_of_points_on_surf_super_shape(mstr, theta)

    # per-shape solid area fraction (for --size vf): polygon area via shoelace,
    # normalised by the largest so the densest cell fills the cell.
    area = np.abs(0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1))
    rmax = np.max(np.hypot(x, y), axis=1) + 1e-12
    if a.size == "vf":
        vf = np.sqrt(area / (np.max(area) + 1e-12))          # 0..1
        scale = a.margin * np.clip(vf, 0.18, 1.0) / rmax     # keep a visible floor
    else:
        scale = a.margin / rmax                               # every shape fills its cell

    fig, ax = plt.subplots(figsize=(nelx / 4 + 1.5, nely / 4 + 1.5))
    ax.set_facecolor("#DAE8FC")
    ctr = 0
    for rw in range(nelx):
        for col in range(nely):
            xc = (rw + 0.5) + x[ctr] * scale[ctr]
            yc = (col + 0.5) + y[ctr] * scale[ctr]
            ax.fill(xc, yc, facecolor="#F4B6B6", edgecolor="#222222", linewidth=0.45)
            ctr += 1
    ax.set_xlim(0, nelx); ax.set_ylim(0, nely)
    for s in ax.spines.values():
        s.set_color("#222222"); s.set_linewidth(1.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.tight_layout(pad=0.2); fig.savefig(a.out, dpi=220); plt.close(fig)
    print("saved", a.out, "size-mode", a.size)


if __name__ == "__main__":
    main()
