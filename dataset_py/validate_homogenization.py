"""Physics-based validation of the Python fluid-homogenization port.

Since MATLAB/Octave is not available to diff against the original
``fluidHomogenization.m`` numerically, we validate the port against known
analytic / qualitative behaviour of Stokes flow through a periodic cell:

1. Plane Poiseuille channel: a straight fluid channel of height ``h`` spanning
   the cell (solid walls above/below) has cell-averaged permeability in the
   flow direction  C ~= h**3 / 12  (mean Poiseuille velocity h**2/12 times the
   fluid area fraction h). The transverse component is ~0 (blocked).
2. Transpose/anisotropy: a vertical channel must give the mirror result, i.e.
   C11 ~= h**3/12 and C00 ~= 0 -> confirms x/y bookkeeping is correct.
3. Symmetry: a 4-fold-symmetric pattern gives C00 ~= C11 and tiny off-diagonals.
4. Monotonicity: increasing solid fraction lowers permeability.
"""

import numpy as np
from fluid_homogenization import fluid_homogenization


def horizontal_channel(nel: int, h: float) -> np.ndarray:
    """Fluid (1) band of height fraction h centred in y; solid (0) elsewhere."""
    img = np.zeros((nel, nel))
    lo = int(round((0.5 - h / 2.0) * nel))
    hi = int(round((0.5 + h / 2.0) * nel))
    img[lo:hi, :] = 1.0
    return img


def main():
    nel = 150  # same resolution used for the real dataset
    print(f"=== Validation at {nel}x{nel} ===\n")

    print("[1] Poiseuille channel (flow in x):   expect C00 ~= h^3/12, C11 ~ 0")
    print(f"{'h':>6} {'C00':>12} {'h^3/12':>12} {'ratio':>8} {'C11':>12}")
    for h in (0.3, 0.5, 0.7):
        CH = fluid_homogenization(horizontal_channel(nel, h))
        analytic = h ** 3 / 12.0
        print(f"{h:>6.2f} {CH[0,0]:>12.5e} {analytic:>12.5e} "
              f"{CH[0,0]/analytic:>8.3f} {CH[1,1]:>12.5e}")

    print("\n[2] Transpose check: vertical channel (flow in y), h=0.5")
    CHv = fluid_homogenization(horizontal_channel(nel, 0.5).T)
    print(f"    C00={CHv[0,0]:.5e}  C11={CHv[1,1]:.5e}  "
          f"(expect C11 ~= 0.5^3/12 = {0.5**3/12:.5e}, C00 ~ 0)")

    print("\n[3] Symmetry: centred solid square (4-fold symmetric)")
    img = np.ones((nel, nel))
    c0, c1 = int(0.35 * nel), int(0.65 * nel)
    img[c0:c1, c0:c1] = 0.0
    CHs = fluid_homogenization(img)
    print(f"    C00={CHs[0,0]:.5e}  C11={CHs[1,1]:.5e}  "
          f"C01={CHs[0,1]:.3e}  C10={CHs[1,0]:.3e}")
    print(f"    |C00-C11|/C00 = {abs(CHs[0,0]-CHs[1,1])/CHs[0,0]:.2e} (expect tiny)")
    print(f"    |C01|/C00     = {abs(CHs[0,1])/CHs[0,0]:.2e} (expect << 1)")

    print("\n[4] Monotonicity: permeability decreases as solid fraction grows")
    print(f"{'solid frac':>10} {'trace(C)':>12}")
    for frac in (0.1, 0.25, 0.5, 0.75):
        img = np.ones((nel, nel))
        s = int(round(np.sqrt(frac) * nel))
        lo = (nel - s) // 2
        img[lo:lo + s, lo:lo + s] = 0.0
        CH = fluid_homogenization(img)
        print(f"{frac:>10.2f} {CH[0,0]+CH[1,1]:>12.5e}")


if __name__ == "__main__":
    main()
