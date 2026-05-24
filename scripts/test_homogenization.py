"""Sanity tests for the Python Stokes-Brinkman homogenizer.

We don't have MATLAB available on this VM, so we validate the Python
implementation by checking analytically tractable limit cases:

1.  Pure-fluid cell (one solid pixel added in the middle to make the
    problem non-singular, just like the MATLAB script does):
    C should be near-isotropic (C00 ~= C11) and large.
2.  Pure-solid cell (also with one fluid pixel so the matrix is
    non-singular): C should be tiny (order of 1/solidPerm = 1e-6).
3.  Square pillar in the middle: C00 should equal C11 (symmetric).
4.  Horizontal solid stripe: flow is much harder along y than x,
    so C00 >> C11 (axes follow MATLAB convention).
5.  Vertical solid stripe: C11 >> C00.
6.  Convergence: refining the grid should leave C roughly stable.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dataset"))

from fluid_homogenization import fluid_homogenization  # noqa: E402

ZETA = (1e6, 0.0)
PHI = 90.0
LX = 1.0
LY = 1.0


def _format(c):
    return "[[{:+.3e} {:+.3e}], [{:+.3e} {:+.3e}]]".format(
        c[0, 0], c[0, 1], c[1, 0], c[1, 1]
    )


def test_almost_pure_fluid():
    x = np.ones((20, 20))
    x[10, 10] = 0  # mimic the safeguard in generate_homogenized_data.m
    C = fluid_homogenization(LX, LY, ZETA, PHI, x)
    print(f"  almost pure fluid (20x20): C = {_format(C)}")
    assert C[0, 0] > 1e-3, "C00 should be much larger than the solid case"
    assert abs(C[0, 0] - C[1, 1]) / max(C[0, 0], C[1, 1]) < 0.05, "should be near-isotropic"


def test_almost_pure_solid():
    x = np.zeros((20, 20))
    x[10, 10] = 1
    C = fluid_homogenization(LX, LY, ZETA, PHI, x)
    print(f"  almost pure solid (20x20): C = {_format(C)}")
    assert C[0, 0] < 1e-3, "C00 should be tiny when nearly everything is solid"
    assert C[1, 1] < 1e-3


def test_square_pillar():
    x = np.ones((20, 20))
    x[8:12, 8:12] = 0  # solid square in the middle
    C = fluid_homogenization(LX, LY, ZETA, PHI, x)
    print(f"  4x4 solid pillar  (20x20): C = {_format(C)}")
    assert abs(C[0, 0] - C[1, 1]) / max(C[0, 0], C[1, 1]) < 0.05, "should be symmetric"


def test_horizontal_stripe():
    x = np.ones((20, 20))
    x[9:11, :] = 0  # solid stripe across full x
    C = fluid_homogenization(LX, LY, ZETA, PHI, x)
    print(f"  horizontal stripe (20x20): C = {_format(C)}")
    assert C[0, 0] > 5 * C[1, 1], "flow much harder across the stripe"


def test_vertical_stripe():
    x = np.ones((20, 20))
    x[:, 9:11] = 0  # solid stripe across full y
    C = fluid_homogenization(LX, LY, ZETA, PHI, x)
    print(f"  vertical   stripe (20x20): C = {_format(C)}")
    assert C[1, 1] > 5 * C[0, 0], "flow much harder across the stripe"


def test_grid_convergence():
    print("  grid convergence on 4x4 pillar:")
    for n in (16, 32, 48):
        x = np.ones((n, n))
        m = n // 4
        c0 = n // 2 - m // 2
        x[c0 : c0 + m, c0 : c0 + m] = 0
        C = fluid_homogenization(LX, LY, ZETA, PHI, x)
        print(f"    n={n}: C00={C[0, 0]:.4e}  C11={C[1, 1]:.4e}  C01={C[0, 1]:+.2e}")


def main():
    print("Validating Python fluid_homogenization vs analytic limits ...\n")
    test_almost_pure_fluid()
    test_almost_pure_solid()
    test_square_pillar()
    test_horizontal_stripe()
    test_vertical_stripe()
    test_grid_convergence()
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
