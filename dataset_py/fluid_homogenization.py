"""Numerical fluid (Stokes/Brinkman) homogenization of a periodic unit cell.

This is a faithful Python port of ``dataset/fluidHomogenization.m`` from the
TOMAS repository (Padhy, Suresh, Chandrasekhar 2024). It computes the 2x2
homogenized permeability tensor ``CH`` of a periodic micro-structure given a
binary material-indicator image.

Method (see Andreassen & Andreasen, *Comput. Mater. Sci.* 2014, and Pereira
et al. 2016): a 4-node bilinear, pressure-stabilized element is used. The
velocity-velocity block combines a viscous term ``ke`` with a Brinkman term
``ke_brink`` scaled by an element "inverse permeability" coefficient ``zeta``
(0 for fluid, 1e6 for solid). Periodic boundary conditions are imposed by
mapping the element DOFs onto the unique periodic node set. Two body-force load
cases (f_x = 1 and f_y = 1) are solved and the cell-averaged velocities give the
diagonal permeability components.

Convention for the input image ``x`` (matches the MATLAB code):
    x == 0  -> solid  (zeta = solid_perm, strongly penalizes flow)
    x == 1  -> fluid  (zeta = fluid_perm = 0)

Notes on the port:
* MATLAB is column-major; every ``reshape``/``(:)`` becomes ``order='F'`` here.
* MATLAB is 1-based; node/DOF *numbers* are kept 1-based during the periodic
  remap (to mirror the original arithmetic) and converted to 0-based only when
  indexing Python arrays / building the sparse matrix.
* The original ``sparse(iA, jA, sA)`` accumulates duplicate (i, j) entries.
  ``scipy.sparse.coo_matrix`` does the same on conversion to CSC, so we build
  the triplets directly in the clean, standard element-assembly form rather
  than replicating the exact ``kron``/transpose index juggling.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    import pypardiso
    _HAVE_PARDISO = True
except Exception:  # pragma: no cover
    _HAVE_PARDISO = False


def _solve(A, F):
    """Solve ``A X = F`` for a sparse ``A`` and dense RHS ``F``.

    Uses Intel MKL PARDISO when available (~60x faster than SuperLU on this
    symmetric-indefinite saddle-point system, and bit-identical), otherwise
    falls back to SciPy's SuperLU ``spsolve``.
    """
    if _HAVE_PARDISO:
        return pypardiso.spsolve(A.tocsr(), F)
    return spla.spsolve(A.tocsc(), F)


def _element_mat_vec(a: float, b: float, phi_deg: float):
    """Port of the nested ``elementMatVec`` function.

    Args:
        a: half element size along x  (= dx/2).
        b: half element size along y  (= dy/2).
        phi_deg: cell wall angle in degrees (90 for an orthogonal/square cell).

    Returns:
        (ke, ke_brink, be, pe, le) element matrices/vectors:
            ke       (8, 8) viscous stiffness   (B' CMu B)
            ke_brink (8, 8) Brinkman/mass term  (N' N)
            be       (8, 4) velocity-pressure coupling
            pe       (4, 4) pressure stabilization
            le       (4,)   body-force load (integrated shape functions)
    """
    CMu = np.diag([2.0, 2.0, 1.0])  # constitutive (deviatoric) matrix

    # Two Gauss points in each direction.
    xx = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    yy = xx.copy()
    ww = np.array([1.0, 1.0])

    ke = np.zeros((8, 8))
    ke_brink = np.zeros((8, 8))
    be = np.zeros((8, 4))
    pe = np.zeros((4, 4))
    le = np.zeros((4, 1))

    # NOTE: reproduced verbatim from the MATLAB source, including the use of
    # cos(phi)/sin(phi) with phi in *radians* here (phi = 90.0) while the
    # Jacobian below uses tan(phi*pi/180) in *degrees*. For phi=90 the cot term
    # in the Jacobian vanishes (square cell); h2 only scales the pressure
    # stabilization and does not affect the velocity-derived permeability.
    h2 = 4.0 * (a ** 2 + b ** 2 + 2.0 * a * b * np.abs(np.cos(phi_deg) / np.sin(phi_deg)))
    stab = h2 / 12.0

    L = np.zeros((3, 4))
    L[0, 0] = 1.0
    L[1, 3] = 1.0
    L[2, 1:3] = 1.0

    cot_phi = 1.0 / np.tan(phi_deg * np.pi / 180.0)
    # Nodal coordinates of the (possibly skewed) quad, shape (2, 4).
    node_coords = np.array([
        [-a, a, a + 2.0 * b * cot_phi, 2.0 * b * cot_phi - a],
        [-b, -b, b, b],
    ])

    for ii in range(len(xx)):
        for jj in range(len(yy)):
            xg = xx[ii]
            yg = yy[jj]

            N = 0.25 * np.array([
                [(1 - yg) * (1 - xg), 0, (1 - yg) * (1 + xg), 0,
                 (1 + yg) * (1 + xg), 0, (1 - xg) * (1 + yg), 0],
                [0, (1 - yg) * (1 - xg), 0, (1 - yg) * (1 + xg), 0,
                 (1 + yg) * (1 + xg), 0, (1 - xg) * (1 + yg)],
            ])

            dNx = 0.25 * np.array([-(1 - yg), (1 - yg), (1 + yg), -(1 + yg)])
            dNy = 0.25 * np.array([-(1 - xg), -(1 + xg), (1 + xg), (1 - xg)])

            # Jacobian = [dNx; dNy] @ node_coords.T   -> (2,2)
            dN_xy = np.vstack((dNx, dNy))          # (2, 4)
            J = dN_xy @ node_coords.T              # (2, 2)
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            invJ = (1.0 / detJ) * np.array([[J[1, 1], -J[0, 1]],
                                            [-J[1, 0], J[0, 0]]])
            Bp = invJ @ dN_xy                      # (2, 4)
            weight = ww[ii] * ww[jj] * detJ

            G = np.zeros((4, 4))
            G[0:2, 0:2] = invJ
            G[2:4, 2:4] = invJ

            dN = np.zeros((4, 8))
            dN[0, 0::2] = dNx
            dN[1, 0::2] = dNy
            dN[2, 1::2] = dNx
            dN[3, 1::2] = dNy

            B = L @ G @ dN                         # (3, 8)

            ke += weight * (B.T @ CMu @ B)
            ke_brink += weight * (N.T @ N)
            # divergence vector (B' * [1 1 0]') outer product with pressure SF
            div = B.T @ np.array([1.0, 1.0, 0.0])  # (8,)
            N_press = N[0, 0::2]                   # (4,) = N(1,1:2:end)
            be += weight * np.outer(div, N_press)
            pe += weight * (Bp.T @ Bp * stab)
            le += weight * N_press.reshape(4, 1)

    return ke, ke_brink, be, pe, le.ravel()


def fluid_homogenization(x: np.ndarray,
                         lx: float = 1.0,
                         ly: float = 1.0,
                         solid_perm: float = 1.0e6,
                         fluid_perm: float = 0.0,
                         phi_deg: float = 90.0) -> np.ndarray:
    """Compute the 2x2 homogenized permeability tensor of a unit cell.

    Args:
        x: (nely, nelx) material-indicator array. 0 -> solid, 1 -> fluid.
        lx, ly: unit-cell lengths.
        solid_perm: Brinkman coefficient applied to solid pixels (1e6).
        fluid_perm: Brinkman coefficient applied to fluid pixels (0).
        phi_deg: cell-wall angle in degrees.

    Returns:
        CH: (2, 2) homogenized permeability tensor.
    """
    x = np.asarray(x, dtype=float)
    nely, nelx = x.shape
    dx, dy = lx / nelx, ly / nely
    nel = nelx * nely

    ke, ke_brink, be, pe, le = _element_mat_vec(dx / 2.0, dy / 2.0, phi_deg)

    # ---- DOF tables for the full (non-periodic) mesh, 1-based to match MATLAB.
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(
        (1 + nely, 1 + nelx), order='F')
    edofVec = (2 * nodenrs[0:nely, 0:nelx] + 1).reshape(nel, order='F')
    vel_off = np.array([0, 1, 2 * nely + 2, 2 * nely + 3,
                        2 * nely, 2 * nely + 1, -2, -1])
    edofMat = edofVec[:, None] + vel_off[None, :]            # (nel, 8)

    base_p = 2 * (nelx + 1) * (nely + 1)
    edofVecp = base_p + (nodenrs[0:nely, 0:nelx] + 1).reshape(nel, order='F')
    pres_off = np.array([0, nely + 1, nely, -1])
    edofMatp = edofVecp[:, None] + pres_off[None, :]         # (nel, 4)

    # ---- Periodic boundary conditions: map onto unique periodic nodes.
    nn = (nelx + 1) * (nely + 1)     # total nodes
    nnP = nelx * nely                # unique periodic nodes
    nnPArray = np.arange(1, nnP + 1).reshape((nely, nelx), order='F')
    nnPArray = np.vstack((nnPArray, nnPArray[0:1, :]))       # mirror top border
    nnPArray = np.hstack((nnPArray, nnPArray[:, 0:1]))       # mirror left border
    nnP_flat = nnPArray.reshape(-1, order='F')               # length nn

    dofVector = np.zeros(3 * nn, dtype=np.int64)
    dofVector[0:2 * nn:2] = 2 * nnP_flat - 1                 # x-velocity dofs
    dofVector[1:2 * nn:2] = 2 * nnP_flat                     # y-velocity dofs
    dofVector[2 * nn:3 * nn] = 2 * nnP + nnP_flat            # pressure dofs

    # Remap element dof tables through dofVector (1-based -> periodic 1-based).
    edofMat = dofVector[edofMat - 1]                         # (nel, 8)
    edofMatp = dofVector[edofMatp - 1]                       # (nel, 4)
    ndof = 3 * nnP

    # ---- Element "inverse permeability" coefficient per element.
    zeta_e = np.where(x == 0, solid_perm, fluid_perm).reshape(nel, order='F')

    # ---- Assemble the global sparse matrix from clean element triplets.
    # Velocity-velocity block (8x8): ke + zeta * ke_brink
    KE = ke[None, :, :] + zeta_e[:, None, None] * ke_brink[None, :, :]  # (nel,8,8)
    iv = np.broadcast_to(edofMat[:, :, None], (nel, 8, 8))
    jv = np.broadcast_to(edofMat[:, None, :], (nel, 8, 8))

    # Velocity-pressure coupling (8x4) and its transpose.
    BE = np.broadcast_to(be[None, :, :], (nel, 8, 4))
    ib = np.broadcast_to(edofMat[:, :, None], (nel, 8, 4))
    jb = np.broadcast_to(edofMatp[:, None, :], (nel, 8, 4))

    # Pressure stabilization block (4x4): -pe
    PE = np.broadcast_to(-pe[None, :, :], (nel, 4, 4))
    ip = np.broadcast_to(edofMatp[:, :, None], (nel, 4, 4))
    jp = np.broadcast_to(edofMatp[:, None, :], (nel, 4, 4))

    rows = np.concatenate([iv.ravel(), ib.ravel(), jb.ravel(), ip.ravel()])
    cols = np.concatenate([jv.ravel(), jb.ravel(), ib.ravel(), jp.ravel()])
    vals = np.concatenate([KE.ravel(), BE.ravel(), BE.ravel(), PE.ravel()])

    # 1-based periodic dof numbers -> 0-based.
    A = sp.coo_matrix((vals, (rows - 1, cols - 1)), shape=(ndof, ndof)).tocsc()

    # ---- Load vectors for the two unit body-force cases.
    xdofs = edofMat[:, 0::2].reshape(-1)   # x-velocity dofs (1-based)
    ydofs = edofMat[:, 1::2].reshape(-1)   # y-velocity dofs (1-based)
    le_tiled = np.tile(le, nel)
    F = np.zeros((ndof, 2))
    np.add.at(F[:, 0], xdofs - 1, le_tiled)
    np.add.at(F[:, 1], ydofs - 1, le_tiled)

    # ---- Solve, constraining the last (pressure) dof to zero.
    chi = np.zeros((ndof, 2))
    A_red = A[0:ndof - 1, 0:ndof - 1]
    chi[0:ndof - 1, :] = _solve(A_red, F[0:ndof - 1, :])

    # ---- Homogenized tensor: cell-averaged velocities.
    xdofs_all = dofVector[0:2 * nn:2] - 1   # all-node x-velocity dofs (0-based)
    ydofs_all = dofVector[1:2 * nn:2] - 1   # all-node y-velocity dofs (0-based)
    CH = np.zeros((2, 2))
    CH[0, 0] = chi[xdofs_all, 0].sum()
    CH[0, 1] = chi[ydofs_all, 0].sum()
    CH[1, 0] = chi[xdofs_all, 1].sum()
    CH[1, 1] = chi[ydofs_all, 1].sum()
    CH /= nel
    return CH


if __name__ == "__main__":
    # Quick smoke test: a fully-fluid cell with a single solid pixel.
    nelx = nely = 50
    img = np.ones((nely, nelx))
    img[nely // 2, nelx // 2] = 0
    CH = fluid_homogenization(img)
    print("CH (single solid pixel):\n", CH)
