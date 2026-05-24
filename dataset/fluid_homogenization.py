"""Stokes-Brinkman fluid homogenization (Python translation of fluidHomogenization.m).

This module re-implements ``dataset/fluidHomogenization.m`` from the
TOMAS project so that the entire pipeline can run on Linux without
MATLAB.

The mathematical model is the standard Stokes-Brinkman formulation with
periodic boundary conditions: solid cells get a very large inverse
permeability ``zeta_solid`` (e.g. 1e6) so flow is effectively blocked,
fluid cells have ``zeta_fluid = 0``.  Two unit-pressure-gradient load
cases are solved, and the homogenized permeability tensor ``C`` is
extracted from the volume-averaged velocity field.

The element matrices use the same Q2-P1 (8-node velocity, 4-node
pressure) layout, two-point Gauss quadrature and stabilisation as the
MATLAB version.  Indexing matches MATLAB column-major conventions; we
build a global sparse matrix with scipy and solve it once per cell with
``scipy.sparse.linalg.splu``.

See :func:`fluid_homogenization` for the public entry point.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _element_mat_vec(
    a: float, b: float, phi_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute element-level matrices for a Q2-P1 quadrilateral.

    Args:
        a: Half of the element width in x (= dx/2).
        b: Half of the element width in y (= dy/2).
        phi_deg: Cell skew angle in degrees (90 = rectangular).

    Returns:
        ke         : (8, 8)  viscous stiffness matrix.
        ke_brink   : (8, 8)  mass-like matrix multiplied by zeta per element.
        be         : (8, 4)  velocity-pressure coupling block.
        pe         : (4, 4)  stabilisation pressure-pressure block.
        le         : (4,)    element load vector for unit pressure gradient.
    """
    phi_rad = phi_deg * np.pi / 180.0
    CMu = np.diag([2.0, 2.0, 1.0])
    xx = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    yy = xx.copy()
    ww = np.array([1.0, 1.0])

    ke = np.zeros((8, 8))
    ke_brink = np.zeros((8, 8))
    be = np.zeros((8, 4))
    pe = np.zeros((4, 4))
    le = np.zeros(4)

    h2 = 4.0 * (a * a + b * b + 2.0 * a * b * abs(np.cos(phi_rad) / np.sin(phi_rad)))
    stab = h2 / 12.0

    L = np.zeros((3, 4))
    L[0, 0] = 1.0
    L[1, 3] = 1.0
    L[2, 1] = 1.0
    L[2, 2] = 1.0

    cot_phi = np.cos(phi_rad) / np.sin(phi_rad)
    verts = np.array(
        [
            [-a, -b],
            [a, -b],
            [a + 2.0 * b * cot_phi, b],
            [2.0 * b * cot_phi - a, b],
        ]
    )

    for ii in range(2):
        for jj in range(2):
            xg = xx[ii]
            yg = yy[jj]
            N = 0.25 * np.array(
                [
                    [
                        (1.0 - yg) * (1.0 - xg), 0.0,
                        (1.0 - yg) * (1.0 + xg), 0.0,
                        (1.0 + yg) * (1.0 + xg), 0.0,
                        (1.0 - xg) * (1.0 + yg), 0.0,
                    ],
                    [
                        0.0, (1.0 - yg) * (1.0 - xg),
                        0.0, (1.0 - yg) * (1.0 + xg),
                        0.0, (1.0 + yg) * (1.0 + xg),
                        0.0, (1.0 - xg) * (1.0 + yg),
                    ],
                ]
            )
            dNx = 0.25 * np.array([-(1.0 - yg), (1.0 - yg), (1.0 + yg), -(1.0 + yg)])
            dNy = 0.25 * np.array([-(1.0 - xg), -(1.0 + xg), (1.0 + xg), (1.0 - xg)])

            J = np.vstack([dNx, dNy]) @ verts
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            invJ = (1.0 / detJ) * np.array(
                [[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]
            )
            Bp = invJ @ np.vstack([dNx, dNy])

            weight = ww[ii] * ww[jj] * detJ

            G = np.zeros((4, 4))
            G[:2, :2] = invJ
            G[2:, 2:] = invJ

            dN = np.zeros((4, 8))
            dN[0, 0::2] = dNx
            dN[1, 0::2] = dNy
            dN[2, 1::2] = dNx
            dN[3, 1::2] = dNy

            B = L @ G @ dN

            ke += weight * (B.T @ CMu @ B)
            ke_brink += weight * (N.T @ N)

            N_uvals = N[0, 0::2]  # (4,)
            BTk = B.T @ np.array([1.0, 1.0, 0.0])  # (8,)
            be += weight * (BTk[:, None] * N_uvals[None, :])
            pe += weight * (Bp.T @ Bp) * stab
            le += weight * N_uvals

    return ke, ke_brink, be, pe, le


def fluid_homogenization(
    lx: float,
    ly: float,
    zeta_vec: Tuple[float, float],
    phi_deg: float,
    x: np.ndarray,
) -> np.ndarray:
    """Compute the homogenized 2x2 permeability tensor of a microstructure.

    Args:
        lx: Cell length in the x direction (typically 1.0).
        ly: Cell length in the y direction (typically 1.0).
        zeta_vec: ``(zeta_solid, zeta_fluid)`` -- inverse permeabilities.
            Use e.g. ``(1e6, 0.0)``.
        phi_deg: Cell skew angle, in degrees (90 = orthogonal cell).
        x: ``(nely, nelx)`` indicator matrix.  ``0`` = solid (high zeta),
            ``1`` = fluid (zero zeta).  Same convention as the MATLAB code.

    Returns:
        ``CH``: ``(2, 2)`` homogenized permeability tensor.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {x.shape}")
    nely, nelx = x.shape
    dx = lx / nelx
    dy = ly / nely
    nel = nelx * nely

    ke, ke_brink, be, pe, le = _element_mat_vec(dx / 2.0, dy / 2.0, phi_deg)

    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape(
        (1 + nely, 1 + nelx), order="F"
    )

    edofVec = (2 * nodenrs[0:nely, 0:nelx] + 1).flatten(order="F").reshape(-1, 1)
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    edofMat = edofVec + offsets  # (nel, 8), 1-based DoFs before periodic remap

    edofVecp = 2 * (nelx + 1) * (nely + 1) + (
        nodenrs[0:nely, 0:nelx] + 1
    ).flatten(order="F").reshape(-1, 1)
    offsets_p = np.array([0, nely + 1, nely, -1])
    edofMatp = edofVecp + offsets_p  # (nel, 4), 1-based DoFs before periodic remap

    nn = (nelx + 1) * (nely + 1)
    nnP = nelx * nely
    nnPArray = np.arange(1, nnP + 1).reshape((nely, nelx), order="F")
    nnPArray = np.vstack([nnPArray, nnPArray[0:1, :]])
    nnPArray = np.hstack([nnPArray, nnPArray[:, 0:1]])
    nnP_flat = nnPArray.flatten(order="F")  # (nn,)

    dofVector = np.zeros(3 * nn, dtype=np.int64)
    dofVector[0 : 2 * nn : 2] = 2 * nnP_flat - 1
    dofVector[1 : 2 * nn : 2] = 2 * nnP_flat
    dofVector[2 * nn : 3 * nn] = 2 * nnP + nnP_flat

    edofMat = dofVector[edofMat.flatten() - 1].reshape(nel, 8)
    edofMatp = dofVector[edofMatp.flatten() - 1].reshape(nel, 4)

    ndof = 3 * nnP

    # Element-wise inverse permeability zeta
    zeta_field = zeta_vec[0] * (x == 0).astype(np.float64) + zeta_vec[1] * (
        x == 1
    ).astype(np.float64)
    zeta_elem = zeta_field.flatten(order="F")  # column-major to match MATLAB

    # Per-element stiffness K_e (with Brinkman correction)
    K_all = ke[None, :, :] + zeta_elem[:, None, None] * ke_brink[None, :, :]

    rows_u = edofMat - 1  # 0-based velocity DoF indices, shape (nel, 8)
    rows_p = edofMatp - 1  # 0-based pressure DoF indices, shape (nel, 4)

    # Broadcast index tensors
    iK = np.broadcast_to(rows_u[:, :, None], (nel, 8, 8))
    jK = np.broadcast_to(rows_u[:, None, :], (nel, 8, 8))
    iB = np.broadcast_to(rows_u[:, :, None], (nel, 8, 4))
    jB = np.broadcast_to(rows_p[:, None, :], (nel, 8, 4))
    iP = np.broadcast_to(rows_p[:, :, None], (nel, 4, 4))
    jP = np.broadcast_to(rows_p[:, None, :], (nel, 4, 4))
    be_all = np.broadcast_to(be[None, :, :], (nel, 8, 4))
    pe_all = np.broadcast_to(pe[None, :, :], (nel, 4, 4))

    row_ind = np.concatenate(
        [iK.ravel(), iB.ravel(), jB.ravel(), iP.ravel()]
    )
    col_ind = np.concatenate(
        [jK.ravel(), jB.ravel(), iB.ravel(), jP.ravel()]
    )
    vals = np.concatenate(
        [K_all.ravel(), be_all.ravel(), be_all.ravel(), -pe_all.ravel()]
    )

    A = coo_matrix((vals, (row_ind, col_ind)), shape=(ndof, ndof)).tocsc()

    u_dofs_elem = edofMat[:, 0::2] - 1  # (nel, 4)
    v_dofs_elem = edofMat[:, 1::2] - 1
    le_tiled = np.tile(le, nel)

    F = np.zeros((ndof, 2))
    np.add.at(F[:, 0], u_dofs_elem.ravel(), le_tiled)
    np.add.at(F[:, 1], v_dofs_elem.ravel(), le_tiled)

    # Constrain the last pressure DoF (matches MATLAB: solfor = 1:ndof-1)
    A_sub = A[:-1, :-1].tocsc()
    F_sub = F[:-1, :]

    solver = splu(A_sub)
    chi_sub_0 = solver.solve(F_sub[:, 0])
    chi_sub_1 = solver.solve(F_sub[:, 1])
    chi = np.zeros((ndof, 2))
    chi[:-1, 0] = chi_sub_0
    chi[:-1, 1] = chi_sub_1

    u_global_idx = dofVector[0 : 2 * nn : 2] - 1
    v_global_idx = dofVector[1 : 2 * nn : 2] - 1

    CH = np.zeros((2, 2))
    CH[0, 0] = chi[u_global_idx, 0].sum()
    CH[0, 1] = chi[v_global_idx, 0].sum()
    CH[1, 0] = chi[u_global_idx, 1].sum()
    CH[1, 1] = chi[v_global_idx, 1].sum()
    CH /= nel

    return CH
