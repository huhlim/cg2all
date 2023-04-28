#!/usr/bin/env python

import torch

from residue_constants import (
    ATOM_INDEX_N,
    ATOM_INDEX_CA,
    ATOM_INDEX_C,
    BOND_LENGTH0,
    BOND_ANGLE0,
    TORSION_ANGLE0,
)
from torch_basics import v_size, v_norm_safe, inner_product, torsion_angle, pi


def rmsd_CA(R, R_ref):
    return torch.sqrt(torch.mean(torch.pow(R[:, ATOM_INDEX_CA, :] - R_ref[:, ATOM_INDEX_CA, :], 2)))


def rmsd_rigid(R, R_ref):
    return torch.sqrt(torch.mean(torch.pow(R[:, :3] - R_ref[:, :3], 2)))


def rmsd_all(R, R_ref, mask):
    dr_sq = torch.sum(torch.pow(R - R_ref, 2) * mask[..., None])
    return torch.sqrt(dr_sq / mask.sum())


def rmse_bonded(R, is_continuous):
    bonded = is_continuous[1:]
    n_bonded = torch.sum(bonded)

    # vector: -C -> N
    v1 = R[1:, ATOM_INDEX_N, :] - R[:-1, ATOM_INDEX_C, :]
    d1 = v_size(v1)
    v1 = v_norm_safe(v1)
    rmse_bond_length = torch.sqrt(torch.sum(torch.pow(d1 - BOND_LENGTH0, 2) * bonded) / n_bonded)

    # vector: -CA -> -C
    v0 = v_norm_safe(R[:-1, ATOM_INDEX_C, :] - R[:-1, ATOM_INDEX_CA, :])
    # vector: N -> CA
    v2 = v_norm_safe(R[1:, ATOM_INDEX_CA, :] - R[1:, ATOM_INDEX_N, :])

    # bond angles in radians
    def bond_angle(v1, v2):
        return torch.acos(torch.clamp(inner_product(v1, v2), -1.0, 1.0))

    a01 = bond_angle(-v0, v1)
    a12 = bond_angle(-v1, v2)

    rmse_bond_angle = torch.pow(a01 - BOND_ANGLE0[0], 2)
    rmse_bond_angle += torch.pow(a12 - BOND_ANGLE0[1], 2)
    rmse_bond_angle = torch.sqrt(torch.sum(rmse_bond_angle * bonded) / n_bonded / 2)

    # torsion angles without their signs
    def torsion_angle_without_sign(v0, v1, v2):
        n0 = v_norm_safe(torch.cross(v2, v1))
        n1 = v_norm_safe(torch.cross(-v0, v1))
        angle = bond_angle(n0, n1)
        return angle  # between 0 and pi

    omg_angle = torsion_angle_without_sign(v0, v1, v2)
    d_omg = torch.minimum(omg_angle - TORSION_ANGLE0[0], TORSION_ANGLE0[1] - omg_angle)
    rmse_omg_angle = torch.sqrt(torch.sum(torch.pow(d_omg, 2) * bonded) / n_bonded)

    return rmse_bond_length, rmse_bond_angle, rmse_omg_angle


def rmsd_torsion_angle(sc0, sc_ref, mask):
    sc = torch.acos(torch.clamp(sc0[..., 0], -1.0, 1.0))
    sc = sc * torch.sign(sc0[..., 1])
    d_sc = (sc - sc_ref) / (2.0 * pi)
    d_sc = torch.minumum(d_sc, 2.0 * pi - d_sc) * mask
    #
    d_bb = torch.sqrt(torch.mean(torch.power(d_sc[:, :2], 2)))
    d_sc = torch.sqrt(torch.sum(torch.power(d_sc[:, 3:], 2)) / mask[:, 3:].sum())
    return d_bb, d_sc
