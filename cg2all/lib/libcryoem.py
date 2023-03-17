#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
import dgl
import mrcfile

import mdtraj

from libconfig import EPS, DTYPE, DATA_HOME
from residue_constants import (
    AMINO_ACID_s,
    MAX_RESIDUE_TYPE,
    MAX_ATOM,
    residue_s,
    BOND_LENGTH0,
    BOND_ANGLE0,
    ATOM_INDEX_N,
    ATOM_INDEX_CA,
    ATOM_INDEX_C,
)
from torch_basics import (
    v_size,
    v_norm,
    inner_product,
    acos_safe,
    torsion_angle,
)

from libloss import loss_f_bonded_energy_aux as loss_f_bonded_energy_aa_aux

import openmm
import openmm.unit


SYSTEM_SIZE_DEPENDENT = True


def trilinear_interpolation(
    r: torch.Tensor, rho: torch.Tensor, xyz_size: torch.Tensor
) -> torch.Tensor:
    """
    Args:
    - r: A torch.Tensor of shape (N, 3)
    - rho: A torch.Tensor of shape (D, H, W)
    - xyz_size: = torch.Tensor([D, H, W])

    Returns:
    - A torch.Tensor of shape (N,)
    """
    # Compute the indices of the 8 corner voxels
    lb = torch.clamp(torch.floor(r), torch.zeros(3, device=r.device), xyz_size - 2)

    # Compute the fractional distances between the point and the nearest corners
    rd = r - lb
    lb = lb.type(torch.long)
    ub = lb + 1

    # Compute the weights for each corner voxel
    c000 = (1.0 - rd[:, 0]) * (1.0 - rd[:, 1]) * (1.0 - rd[:, 2])
    c001 = (1.0 - rd[:, 0]) * (1.0 - rd[:, 1]) * rd[:, 2]
    c010 = (1.0 - rd[:, 0]) * rd[:, 1] * (1.0 - rd[:, 2])
    c011 = (1.0 - rd[:, 0]) * rd[:, 1] * rd[:, 2]
    c100 = rd[:, 0] * (1.0 - rd[:, 1]) * (1.0 - rd[:, 2])
    c101 = rd[:, 0] * (1.0 - rd[:, 1]) * rd[:, 2]
    c110 = rd[:, 0] * rd[:, 1] * (1.0 - rd[:, 2])
    c111 = rd[:, 0] * rd[:, 1] * rd[:, 2]

    out = (
        c000 * rho[lb[:, 0], lb[:, 1], lb[:, 2]]
        + c001 * rho[lb[:, 0], lb[:, 1], ub[:, 2]]
        + c010 * rho[lb[:, 0], ub[:, 1], lb[:, 2]]
        + c011 * rho[lb[:, 0], ub[:, 1], ub[:, 2]]
        + c100 * rho[ub[:, 0], lb[:, 1], lb[:, 2]]
        + c101 * rho[ub[:, 0], lb[:, 1], ub[:, 2]]
        + c110 * rho[ub[:, 0], ub[:, 1], lb[:, 2]]
        + c111 * rho[ub[:, 0], ub[:, 1], ub[:, 2]]
    )
    return out


class CryoEM_loss(object):
    def __init__(self, mrc_fn, data, density_threshold, device, is_all=True):
        self.is_all = is_all
        self.device = device
        self.read_mrc_file(mrc_fn)
        self.set_weights(data)
        self.rho_thr = density_threshold

    def read_mrc_file(self, mrc_fn):
        with mrcfile.open(mrc_fn) as mrc:
            header = mrc.header
            self.header = header
            #
            data = mrc.data
            #
            axis_order = (3 - header.maps, 3 - header.mapr, 3 - header.mapc)
            data = np.moveaxis(data, source=(0, 1, 2), destination=axis_order)

            apix = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
            xyz_origin = np.array([header.origin.x, header.origin.y, header.origin.z])
            if np.all(xyz_origin == 0.0):
                xyz_origin = np.array(
                    [header.nxstart, header.nystart, header.nzstart], dtype=float
                )
                xyz_origin *= apix
            xyz_size = np.array([header.mx, header.my, header.mz], dtype=int)
        #
        self.apix_np = apix
        self.apix = torch.as_tensor(apix, device=self.device)
        self.xyz_origin = torch.as_tensor(xyz_origin, device=self.device)
        self.xyz_size = torch.as_tensor(xyz_size, device=self.device)
        self.rho = torch.swapaxes(torch.as_tensor(data, device=self.device), 0, -1)
        self.rho_max = torch.max(self.rho)

    def set_weights(self, data):
        self.weights = torch.zeros(
            (data.cg.n_residue, MAX_ATOM), dtype=DTYPE, device=self.device
        )
        self.mask = torch.zeros((data.cg.n_residue, MAX_ATOM), dtype=DTYPE)
        #
        for i_res, residue_type_index in enumerate(data.cg.residue_index):
            residue_name = AMINO_ACID_s[residue_type_index]
            if residue_name == "UNK":
                continue
            ref_res = residue_s[residue_name]
            #
            mask = data.cg.atom_mask[i_res]
            for i_atm, atom_name in zip(
                ref_res.output_atom_index, ref_res.output_atom_s
            ):
                if mask[i_atm] > 0.0:
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    self.mask[i_res, i_atm] = mask[i_atm]
                    self.weights[i_res, i_atm] = element.mass
        #
        if self.is_all:
            self.weights = self.weights[self.mask > 0.0]
        else:
            self.weights = self.weights.sum(-1)

    def eval(self, R):
        if self.is_all:
            xyz = (R * 10.0)[self.mask > 0.0]
        else:
            xyz = (R * 10.0)[:, 0]
        xyz = (xyz - self.xyz_origin) / self.apix
        rho = trilinear_interpolation(xyz, self.rho, self.xyz_size)
        v_em = 1.0 - (rho - self.rho_thr) / (self.rho_max - self.rho_thr)
        v_em = torch.clamp(v_em, max=1.0)
        if SYSTEM_SIZE_DEPENDENT:
            u_em = torch.sum(self.weights * v_em)
        else:
            u_em = torch.mean(self.weights * v_em)
        return u_em


def loss_f_bonded_energy_aa(batch: dgl.DGLGraph, R: torch.Tensor, weight_s=(1.0, 0.5)):
    if weight_s[0] == 0.0:
        return 0.0

    bonded = batch.ndata["continuous"][1:]
    n_bonded = torch.sum(bonded)

    # vector: -C -> N
    v1 = R[1:, ATOM_INDEX_N, :] - R[:-1, ATOM_INDEX_C, :]
    #
    # bond lengths
    d1 = v_size(v1) - BOND_LENGTH0
    if SYSTEM_SIZE_DEPENDENT:
        bond_energy = torch.sum(torch.square(d1) * bonded)
    else:
        bond_energy = torch.sum(torch.square(d1) * bonded) / n_bonded
    if weight_s[1] == 0.0:
        return bond_energy * weight_s[0]
    #
    # vector: -CA -> -C
    v0 = R[:-1, ATOM_INDEX_C, :] - R[:-1, ATOM_INDEX_CA, :]
    # vector: N -> CA
    v2 = R[1:, ATOM_INDEX_CA, :] - R[1:, ATOM_INDEX_N, :]
    #
    # bond angles
    def bond_angle(v1, v2):
        return acos_safe(inner_product(v1, v2))

    v0 = v_norm(v0)
    v1 = v_norm(v1)
    v2 = v_norm(v2)
    a01 = bond_angle(-v0, v1) - BOND_ANGLE0[0]
    a12 = bond_angle(-v1, v2) - BOND_ANGLE0[1]
    #
    angle_energy = torch.square(a01) + torch.square(a12)
    if SYSTEM_SIZE_DEPENDENT:
        angle_energy = torch.sum(angle_energy * bonded)
    else:
        angle_energy = torch.sum(angle_energy * bonded) / n_bonded
    #
    return bond_energy * weight_s[0] + angle_energy * weight_s[1]


class DistanceRestraint(object):
    def __init__(self, data, device, radius=1.0):
        valid_residue = data.cg.atom_mask_cg[:, 0] > 0.0
        r_cg0 = data.r_cg[valid_residue, 0].clone().detach().to(device)
        g, d0 = dgl.radius_graph(r_cg0, radius, self_loop=False, get_distances=True)
        self.edge_src, self.edge_dst = g.edges()
        self.d0 = d0[:, 0]

    def eval(self, batch):
        r_cg = batch.ndata["pos"]
        dr = r_cg[self.edge_dst] - r_cg[self.edge_src]
        d = torch.sqrt(torch.square(dr).sum(dim=-1))
        if SYSTEM_SIZE_DEPENDENT:
            loss = torch.sum(torch.square(d - self.d0))
        else:
            loss = torch.mean(torch.square(d - self.d0))
        return loss


class CoarseGrainedGeometryEnergy(object):
    def __init__(self, device):
        self.set_param(device)

    def set_param(self, device):
        with open(DATA_HOME / "calpha_geometry_params.dat") as fp:
            self.vdw = torch.zeros(
                (MAX_RESIDUE_TYPE, MAX_RESIDUE_TYPE), dtype=DTYPE, device=device
            )
            for line in fp:
                x = line.strip().split()
                if x[0] == "BOND_LENGTH":
                    self.b_len0 = (float(x[1]), float(x[2]))
                elif x[0] == "BOND_ANGLE":
                    self.b_ang0 = (float(x[1]), float(x[2]))
                else:
                    i = AMINO_ACID_s.index(x[1])
                    j = AMINO_ACID_s.index(x[2])
                    self.vdw[i, j] = float(x[3])

    def eval(self, batch):
        return self.eval_bonded(batch) + self.eval_vdw(batch)

    def eval_bonded(self, batch):
        r_cg = batch.ndata["pos"]
        #
        bonded = batch.ndata["continuous"][1:]
        n_bonded = torch.sum(bonded)
        v1 = r_cg[1:] - r_cg[:-1]
        d = (v_size(v1) - self.b_len0[0]) / self.b_len0[1]
        if SYSTEM_SIZE_DEPENDENT:
            bond_energy = torch.sum(torch.square(d) * bonded)
        else:
            bond_energy = torch.sum(torch.square(d) * bonded) / n_bonded
        #
        angled = batch.ndata["continuous"][1:] * batch.ndata["continuous"][:-1]
        n_angled = torch.sum(angled)
        v1 = v_norm(v1)
        v0 = -v1
        angle = acos_safe(inner_product(v0, v1))
        angle = (angle - self.b_ang0[0]) / self.b_ang0[1]
        #
        angle_energy = torch.square(angle)
        if SYSTEM_SIZE_DEPENDENT:
            angle_energy = torch.sum(torch.square(angle) * angled)
        else:
            angle_energy = torch.sum(torch.square(angle) * angled) / n_angled
        #
        return bond_energy + angle_energy

    def eval_vdw(self, batch):
        r_cg = batch.ndata["pos"]
        #
        g = dgl.radius_graph(r_cg, 0.5, self_loop=False)
        edges = g.edges()
        sequence_separation = edges[1] - edges[0]
        valid = sequence_separation > 2
        edges = (edges[0][valid], edges[1][valid])
        #
        dij = v_size(r_cg[edges[0]] - r_cg[edges[1]])
        #
        index = batch.ndata["residue_type"]
        i = index[edges[0]]
        j = index[edges[1]]
        vdw_sum = self.vdw[i, j]
        #
        x = -torch.clamp(dij - vdw_sum, max=0.0)
        energy = torch.square(x).sum()
        return energy


class CryoEMLossFunction(object):
    def __init__(self, mrc_fn, data, device, is_all=True, restraint=100.0):
        self.is_all = is_all
        #
        self.cryoem_loss_f = CryoEM_loss(mrc_fn, data, 0.0, device, is_all=is_all)
        self.distance_restraint = DistanceRestraint(data, device, radius=1.0)
        self.geometry_energy = CoarseGrainedGeometryEnergy(device)
        #
        self.weight = {}
        self.weight["cryo_em"] = 1.0
        self.weight["bond_length"] = 1.0
        self.weight["bond_length_aux"] = 1.0
        self.weight["geometry"] = 0.1
        self.weight["restraint"] = restraint

    def eval(self, batch, R):
        loss = {}
        loss["cryo_em"] = self.cryoem_loss_f.eval(R)
        if self.is_all:
            loss["bond_length"] = loss_f_bonded_energy_aa(batch, R)
            loss["bond_length_aux"] = loss_f_bonded_energy_aa_aux(batch, R) * R.size(0)
        loss["geometry"] = self.geometry_energy.eval(batch)
        loss["restraint"] = self.distance_restraint.eval(batch)
        #
        loss_sum = 0.0
        for name, value in loss.items():
            loss_sum = loss_sum + value * self.weight[name]
        return loss_sum, loss


class CryoEM_openmm_energy(openmm.openmm.CustomCompoundBondForce):
    def __init__(self, mrc_fn, psf, density_threshold):
        super().__init__(1, "")
        self.read_mrc_file(mrc_fn)
        self.rho_thr = density_threshold
        self.set_energy_function(psf)

    def read_mrc_file(self, mrc_fn):
        with mrcfile.open(mrc_fn) as mrc:
            header = mrc.header
            self.header = header
            #
            data = mrc.data
            #
            axis_order = (3 - header.maps, 3 - header.mapr, 3 - header.mapc)
            data = np.moveaxis(data, source=(0, 1, 2), destination=axis_order)

            apix = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
            xyz_origin = np.array([header.origin.x, header.origin.y, header.origin.z])
            if np.all(xyz_origin == 0.0):
                xyz_origin = np.array(
                    [header.nxstart, header.nystart, header.nzstart], dtype=float
                )
                xyz_origin *= apix
            xyz_size = np.array([header.mx, header.my, header.mz], dtype=int)
        #
        self.apix = apix * 0.1
        self.xyz_origin = xyz_origin * 0.1
        self.xyz_size = xyz_size
        self.boxsize = (self.xyz_size - 1) * self.apix
        self.rho = np.swapaxes(data, 0, -1)
        self.rho_max = self.rho.max()

    def set_energy_function(self, psf):
        rho = openmm.openmm.Continuous3DFunction(
            *self.xyz_size,
            self.rho.flatten(),
            0.0,
            self.boxsize[0],
            0.0,
            self.boxsize[1],
            0.0,
            self.boxsize[2],
        )
        #
        form = "weight * (1 - (density(x1,y1,z1) - d_thr) / (d_max - d_thr)) ; dt=density(x1,y1,z1)"
        self.setEnergyFunction(form)
        self.addTabulatedFunction("density", rho)
        self.addGlobalParameter("d_max", self.rho_max)
        self.addGlobalParameter("d_thr", self.rho_thr)
        self.addPerBondParameter("weight")
        #
        for i, atom in enumerate(psf.topology.atoms()):
            self.addBond(
                [i], [10.0 * atom.element.mass.value_in_unit(openmm.unit.dalton)]
            )


def construct_distance_restraints(psf, crd, force_const):
    force_const = (
        force_const * openmm.unit.kilojoules_per_mole / openmm.unit.nanometer**2
    )
    bond = openmm.openmm.CustomBondForce("k * (r-r0)^2")
    bond.addGlobalParameter("k", force_const)
    bond.addPerBondParameter("r0")
    #
    calphaIndex = [atom.index for atom in psf.topology.atoms() if atom.name == "CA"]
    xyz = crd.positions.value_in_unit(openmm.unit.nanometer)
    xyz = np.array(xyz)[calphaIndex]
    #
    dr = xyz[None, :] - xyz[:, None]
    d0 = np.sqrt(np.sum(dr**2, -1))
    pair = np.where(d0 < 1.0)
    #
    for i, j in zip(*pair):
        if i >= j:
            continue
        bond.addBond(calphaIndex[i], calphaIndex[j], [d0[i, j] * openmm.unit.nanometer])
    return bond
