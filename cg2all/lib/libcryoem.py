#!/usr/bin/env python

import os
import sys
import torch
import numpy as np
import dgl
import mrcfile

import mdtraj

from libconfig import DTYPE, DATA_HOME
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
from libdata import resSeq_to_number

from libloss import loss_f_bonded_energy_aux as loss_f_bonded_energy_aa_aux
from libloss import CoarseGrainedGeometryEnergy, loss_f_torsion_energy


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
                xyz_origin = np.array([header.nxstart, header.nystart, header.nzstart], dtype=float)
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
        self.weights = torch.zeros((data.cg.n_residue, MAX_ATOM), dtype=DTYPE, device=self.device)
        self.mask = torch.zeros((data.cg.n_residue, MAX_ATOM), dtype=DTYPE)
        #
        for i_res, residue_type_index in enumerate(data.cg.residue_index):
            residue_name = AMINO_ACID_s[residue_type_index]
            if residue_name == "UNK":
                continue
            ref_res = residue_s[residue_name]
            #
            mask = data.cg.atom_mask[i_res]
            for i_atm, atom_name in zip(ref_res.output_atom_index, ref_res.output_atom_s):
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
        u_em = torch.sum(self.weights * v_em)
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
    bond_energy = torch.sum(torch.square(d1) * bonded)
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
    angle_energy = torch.sum(angle_energy * bonded)
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
        loss = torch.square(d - self.d0)
        loss = torch.sum(loss)
        return loss


class CryoEMLossFunction(object):
    def __init__(
        self,
        mrc_fn,
        data,
        device,
        model,
        model_type="ResidueBasedModel",
        is_all=True,
        restraint=100.0,
    ):
        self.is_all = is_all
        if is_all:
            self.RIGID_OPs = model.RIGID_OPs
            self.TORSION_PARs = model.TORSION_PARs
        #
        self.cryoem_loss_f = CryoEM_loss(mrc_fn, data, 0.0, device, is_all=is_all)
        self.distance_restraint = DistanceRestraint(data, device, radius=1.0)
        self.geometry_energy = CoarseGrainedGeometryEnergy(model_type, device)
        #
        self.weight = {}
        self.weight["cryo_em"] = 1.0
        self.weight["bond"] = 1.0
        self.weight["cg"] = 0.1
        self.weight["restraint"] = restraint

    def eval(self, batch, ret):
        R = ret["R"]
        #
        loss = {}
        loss["cryo_em"] = self.cryoem_loss_f.eval(R)
        if self.is_all:
            if self.weight.get("bond", 0.0) > 0.0:
                loss["bond"] = loss_f_bonded_energy_aa(batch, R)
                loss["bond"] = loss["bond"] + loss_f_bonded_energy_aa_aux(batch, R) * R.size(0)
            if self.weight.get("torsion", 0.0) > 0.0:
                loss["torsion"] = loss_f_torsion_energy(
                    batch, R, ret["ss"], self.TORSION_PARs
                ) * R.size(0)

        loss["cg"] = self.geometry_energy.eval(batch)
        loss["restraint"] = self.distance_restraint.eval(batch)
        #
        loss_sum = 0.0
        for name, value in loss.items():
            loss_sum = loss_sum + value * self.weight[name]
        return loss_sum, loss

    def __call__(self, batch, ret):
        return self.eval(batch, ret)[0]


class MinimizableData(object):
    def __init__(self, pdb_fn, cg_model, is_all=False, radius=1.0, dtype=DTYPE):
        super().__init__()
        #
        self.pdb_fn = pdb_fn
        #
        self.radius = radius
        self.dtype = dtype
        #
        self.cg = cg_model(self.pdb_fn, is_all=is_all)
        #
        self.r_cg = torch.as_tensor(self.cg.R_cg[0], dtype=self.dtype)
        self.r_cg.requires_grad = True

    def convert_to_batch(self, r_cg):
        valid_residue = self.cg.atom_mask_cg[:, 0] > 0.0
        pos = r_cg[valid_residue, :]
        geom_s = self.cg.get_geometry(pos, self.cg.atom_mask_cg, self.cg.continuous[0])
        #
        node_feat = self.cg.geom_to_feature(geom_s, self.cg.continuous, dtype=self.dtype)
        data = dgl.radius_graph(pos[:, 0], self.radius, self_loop=False)
        data.ndata["pos"] = pos[:, 0]
        data.ndata["node_feat_0"] = node_feat["0"][..., None]  # shape=(N, 16, 1)
        data.ndata["node_feat_1"] = node_feat["1"]  # shape=(N, 4, 3)
        #
        edge_src, edge_dst = data.edges()
        data.edata["rel_pos"] = pos[edge_dst, 0] - pos[edge_src, 0]
        #
        data.ndata["chain_index"] = torch.as_tensor(self.cg.chain_index, dtype=torch.long)
        resSeq, resSeqIns = resSeq_to_number(self.cg.resSeq)
        data.ndata["resSeq"] = torch.as_tensor(resSeq, dtype=torch.long)
        data.ndata["resSeqIns"] = torch.as_tensor(resSeqIns, dtype=torch.long)
        data.ndata["residue_type"] = torch.as_tensor(self.cg.residue_index, dtype=torch.long)
        data.ndata["continuous"] = torch.as_tensor(self.cg.continuous[0], dtype=self.dtype)
        data.ndata["output_atom_mask"] = torch.as_tensor(self.cg.atom_mask, dtype=self.dtype)  
        #
        ssbond_index = torch.full((data.num_nodes(),), -1, dtype=torch.long)
        for cys_i, cys_j in self.cg.ssbond_s:
            if cys_i < cys_j:  # because of loss_f_atomic_clash
                ssbond_index[cys_j] = cys_i
            else:
                ssbond_index[cys_i] = cys_j
        data.ndata["ssbond_index"] = ssbond_index
        #
        edge_feat = torch.zeros((data.num_edges(), 3), dtype=self.dtype)  # bonded / ssbond / space
        #
        # bonded
        pair_s = [(i - 1, i) for i, cont in enumerate(self.cg.continuous[0]) if cont]
        pair_s = torch.as_tensor(pair_s, dtype=torch.long)
        has_edges = data.has_edges_between(pair_s[:, 0], pair_s[:, 1])
        pair_s = pair_s[has_edges]
        eid = data.edge_ids(pair_s[:, 0], pair_s[:, 1])
        edge_feat[eid, 0] = 1.0
        eid = data.edge_ids(pair_s[:, 1], pair_s[:, 0])
        edge_feat[eid, 0] = 1.0
        #
        # ssbond
        if len(self.cg.ssbond_s) > 0:
            pair_s = torch.as_tensor(self.cg.ssbond_s, dtype=torch.long)
            has_edges = data.has_edges_between(pair_s[:, 0], pair_s[:, 1])
            pair_s = pair_s[has_edges]
            eid = data.edge_ids(pair_s[:, 0], pair_s[:, 1])
            edge_feat[eid, 1] = 1.0
            eid = data.edge_ids(pair_s[:, 1], pair_s[:, 0])
            edge_feat[eid, 1] = 1.0
        #
        # space
        edge_feat[edge_feat.sum(dim=-1) == 0.0, 2] = 1.0
        data.edata["edge_feat_0"] = edge_feat[..., None]

        return data
