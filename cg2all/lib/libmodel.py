#!/usr/bin/env python

import sys
import copy
import functools
import logging
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import dgl

from se3_transformer import Fiber, SE3Transformer
from se3_transformer.layers import LinearSE3, NormSE3

from residue_constants import (
    MAX_SS,
    MAX_RESIDUE_TYPE,
    MAX_ATOM,
    MAX_TORSION,
    MAX_RIGID,
    ATOM_INDEX_CA,
    ATOM_INDEX_N,
    ATOM_INDEX_C,
    ATOM_INDEX_O,
    RIGID_TRANSFORMS_TENSOR,
    RIGID_TRANSFORMS_DEP,
    RIGID_GROUPS_TENSOR,
    RIGID_GROUPS_DEP,
    TORSION_ENERGY_TENSOR,
    TORSION_ENERGY_DEP,
)
from libloss import loss_f, find_atomic_clash
from torch_basics import (
    v_size,
    v_norm_safe,
    inner_product,
    rotate_matrix,
    rotate_vector,
)
from libmetric import rmsd_CA, rmsd_rigid, rmsd_all, rmse_bonded
from libcg import get_residue_center_of_mass, get_backbone_angles
from libconfig import DTYPE


CONFIG = ConfigDict()

CONFIG["train"] = ConfigDict()
CONFIG["train"]["dataset"] = "pdb.processed"
CONFIG["train"]["md_frame"] = -1
CONFIG["train"]["batch_size"] = 4
CONFIG["train"]["crop_size"] = 384
CONFIG["train"]["lr"] = 1e-3
CONFIG["train"]["lr_sc"] = 1e-2
CONFIG["train"]["lr_gamma"] = 0.995
CONFIG["train"]["use_pt"] = "CA"
CONFIG["train"]["augment"] = ""
CONFIG["train"]["min_cg"] = ""
CONFIG["train"]["perturb_pos"] = -1.0

CONFIG["globals"] = ConfigDict()
CONFIG["globals"]["radius"] = 1.0
CONFIG["globals"]["ss_dep"] = True
CONFIG["globals"]["fix_atom"] = False

# embedding module
EMBEDDING_MODULE = ConfigDict()
EMBEDDING_MODULE["num_embeddings"] = MAX_RESIDUE_TYPE
EMBEDDING_MODULE["embedding_dim"] = 40
CONFIG["embedding_module"] = EMBEDDING_MODULE

# the base config for using ConvLayer or SE3Transformer
STRUCTURE_MODULE = ConfigDict()
STRUCTURE_MODULE["low_memory"] = True
STRUCTURE_MODULE["num_graph_layers"] = 6
STRUCTURE_MODULE["num_linear_layers"] = 6
STRUCTURE_MODULE["num_heads"] = 8  # number of attention heads
STRUCTURE_MODULE["mid_dim"] = 32  # # of neurons in radial_profile
STRUCTURE_MODULE["norm"] = [
    True,
    True,
]  # norm between attention blocks / within attention blocks
STRUCTURE_MODULE["nonlinearity"] = "elu"

# fiber_in: is determined by input features
STRUCTURE_MODULE["fiber_init"] = None
STRUCTURE_MODULE["fiber_struct"] = None
# fiber_out: is determined by outputs
# - degree 0: cosine/sine values of torsion angles
# - degree 1: two for BB rigid body rotation matrix and one for CA translation
STRUCTURE_MODULE["rotation_rep"] = "6D"
STRUCTURE_MODULE["fiber_out"] = None
STRUCTURE_MODULE["fiber_pass"] = [(0, 64), (1, 32)]
# num_degrees and num_channels are for fiber_hidden
# - they will be converted to Fiber using Fiber.create(num_degrees, num_channels)
# - which is {degree: num_channels for degree in range(num_degrees)}
STRUCTURE_MODULE["fiber_hidden"] = None
STRUCTURE_MODULE["num_degrees"] = 3
STRUCTURE_MODULE["num_channels"] = 32
STRUCTURE_MODULE["channels_div"] = 2  # no idea... # of channels is divided by this number
STRUCTURE_MODULE["fiber_edge"] = None
#
STRUCTURE_MODULE["loss_weight"] = ConfigDict()
STRUCTURE_MODULE["loss_weight"].update(
    {
        "rigid_body": 1.0,
        "FAPE_CA": 5.0,
        "FAPE_all": 0.0,
        "FAPE_d_clamp": 1.0,
        "v_cntr": 1.0,
        "bonded_energy": 1.0,
        "rotation_matrix": 1.0,
        "backbone_torsion": 0.0,
        "torsion_angle": 5.0,
        "torsion_energy": 0.1,
        "torsion_energy_clamp": 0.6,
        "atomic_clash": 5.0,
        "atomic_clash_vdw": 1.0,
        "atomic_clash_clamp": 0.0,
        "ss": 0.1,
    }
)
CONFIG["structure_module"] = STRUCTURE_MODULE


def _get_gpu_mem():
    return (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.memory_allocated() / 1024 / 1024,
    )


def set_model_config(arg: dict, cg_model, flattened=True) -> ConfigDict:
    config = copy.deepcopy(CONFIG)
    if flattened:
        config.update_from_flattened_dict(arg)
    else:
        config.update(arg)
    #
    embedding_dim = config.embedding_module.embedding_dim
    if embedding_dim > 0:
        n_node_scalar = cg_model.n_node_scalar + embedding_dim
    else:
        n_node_scalar = cg_model.n_node_scalar + config.embedding_module.num_embeddings
    #
    config.structure_module.fiber_init = []
    if n_node_scalar > 0:
        config.structure_module.fiber_init.append((0, n_node_scalar))
    if cg_model.n_node_vector > 0:
        config.structure_module.fiber_init.append((1, cg_model.n_node_vector))
    #
    config.structure_module.fiber_struct = config.structure_module.fiber_pass
    #
    if config.structure_module.fiber_hidden is None:
        config.structure_module.fiber_hidden = [
            (d, config.structure_module.num_channels)
            for d in range(config.structure_module.num_degrees)
        ]
    #
    if config.structure_module.rotation_rep == "6D":
        config.structure_module["fiber_out"] = [(0, MAX_TORSION * 2), (1, 3)]
    elif config.structure_module.rotation_rep.startswith("quat"):
        x = config.structure_module.rotation_rep.split("_")[1:]
        config.structure_module["fiber_out"] = [
            (0, MAX_TORSION * 2 + int(x[0])),
            (1, 1 + int(x[1])),
        ]
    else:
        raise ValueError(f"Unknown rotation representation, {config.structure_module.rotation_rep}")
    #
    if config.globals.ss_dep:
        fiber_out = []
        for degree, n_feats in config.structure_module.fiber_out:
            if degree == 0:
                n_feats += MAX_SS
            fiber_out.append((degree, n_feats))
        config.structure_module.fiber_out = fiber_out
    #
    if config.globals.get("fix_atom", False):
        cg_model_name = cg_model.NAME
        if cg_model_name in ["CalphaBasedModel", "CalphaCMModel", "CalphaSCModel"]:
            fix_atom = [True, False, False]
        elif cg_model_name in ["BackboneModel"]:
            fix_atom = [True, True, False]
        elif cg_model_name in ["MainchainModel"]:
            fix_atom = [True, True, True]
        else:
            fix_atom = [False, False, False]
    else:
        fix_atom = [False, False, False]
    config.structure_module.fix_atom = fix_atom
    if fix_atom[1]:
        config.structure_module.loss_weight["rigid_body"] = 0.0
        config.structure_module.loss_weight["rotation_matrix"] = 0.0
        config.structure_module.loss_weight["FAPE_CA"] = 0.0
    #
    config.structure_module.fiber_edge = []
    if cg_model.n_edge_scalar > 0:
        config.structure_module.fiber_edge.append((0, cg_model.n_edge_scalar))
    if cg_model.n_edge_vector > 0:
        config.structure_module.fiber_edge.append((1, cg_model.n_edge_vector))
    #
    if config.train.get("perturb_pos", 0.0) > 0.0:
        config.train.use_pt = None
    #
    return config


class EmbeddingModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        if config.embedding_dim > 0:
            self.use_embedding = True
            self.layer = nn.Embedding(config.num_embeddings, config.embedding_dim)
        else:
            self.use_embedding = False
            self.register_buffer("one_hot_encoding", torch.eye(config.num_embeddings))

    def forward(self, batch: dgl.DGLGraph):
        if self.use_embedding:
            return self.layer(batch.ndata["residue_type"])
        else:
            return self.one_hot_encoding[batch.ndata["residue_type"]]


class InitializationModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        if config.nonlinearity == "elu":
            nonlinearity = nn.ELU()
        elif config.nonlinearity == "relu":
            nonlinearity = nn.ReLU()
        elif config.nonlinearity == "tanh":
            nonlinearity = nn.Tanh()
        #
        linear_module = []
        if config.norm[0]:
            linear_module.append(NormSE3(Fiber(config.fiber_init), nonlinearity=nonlinearity))
        linear_module.append(LinearSE3(Fiber(config.fiber_init), Fiber(config.fiber_pass)))
        #
        for _ in range(config.num_linear_layers - 1):
            if config.norm[0]:
                linear_module.append(NormSE3(Fiber(config.fiber_pass), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(config.fiber_pass), Fiber(config.fiber_pass)))
        #
        self.linear_module = nn.Sequential(*linear_module)

    def forward(self, feats):
        out = self.linear_module(feats)
        return out


class EdgeModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        if config.nonlinearity == "elu":
            nonlinearity = nn.ELU()
        elif config.nonlinearity == "relu":
            nonlinearity = nn.ReLU()
        elif config.nonlinearity == "tanh":
            nonlinearity = nn.Tanh()
        #
        linear_module = []
        for _ in range(config.num_linear_layers):
            if config.norm[0]:
                linear_module.append(NormSE3(Fiber(config.fiber_edge), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(config.fiber_edge), Fiber(config.fiber_edge)))
        #
        self.linear_module = nn.Sequential(*linear_module)

    def forward(self, feats):
        out = self.linear_module(feats)
        return out


class InteractionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        if config.nonlinearity == "elu":
            nonlinearity = nn.ELU()
        elif config.nonlinearity == "relu":
            nonlinearity = nn.ReLU()
        elif config.nonlinearity == "tanh":
            nonlinearity = nn.Tanh()
        #
        self.graph_module = SE3Transformer(
            num_layers=config.num_graph_layers,
            fiber_in=Fiber(config.fiber_pass),
            fiber_hidden=Fiber(config.fiber_hidden),
            fiber_out=Fiber(config.fiber_pass),
            num_heads=config.num_heads,
            channels_div=config.channels_div,
            fiber_edge=Fiber(config.fiber_edge or {}),
            mid_dim=config.mid_dim,
            norm=config.norm[0],
            use_layer_norm=config.norm[1],
            nonlinearity=nonlinearity,
            low_memory=config.low_memory,
        )

    def forward(self, batch: dgl.DGLGraph, node_feats, edge_feats):
        out = self.graph_module(batch, node_feats=node_feats, edge_feats=edge_feats)
        return out


class StructureModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        #
        self.loss_weight = config.loss_weight
        self.rotation_rep = config.rotation_rep
        self.fix_atom = config.get("fix_atom", [False, False, False])
        #
        if config.nonlinearity == "elu":
            nonlinearity = nn.ELU()
        elif config.nonlinearity == "relu":
            nonlinearity = nn.ReLU()
        elif config.nonlinearity == "tanh":
            nonlinearity = nn.Tanh()
        #
        linear_module = []
        #
        if config.norm[0]:
            linear_module.append(NormSE3(Fiber(config.fiber_struct), nonlinearity=nonlinearity))
        linear_module.append(LinearSE3(Fiber(config.fiber_struct), Fiber(config.fiber_pass)))
        #
        for _ in range(config.num_linear_layers - 2):
            if config.norm[0]:
                linear_module.append(NormSE3(Fiber(config.fiber_pass), nonlinearity=nonlinearity))
            linear_module.append(LinearSE3(Fiber(config.fiber_pass), Fiber(config.fiber_pass)))
        #
        if config.norm[0]:
            linear_module.append(NormSE3(Fiber(config.fiber_pass), nonlinearity=nonlinearity))
        linear_module.append(LinearSE3(Fiber(config.fiber_pass), Fiber(config.fiber_out)))
        #
        self.linear_module = nn.Sequential(*linear_module)

    def forward(self, feats):
        out = self.linear_module(feats)
        return out

    def output_to_opr(self, output, ss_dep=False):
        if self.rotation_rep == "6D":
            return self.output_to_opr_6D(output, ss_dep=ss_dep)
        elif self.rotation_rep.startswith("quat"):
            deg_0, deg_1 = self.rotation_rep.split("_")[1:]
            return self.output_to_opr_quat(output, int(deg_0), int(deg_1), ss_dep=ss_dep)

    def output_to_opr_6D(self, output, ss_dep=False):
        bb0 = output["1"][:, :2]
        v0 = bb0[:, 0]
        v1 = bb0[:, 1]
        e0 = v_norm_safe(v0, index=0)
        u1 = v1 - e0 * inner_product(e0, v1)[:, None]
        e1 = v_norm_safe(u1, index=1)
        e2 = torch.cross(e0, e1)
        rot = torch.stack([e0, e1, e2], dim=1).mT
        #
        t = 0.1 * output["1"][:, 2][..., None, :]
        bb = torch.cat([rot, t], dim=1)
        #
        n_torsion_output = MAX_TORSION * 2
        sc0 = output["0"][:, :n_torsion_output].reshape(-1, MAX_TORSION, 2)
        sc = v_norm_safe(sc0)
        #
        if ss_dep:
            ss0 = output["0"][:, n_torsion_output:, 0]
        else:
            device = output["0"].device
            dtype = output["0"].dtype
            ss0 = torch.zeros((output["0"].size(0), MAX_SS), dtype=dtype, device=device)
            ss0[:, 0] = 1.0

        return bb, sc, bb0, sc0, ss0

    def output_to_opr_quat(self, output, deg_0, deg_1, ss_dep=False):
        n_residue = output["0"].size(0)
        n_torsion_output = MAX_TORSION * 2
        sc0 = output["0"][:, :n_torsion_output].reshape(-1, MAX_TORSION, 2)
        sc = v_norm_safe(sc0)
        #
        device = output["0"].device
        dtype = output["0"].dtype
        #
        if deg_0 == 4 and deg_1 == 0:
            q = output["0"][:, n_torsion_output : n_torsion_output + deg_0, 0]
        elif deg_0 == 3 and deg_1 == 0:
            q = torch.ones((n_residue, 4), dtype=dtype, device=device)
            q[:, 1:] = output["0"][:, n_torsion_output : n_torsion_output + deg_0, 0]
        elif deg_0 == 1 and deg_1 == 1:
            angle = output["0"][:, n_torsion_output, 0]
            axis = v_norm_safe(output["1"][:, 1, :])
            #
            q = torch.zeros((n_residue, 4), dtype=dtype, device=device)
            q[:, 0] = torch.cos(angle / 2.0)
            q[:, 1:] = axis * torch.sin(angle / 2.0)[:, None]
        q = v_norm_safe(q)
        #
        rot = torch.zeros((n_residue, 3, 3), dtype=dtype, device=device)
        rot[:, 0, 0] = q[:, 0] ** 2 + q[:, 1] ** 2 - q[:, 2] ** 2 - q[:, 3] ** 2
        rot[:, 1, 1] = q[:, 0] ** 2 - q[:, 1] ** 2 + q[:, 2] ** 2 - q[:, 3] ** 2
        rot[:, 2, 2] = q[:, 0] ** 2 - q[:, 1] ** 2 - q[:, 2] ** 2 + q[:, 3] ** 2
        rot[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
        rot[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
        rot[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        rot[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
        rot[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
        rot[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
        #
        t = 0.1 * output["1"][:, 0][..., None, :]
        bb = torch.cat([rot, t], dim=1)
        #
        if ss_dep:
            ss0 = output["0"][:, n_torsion_output + deg_0 :, 0]
        else:
            ss0 = torch.zeros((n_residue, MAX_SS), dtype=dtype, device=device)
            ss0[:, 0] = 1.0
        #
        return bb, sc, None, sc0, ss0


class Model(nn.Module):
    def __init__(self, _config, cg_model, compute_loss=False):
        super().__init__()
        #
        self.cg_model = cg_model
        self.ss_dep = _config.globals.ss_dep
        self.compute_loss = compute_loss
        #
        self.embedding_module = EmbeddingModule(_config.embedding_module)
        #
        self.initialization_module = InitializationModule(_config.structure_module)
        self.interaction_module = InteractionModule(_config.structure_module)
        self.structure_module = StructureModule(_config.structure_module)

    def set_constant_tensors(self, device, dtype=DTYPE):
        _RIGID_TRANSFORMS_TENSOR = RIGID_TRANSFORMS_TENSOR.type(dtype)
        _RIGID_GROUPS_TENSOR = RIGID_GROUPS_TENSOR.type(dtype)
        _TORSION_ENERGY_TENSOR = TORSION_ENERGY_TENSOR.type(dtype)
        #
        _RIGID_TRANSFORMS_TENSOR = _RIGID_TRANSFORMS_TENSOR.to(device)
        _RIGID_GROUPS_TENSOR = _RIGID_GROUPS_TENSOR.to(device)
        _TORSION_ENERGY_TENSOR = _TORSION_ENERGY_TENSOR.to(device)
        #
        _RIGID_TRANSFORMS_DEP = RIGID_TRANSFORMS_DEP.to(device)
        _RIGID_GROUPS_DEP = RIGID_GROUPS_DEP.to(device)
        _TORSION_ENERGY_DEP = TORSION_ENERGY_DEP.to(device)
        #
        self.RIGID_OPs = (
            (_RIGID_TRANSFORMS_TENSOR, _RIGID_GROUPS_TENSOR),
            (_RIGID_TRANSFORMS_DEP, _RIGID_GROUPS_DEP),
        )
        #
        self.TORSION_PARs = (_TORSION_ENERGY_TENSOR, _TORSION_ENERGY_DEP)

    def forward(self, batch: dgl.DGLGraph):
        loss = {}
        ret = {}
        #
        # residue_type --> embedding
        embedding = self.embedding_module(batch)
        #
        edge_feats = {"0": batch.edata["edge_feat_0"]}
        node_feats = {
            "0": torch.cat([batch.ndata["node_feat_0"], embedding[..., None]], dim=1),
            "1": batch.ndata["node_feat_1"],
        }
        #
        out0 = self.initialization_module(node_feats)
        #
        # first-pass
        out = self.interaction_module(batch, node_feats=out0, edge_feats=edge_feats)
        for degree, out_d in out0.items():
            out[degree] = out[degree] + out_d
        out0 = {degree: feat.clone() for degree, feat in out.items()}
        #
        out = self.structure_module(out)
        #
        bb, sc, bb0, sc0, ss0 = self.structure_module.output_to_opr(out, ss_dep=self.ss_dep)
        if self.structure_module.fix_atom[0]:
            bb[:, 3] = bb[:, 3] * 0.0
        if self.structure_module.fix_atom[1]:
            bb[:, :3] = batch.ndata["correct_bb"][:, :3]
        ret["bb"] = bb
        ret["sc"] = sc
        ret["bb0"] = bb0
        ret["sc0"] = sc0
        #
        ss = torch.argmax(ss0, dim=-1)
        ret["ss"] = ss
        ret["ss0"] = ss0
        #
        ret["R"], ret["opr_bb"] = build_structure(self.RIGID_OPs, batch, ss, bb, sc=sc)
        if self.structure_module.fix_atom[1]:
            ret["R"][:, ATOM_INDEX_N] = batch.ndata["output_xyz"][:, ATOM_INDEX_N]
            ret["R"][:, ATOM_INDEX_C] = batch.ndata["output_xyz"][:, ATOM_INDEX_C]
        if self.structure_module.fix_atom[2]:
            mask = batch.ndata["pdb_atom_mask"][:, ATOM_INDEX_O] > 0.0
            ret["R"][mask, ATOM_INDEX_O] = batch.ndata["output_xyz"][mask, ATOM_INDEX_O]
        #
        if self.compute_loss or self.training:
            loss["final"] = loss_f(
                batch,
                ret,
                self.structure_module.loss_weight,
                RIGID_OPs=self.RIGID_OPs,
                TORSION_PARs=self.TORSION_PARs,
            )
            metrics = self.calc_metrics(batch, ret)
        else:
            metrics = {}
        #
        return ret, loss, metrics

    def calc_metrics(self, batch, ret):
        R = ret["R"]
        R_ref = batch.ndata["output_xyz"]
        #
        metric_s = {}
        metric_s["clash"] = find_atomic_clash(batch, R, self.RIGID_OPs).mean()
        #
        metric_s["rmsd_CA"] = rmsd_CA(R, R_ref)
        metric_s["rmsd_rigid"] = rmsd_rigid(R, R_ref)
        metric_s["rmsd_all"] = rmsd_all(R, R_ref, batch.ndata["heavy_atom_mask"])
        #
        bonded = rmse_bonded(R, batch.ndata["continuous"])
        metric_s["bond_length"] = bonded[0]
        metric_s["bond_angle"] = bonded[1]
        metric_s["omega_angle"] = bonded[2]
        #
        return metric_s


def combine_operations(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    y = Y.clone()
    Y[..., :3, :] = rotate_matrix(X[..., :3, :], y[..., :3, :])
    Y[..., 3, :] = rotate_vector(X[..., :3, :], y[..., 3, :]) + X[..., 3, :]
    return Y


def build_structure(
    RIGID_OPs,
    batch: dgl.DGLGraph,
    ss: torch.Tensor,
    bb: torch.Tensor,
    sc: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = bb.dtype
    device = bb.device
    residue_type = batch.ndata["residue_type"]
    #
    transforms = RIGID_OPs[0][0][ss, residue_type]
    rigids = RIGID_OPs[0][1][ss, residue_type]
    transforms_dep = RIGID_OPs[1][0][residue_type]
    rigids_dep = RIGID_OPs[1][1][residue_type]
    #
    opr = torch.zeros_like(transforms, device=device)
    #
    # backbone operations
    opr[:, 0, :3] = bb[:, :3]
    opr[:, 0, 3] = bb[:, 3] + batch.ndata["pos"]

    # sidechain operations
    if sc is not None:
        # assume that sc is v_norm_safed
        opr[:, 1:, 0, 0] = 1.0
        opr[:, 1:, 1, 1] = sc[:, :, 0]
        opr[:, 1:, 1, 2] = -sc[:, :, 1]
        opr[:, 1:, 2, 1] = sc[:, :, 1]
        opr[:, 1:, 2, 2] = sc[:, :, 0]
    #
    opr = combine_operations(transforms, opr)
    #
    if sc is not None:
        for i_tor in range(1, MAX_RIGID):
            prev = torch.take_along_dim(
                opr.clone(), transforms_dep[:, i_tor][:, None, None, None], 1
            )
            opr[:, i_tor] = combine_operations(prev[:, 0], opr[:, i_tor])

    opr = torch.take_along_dim(opr, rigids_dep[..., None, None], axis=1)
    R = rotate_vector(opr[:, :, :3], rigids) + opr[:, :, 3]
    return R, opr[:, 0]


def download_ckpt_file(_model_type, ckpt_fn, fix_atom=False):
    if fix_atom:
        model_type = f"{_model_type}-FIX"
    else:
        model_type = _model_type

    try:
        import gdown

        #
        sys.stdout.write(f"Downloading from Google Drive ... {ckpt_fn}\n")
        url_s = {
            "CalphaBasedModel": "1uzsVPB_0t0RDp2P8qJ44LzE3JiVowtTx",
            "ResidueBasedModel": "1KsxfB0B90YQQd1iBzw3buznHIwzN_0sA",
            "CalphaCMModel": "1kLrmeO2F0WXvy0ujq0H4U5drjnuxNy8d",
            "BackboneModel": "17OZDDCiwo7M8egPgRIlMfHujOT-oy0Fz",
            "MainchainModel": "1Q6Xlop_u1hQdLwTlHHdCDxWTC34I8TQg",
            "Martini": "1GiEtLiIOotLrj--7-jJI8aRE10duQoBE",
            "PRIMO": "1FW_QFijewI-z48GC-aDEjHMO_8g1syTH",
            "CalphaBasedModel-FIX": "16FfIW72BDy-RT46kgVoRsGYCcpHOeee1",
            "CalphaCMModel-FIX": "1xdDT-6kkkNiXcg3WxJm1gkw7wDj07Mw9",
            "BackboneModel-FIX": "1uosDHt20KokQBMqyZylO0m8VEONcEuK6",
            "MainchainModel-FIX": "1TaOn42s-3HPlxB4sJ8V21g8rO447F4_v",
        }
        url = url_s[model_type]
        if not ckpt_fn.parent.exists():
            ckpt_fn.parent.mkdir()
        gdown.download(id=url, output=str(ckpt_fn), quiet=True)
    except:
        import requests

        #
        sys.stdout.write(f"Downloading from Zenodo ... {ckpt_fn}\n")
        url = f"https://zenodo.org/record/8015059/files/{ckpt_fn.name}"
        if not ckpt_fn.parent.exists():
            ckpt_fn.parent.mkdir()
        with open(ckpt_fn, "wb") as fout:
            fout.write(requests.get(url).content)
