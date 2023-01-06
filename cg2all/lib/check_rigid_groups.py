#!/usr/bin/env python

import os
import sys
import json
import logging
import pathlib
import functools
import subprocess as sp
import argparse

import tqdm
import numpy as np

import torch
import dgl

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

import libmodel
from libconfig import BASE, DTYPE
from libdata import PDBset, create_trajectory_from_batch
import libcg
from torch_basics import v_norm_safe, inner_product
from libloss import loss_f
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
from libpdb import write_SSBOND
from residue_constants import (
    RIGID_TRANSFORMS_TENSOR,
    RIGID_TRANSFORMS_DEP,
    RIGID_GROUPS_TENSOR,
    RIGID_GROUPS_DEP,
    MAX_TORSION,
    TORSION_ENERGY_TENSOR,
    TORSION_ENERGY_DEP,
    AMINO_ACID_s,
    ATOM_INDEX_CA,
)

import warnings

warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy("file_system")


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))


class PDBset(torch.utils.data.Dataset):
    def __init__(self, cg_model, pdb_fn_s, input_index=1):
        super().__init__()
        self.cg_model = cg_model
        self.pdb_fn_s = pdb_fn_s
        self.dtype = DTYPE
        self.radius = 1.0
        self.input_index = input_index

    def __len__(self):
        return len(self.pdb_fn_s)

    def __getitem__(self, index):
        pdb_fn = self.pdb_fn_s[index]
        #
        cg = self.cg_model(pdb_fn)
        cg.get_structure_information()
        #
        r_cg = torch.as_tensor(cg.R_cg[0], dtype=self.dtype)
        pos = r_cg[cg.atom_mask_cg > 0.0, :]
        #
        data = dgl.radius_graph(pos, self.radius)
        data.ndata["pos"] = pos
        #
        data.ndata["chain_index"] = torch.as_tensor(cg.chain_index, dtype=torch.long)
        data.ndata["residue_type"] = torch.as_tensor(cg.residue_index, dtype=torch.long)
        data.ndata["continuous"] = torch.as_tensor(cg.continuous[0], dtype=self.dtype)
        data.ndata["ss"] = torch.as_tensor(cg.ss[0], dtype=torch.long)
        #
        ssbond_index = torch.full((data.num_nodes(),), -1, dtype=torch.long)
        for cys_i, cys_j in cg.ssbond_s:
            if cys_i < cys_j:  # because of loss_f_atomic_clash
                ssbond_index[cys_j] = cys_i
            else:
                ssbond_index[cys_i] = cys_j
        data.ndata["ssbond_index"] = ssbond_index
        #
        edge_feat = torch.zeros(
            (data.num_edges(), 3), dtype=self.dtype
        )  # bonded / ssbond / space
        for i, cont in enumerate(cg.continuous[0]):
            if cont and data.has_edges_between(i - 1, i):  # i-1 and i is connected
                eid = data.edge_ids(i - 1, i)
                edge_feat[eid, 0] = 1.0
                eid = data.edge_ids(i, i - 1)
                edge_feat[eid, 0] = 1.0
        for cys_i, cys_j in cg.ssbond_s:
            if data.has_edges_between(cys_i, cys_j):
                eid = data.edge_ids(cys_i, cys_j)
                edge_feat[eid, 1] = 1.0
                eid = data.edge_ids(cys_j, cys_i)
                edge_feat[eid, 1] = 1.0
        edge_feat[edge_feat.sum(dim=-1) == 0.0, 2] = 1.0
        data.edata["edge_feat_0"] = edge_feat[..., None]
        #
        data.ndata["atomic_radius"] = torch.as_tensor(
            cg.atomic_radius, dtype=self.dtype
        )
        data.ndata["atomic_mass"] = torch.as_tensor(cg.atomic_mass, dtype=self.dtype)
        data.ndata["input_atom_mask"] = torch.as_tensor(
            cg.atom_mask_cg, dtype=self.dtype
        )
        data.ndata["output_atom_mask"] = torch.as_tensor(cg.atom_mask, dtype=self.dtype)
        data.ndata["pdb_atom_mask"] = torch.as_tensor(
            cg.atom_mask_pdb, dtype=self.dtype
        )
        data.ndata["heavy_atom_mask"] = torch.as_tensor(
            cg.atom_mask_heavy, dtype=self.dtype
        )
        data.ndata["output_xyz"] = torch.as_tensor(cg.R[0], dtype=self.dtype)
        data.ndata["output_xyz_alt"] = torch.as_tensor(cg.R_alt[0], dtype=self.dtype)
        #
        r_cntr = libcg.get_residue_center_of_mass(
            data.ndata["output_xyz"], data.ndata["atomic_mass"]
        )
        v_cntr = r_cntr - data.ndata["output_xyz"][:, ATOM_INDEX_CA]
        data.ndata["v_cntr"] = v_cntr
        #
        data.ndata["correct_bb"] = torch.as_tensor(cg.bb[0], dtype=self.dtype)
        data.ndata["correct_torsion"] = torch.as_tensor(cg.torsion[0], dtype=self.dtype)
        data.ndata["torsion_mask"] = torch.as_tensor(cg.torsion_mask, dtype=self.dtype)
        #
        bb = torch.as_tensor(cg.bb[self.input_index], dtype=self.dtype)
        rot = bb[:, :2]
        tr = 0.1 * (bb[:, 3] - pos)
        data.ndata["input_rot"] = rot
        data.ndata["input_tr"] = tr
        sc = torch.as_tensor(cg.torsion[self.input_index][..., 0], dtype=self.dtype)
        data.ndata["input_sc"] = torch.zeros(
            (sc.size(0), MAX_TORSION, 2), dtype=self.dtype
        )
        data.ndata["input_sc"][..., 0] = torch.cos(sc)
        data.ndata["input_sc"][..., 1] = torch.sin(sc)
        return pdb_fn, data


def run(pdb_fn, batch, RIGID_OPs, TORSION_PARs):
    pdb_fn = pathlib.Path(pdb_fn)
    out_f = pdb_fn.parents[0] / f"rigid.{pdb_fn.stem}.pdb"
    if os.path.exists(out_f):
        return

    ret = {}
    rot0 = batch.ndata["input_rot"].clone()
    tr0 = batch.ndata["input_tr"].clone()
    sc0 = batch.ndata["input_sc"].clone()
    ss = batch.ndata["ss"].clone()
    bb, sc = output_to_opr(rot0, tr0, sc0)
    #
    ret["bb"] = bb
    ret["sc"] = sc
    ret["R"], ret["opr_bb"] = libmodel.build_structure(RIGID_OPs, batch, ss, bb, sc)
    #
    traj_s, ssbond_s = create_trajectory_from_batch(batch, ret["R"], write_native=True)

    traj = traj_s[0]
    traj.save(out_f)
    if len(ssbond_s[0]) > 0:
        write_SSBOND(out_f, traj.top, ssbond_s[0])


def output_to_opr(_rot, _tr, _sc):
    v0 = _rot[:, 0]
    v1 = _rot[:, 1]
    e0 = v_norm_safe(v0, index=0)
    u1 = v1 - e0 * inner_product(e0, v1)[:, None]
    e1 = v_norm_safe(u1, index=1)
    e2 = torch.cross(e0, e1)
    rot = torch.stack([e0, e1, e2], dim=1)
    #
    bb = torch.cat([rot, _tr[:, None, :]], dim=1)
    sc = v_norm_safe(_sc)
    return bb, sc


def main():
    arg = argparse.ArgumentParser(prog="cg2all")
    arg.add_argument("--config", dest="config_json_fn", default=None)
    arg.add_argument("--pdb", dest="pdb_fn_s", nargs="*")
    arg = arg.parse_args()
    if len(arg.pdb_fn_s) == 0:
        return
    #
    if arg.config_json_fn is not None:
        with open(arg.config_json_fn) as fp:
            config = json.load(fp)
    else:
        config = {}
    config["globals.loss_weight.FAPE_all"] = 0.0
    config["globals.loss_weight.v_cntr"] = 0.0
    #
    # configure
    config["cg_model"] = config.get("cg_model", "CalphaBasedModel")
    if config["cg_model"] == "CalphaBasedModel":
        cg_model = CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = ResidueBasedModel
    config = libmodel.set_model_config(config, cg_model)
    #
    device = torch.device("cpu")
    #
    RIGID_OPs = (
        (RIGID_TRANSFORMS_TENSOR.to(device), RIGID_GROUPS_TENSOR.to(device)),
        (RIGID_TRANSFORMS_DEP.to(device), RIGID_GROUPS_DEP.to(device)),
    )
    TORSION_PARs = (TORSION_ENERGY_TENSOR.to(device), TORSION_ENERGY_DEP.to(device))
    #
    pdb_s = PDBset(cg_model, arg.pdb_fn_s, input_index=0)
    for pdb_fn, batch in tqdm.tqdm(pdb_s):
        run(pdb_fn, batch, RIGID_OPs, TORSION_PARs)


if __name__ == "__main__":
    main()
