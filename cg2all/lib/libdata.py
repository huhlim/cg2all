#!/usr/bin/env python

import os
import sys
import copy
import random
import numpy as np
import pathlib
import mdtraj
from mdtraj.formats import PDBTrajectoryFile
from typing import List
from string import ascii_letters

import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset

import dgl

from libconfig import BASE, DTYPE
import libcg
from torch_basics import v_norm, v_size
from residue_constants import (
    MAX_ATOM,
    AMINO_ACID_s,
    AMINO_ACID_REV_s,
    ATOM_INDEX_CA,
    residue_s,
    read_coarse_grained_topology,
)

torch.multiprocessing.set_sharing_strategy("file_system")


class PDBset(Dataset):
    def __init__(
        self,
        basedir: str,
        pdblist: str,
        cg_model,
        topology_map=None,
        radius=1.0,
        self_loop=False,
        augment="",
        min_cg="",
        use_pt=None,
        crop=-1,
        use_md=False,
        perturb_pos=0.0,
        n_frame=1,
        dtype=DTYPE,
    ):
        super().__init__()
        #
        self.dtype = dtype
        self.basedir = pathlib.Path(basedir)
        self.pdb_s = []
        with open(pdblist) as fp:
            for line in fp:
                if line.startswith("#"):
                    continue
                self.pdb_s.append(line.strip())
        #
        self.n_pdb = len(self.pdb_s)
        self.cg_model = cg_model
        self.topology_map = topology_map
        self.radius = radius
        self.self_loop = self_loop
        self.augment = augment
        self.min_cg = min_cg
        self.perturb_pos = perturb_pos
        #
        self.use_pt = use_pt
        self.crop = crop
        #
        self.use_md = use_md
        self.n_frame = n_frame
        if use_md:
            self.use_pt = None
            self.augment = ""

    def __len__(self):
        return self.n_pdb

    def pdb_to_cg(self, *arg, **argv):
        if self.topology_map is None:
            return self.cg_model(*arg, **argv)
        else:
            return self.cg_model(*arg, **argv, topology_map=self.topology_map)

    def __getitem__(self, index):
        pdb_index = index // self.n_frame
        pdb_id = self.pdb_s[pdb_index]
        frame_index = np.random.randint(0, self.n_frame)
        #
        if self.use_pt is not None:
            if self.use_md:
                pt_fn = self.basedir / f"{pdb_id}_{self.use_pt}.{frame_index}.pt"
            else:
                pt_fn = self.basedir / f"{pdb_id}_{self.use_pt}.pt"
            #
            if pt_fn.exists():
                try:
                    data = torch.load(pt_fn)
                    # temporary
                    if "bfactors" in data.ndata:
                        del data.ndata["bfactors"]
                        torch.save(data, pt_fn)
                    if self.crop > 0:
                        return self.get_subgraph(data)
                    else:
                        return data
                except:
                    print(pt_fn, "removed")
                    os.remove(pt_fn)
        #
        if self.use_md:
            pdb_fn = self.basedir / f"{pdb_id}/pdb/sample.{frame_index}.pdb"
        else:
            pdb_fn = self.basedir / f"{pdb_id}.pdb"
        cg = self.pdb_to_cg(pdb_fn)
        cg.get_structure_information()
        #
        if self.augment != "":
            pdb_fn_aug = self.basedir / f"{self.augment}/{pdb_id}.pdb"
            if pdb_fn_aug.exists():
                cg_aug = self.pdb_to_cg(pdb_fn_aug)
                cg_aug.get_structure_information()
                self.augment_torsion(cg, cg_aug)
            else:
                sys.stderr.write(f"WARNING: augment PDB does NOT exist, {str(pdb_fn_aug)}\n")
        #
        if self.min_cg != "":
            pdb_fn_cg = self.basedir / f"cg/{pdb_id}/{pdb_id}.{self.min_cg}.pdb"
            if pdb_fn_cg.exists():
                cg_min = self.pdb_to_cg(pdb_fn_cg, check_validity=False)
                assert cg_min.R_cg[0].shape == cg.R_cg[0].shape
                r_cg = torch.as_tensor(cg_min.R_cg[0], dtype=self.dtype)
            else:
                sys.stderr.write(f"WARNING: min_cg PDB does NOT exist, {str(pdb_fn_cg)}\n")
        else:
            r_cg = torch.as_tensor(cg.R_cg[0], dtype=self.dtype)
        #
        valid_residue = cg.atom_mask_cg[:, 0] > 0.0
        pos = r_cg[valid_residue, :]
        if self.perturb_pos > 0.0:
            dx = torch.randn_like(pos) * self.perturb_pos
            pos = pos + dx
        geom_s = cg.get_geometry(
            pos, cg.atom_mask_cg[valid_residue], cg.continuous[0][valid_residue]
        )
        #
        node_feat = cg.geom_to_feature(geom_s, cg.continuous, dtype=self.dtype)
        data = dgl.radius_graph(pos[:, 0], self.radius, self_loop=self.self_loop)
        data.ndata["pos"] = pos[:, 0]
        data.ndata["node_feat_0"] = node_feat["0"][..., None]  # shape=(N, 16, 1)
        data.ndata["node_feat_1"] = node_feat["1"]  # shape=(N, 4, 3)
        #
        edge_src, edge_dst = data.edges()
        data.edata["rel_pos"] = pos[edge_dst, 0] - pos[edge_src, 0]
        #
        data.ndata["chain_index"] = torch.as_tensor(cg.chain_index, dtype=torch.long)
        resSeq, resSeqIns = resSeq_to_number(cg.resSeq)
        data.ndata["resSeq"] = torch.as_tensor(resSeq, dtype=torch.long)
        data.ndata["resSeqIns"] = torch.as_tensor(resSeqIns, dtype=torch.long)
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
        edge_feat = torch.zeros((data.num_edges(), 3), dtype=self.dtype)  # bonded / ssbond / space
        #
        # bonded
        pair_s = [(i - 1, i) for i, cont in enumerate(cg.continuous[0]) if cont]
        pair_s = torch.as_tensor(pair_s, dtype=torch.long)
        has_edges = data.has_edges_between(pair_s[:, 0], pair_s[:, 1])
        pair_s = pair_s[has_edges]
        eid = data.edge_ids(pair_s[:, 0], pair_s[:, 1])
        edge_feat[eid, 0] = 1.0
        eid = data.edge_ids(pair_s[:, 1], pair_s[:, 0])
        edge_feat[eid, 0] = 1.0
        #
        # ssbond
        if len(cg.ssbond_s) > 0:
            pair_s = torch.as_tensor(cg.ssbond_s, dtype=torch.long)
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
        #
        data.ndata["atomic_radius"] = torch.as_tensor(cg.atomic_radius, dtype=self.dtype)
        data.ndata["atomic_mass"] = torch.as_tensor(cg.atomic_mass, dtype=self.dtype)
        data.ndata["input_atom_mask"] = torch.as_tensor(cg.atom_mask_cg, dtype=self.dtype)
        data.ndata["output_atom_mask"] = torch.as_tensor(cg.atom_mask, dtype=self.dtype)
        data.ndata["pdb_atom_mask"] = torch.as_tensor(cg.atom_mask_pdb, dtype=self.dtype)
        data.ndata["heavy_atom_mask"] = torch.as_tensor(cg.atom_mask_heavy, dtype=self.dtype)
        data.ndata["output_xyz"] = torch.as_tensor(cg.R[0], dtype=self.dtype)
        data.ndata["output_xyz_alt"] = torch.as_tensor(cg.R_alt[0], dtype=self.dtype)
        #
        data.ndata["correct_bb"] = torch.as_tensor(cg.bb[0], dtype=self.dtype)
        data.ndata["correct_torsion"] = torch.as_tensor(cg.torsion[0], dtype=self.dtype)
        data.ndata["torsion_mask"] = torch.as_tensor(cg.torsion_mask, dtype=self.dtype)
        #
        r_cntr = libcg.get_residue_center_of_mass(
            data.ndata["output_xyz"], data.ndata["atomic_mass"]
        )
        v_cntr = r_cntr - data.ndata["output_xyz"][:, ATOM_INDEX_CA]
        data.ndata["v_cntr"] = v_cntr
        #
        if self.use_pt is not None:
            torch.save(data, pt_fn)
        #
        if self.crop > 0:
            return self.get_subgraph(data)
        else:
            return data

    def get_subgraph(self, graph):
        n_residues = graph.num_nodes()
        if n_residues < self.crop:
            nodes = range(0, n_residues)
            sub = graph.subgraph(nodes)
            return sub
        else:
            begin = np.random.randint(low=0, high=n_residues - self.crop + 1)
            nodes = range(begin, begin + self.crop)
        #
        sub = graph.subgraph(nodes)
        #
        for i in range(self.crop):
            cys = sub.ndata["ssbond_index"][i]
            if cys != -1:
                cys = cys - begin
                if cys >= self.crop:
                    cys = -1
                sub.ndata["ssbond_index"][i] = cys
        return sub

    def augment_torsion(self, cg, cg_aug):
        torsion = cg.torsion.copy()
        for i in range(cg.n_residue):
            chain_index = cg.chain_index[i] == cg_aug.chain_index
            resSeq = cg.resSeq[i] == cg_aug.resSeq
            j = np.where(chain_index & resSeq)[0]
            if len(j) == 0:
                continue
            torsion[:, i] = cg_aug.torsion[:, j[0]]
        cg.torsion = np.concatenate([cg.torsion, torsion], axis=-1)


class PredictionData(Dataset):
    def __init__(
        self,
        pdb_fn,
        cg_model,
        topology_map=None,
        dcd_fn=None,
        radius=1.0,
        self_loop=False,
        chain_break_cutoff=1.0,
        is_all=False,
        fix_atom=False,
        batch_size=1,
        dtype=DTYPE,
    ):
        super().__init__()
        #
        self.pdb_fn = pdb_fn
        self.dcd_fn = dcd_fn
        #
        self.cg_model = cg_model
        self.topology_map = topology_map
        #
        self.radius = radius
        self.self_loop = self_loop
        self.dtype = dtype
        self.chain_break_cutoff = chain_break_cutoff
        self.is_all = is_all
        self.fix_atom = fix_atom
        #
        if self.dcd_fn is None:
            self.n_frame0 = 1
            self.n_frame = 1
        else:
            self.cg = self.pdb_to_cg(self.pdb_fn, dcd_fn=self.dcd_fn)
            self.n_frame0 = self.cg.n_frame
            if self.n_frame0 % batch_size == 0:
                self.n_frame = self.n_frame0
            else:
                self.n_frame = self.n_frame0 + (batch_size - self.n_frame0 % batch_size)

    def __len__(self):
        return self.n_frame

    def pdb_to_cg(self, *arg, **argv):
        if self.topology_map is None:
            return self.cg_model(
                chain_break_cutoff=self.chain_break_cutoff,
                is_all=self.is_all,
                *arg,
                **argv,
            )
        else:
            return self.cg_model(
                chain_break_cutoff=self.chain_break_cutoff,
                is_all=self.is_all,
                *arg,
                **argv,
                topology_map=self.topology_map,
            )

    def __getitem__(self, index):
        if index >= self.n_frame0:
            return dgl.graph([])
        #
        if self.dcd_fn is None:
            cg = self.pdb_to_cg(self.pdb_fn)
            R_cg = cg.R_cg[0]
        else:
            cg = self.cg
            R_cg = cg.R_cg[index]
        #
        r_cg = torch.as_tensor(R_cg, dtype=self.dtype)
        #
        valid_residue = cg.atom_mask_cg[:, 0] > 0.0
        pos = r_cg[valid_residue, :]
        geom_s = cg.get_geometry(
            pos, cg.atom_mask_cg[valid_residue], cg.continuous[0][valid_residue]
        )
        #
        node_feat = cg.geom_to_feature(geom_s, cg.continuous, dtype=self.dtype)
        data = dgl.radius_graph(pos[:, 0], self.radius, self_loop=self.self_loop)
        data.ndata["pos"] = pos[:, 0]
        data.ndata["node_feat_0"] = node_feat["0"][..., None]  # shape=(N, 16, 1)
        data.ndata["node_feat_1"] = node_feat["1"]  # shape=(N, 4, 3)
        #
        edge_src, edge_dst = data.edges()
        data.edata["rel_pos"] = pos[edge_dst, 0] - pos[edge_src, 0]
        #
        data.ndata["chain_index"] = torch.as_tensor(cg.chain_index, dtype=torch.long)
        resSeq, resSeqIns = resSeq_to_number(cg.resSeq)
        data.ndata["resSeq"] = torch.as_tensor(resSeq, dtype=torch.long)
        data.ndata["resSeqIns"] = torch.as_tensor(resSeqIns, dtype=torch.long)
        data.ndata["residue_type"] = torch.as_tensor(cg.residue_index, dtype=torch.long)
        data.ndata["continuous"] = torch.as_tensor(cg.continuous[0], dtype=self.dtype)
        data.ndata["output_atom_mask"] = torch.as_tensor(cg.atom_mask, dtype=self.dtype)
        #
        if self.fix_atom:
            if self.cg_model.NAME in ["MainchainModel", "BackboneModel"]:
                atom_order = [cg.WRITE_BEAD.index(atom_name) for atom_name in cg.NAME_BEAD]
                cg.atom_mask_pdb = cg.atom_mask_cg[:, atom_order]
                cg.R = cg.R_cg[:, :, atom_order]
                cg.get_structure_information(bb_only=True)
                data.ndata["correct_bb"] = torch.as_tensor(cg.bb[0], dtype=self.dtype)
                data.ndata["output_xyz"] = torch.as_tensor(cg.R[0], dtype=self.dtype)
                data.ndata["pdb_atom_mask"] = torch.as_tensor(cg.atom_mask_pdb, dtype=self.dtype)
        #
        ssbond_index = torch.full((data.num_nodes(),), -1, dtype=torch.long)
        for cys_i, cys_j in cg.ssbond_s:
            if cys_i < cys_j:  # because of loss_f_atomic_clash
                ssbond_index[cys_j] = cys_i
            else:
                ssbond_index[cys_i] = cys_j
        data.ndata["ssbond_index"] = ssbond_index
        #
        edge_feat = torch.zeros((data.num_edges(), 3), dtype=self.dtype)  # bonded / ssbond / space
        #
        # bonded
        pair_s = [(i - 1, i) for i, cont in enumerate(cg.continuous[0]) if cont]
        pair_s = torch.as_tensor(pair_s, dtype=torch.long)
        has_edges = data.has_edges_between(pair_s[:, 0], pair_s[:, 1])
        pair_s = pair_s[has_edges]
        eid = data.edge_ids(pair_s[:, 0], pair_s[:, 1])
        edge_feat[eid, 0] = 1.0
        eid = data.edge_ids(pair_s[:, 1], pair_s[:, 0])
        edge_feat[eid, 0] = 1.0
        #
        # ssbond
        if len(cg.ssbond_s) > 0:
            pair_s = torch.as_tensor(cg.ssbond_s, dtype=torch.long)
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


class XYZData(Dataset):
    def __init__(
        self,
        xyz,
        residue_type,
        chain_index=None,
        mask=None,
        ssbond_s=[],
        radius=1.0,
        fix_atom=False,
        cg_model=None,
    ):
        # xyz: input CG coordinates, shape=(batch_size, max_residue, max_bead, 3)
        # residue_type: input residue type, shape=(batch_size, max_residue) or (max_residue,)
        # chain_index: input chain index, shape=(batch_size, max_residue) or (max_residue,)
        # mask: input atom_mask_cg, shape=(batch_size, max_residue, max_bead) or (max_residue, max_bead)
        #
        self.device = xyz.device
        self.dtype = xyz.dtype
        self.batch_size = xyz.shape[0]
        self.max_residue = xyz.shape[1]
        self.max_bead = xyz.shape[2]
        self.radius = radius
        self.fix_atom = fix_atom
        self.cg_model = cg_model
        #
        if len(residue_type.shape) == 1:
            residue_type = residue_type.repeat(self.batch_size, 1)
        if chain_index is None:
            chain_index = torch.zeros(
                (self.batch_size, self.max_residue), dtype=torch.long, device=self.device
            )
        elif len(chain_index.shape) == 1:
            chain_index = chain_index.repeat(self.batch_size, 1)
        if mask is None:
            mask = torch.ones(
                (self.batch_size, self.max_residue, self.max_bead),
                dtype=self.dtype,
                device=self.device,
            )
        elif len(mask.shape) == 2:
            mask = mask.repeat(self.batch_size, 1, 1)
        if len(ssbond_s) == 0 or (not isinstance(ssbond_s[0], list)):
            ssbond_s = [ssbond_s for _ in range(self.batch_size)]

        continuous = torch.zeros(
            (self.batch_size, 2, self.max_residue), dtype=self.dtype, device=self.device
        )
        same_chain = chain_index[:, 1:] == chain_index[:, :-1]
        continuous[:, 0, 1:] = same_chain
        continuous[:, 1, :-1] = same_chain
        #
        self.xyz = xyz
        self.residue_type = residue_type
        self.chain_index = chain_index
        self.continuous = continuous
        self.mask = mask
        self.ssbond_s = ssbond_s

    def __len__(self):
        return self.batch_size

    def get_atom_mask(self, index=None):
        def get_atom_mask_index(index):
            atom_mask = np.zeros((self.max_residue, MAX_ATOM), dtype=float)
            for i_res, residue_type in enumerate(self.residue_type[index]):
                residue_name = AMINO_ACID_s[residue_type]
                ref_res = residue_s[residue_name]
                n_atom = len(ref_res.atom_s)
                atom_mask[i_res, :n_atom] = 1.0
                #
                is_ssbond = False
                for ssbond in self.ssbond_s[index]:
                    if i_res in ssbond:
                        is_ssbond = True
                        break
                if is_ssbond:
                    HG1_index = ref_res.atom_s.index("HG1")
                    atom_mask[i_res, HG1_index] = 0.0
            return atom_mask

        #
        if index is None:
            atom_mask = np.array(list(map(get_atom_mask_index, range(self.batch_size))))
        else:
            atom_mask = get_atom_mask_index(index)[None, :]
        #
        return atom_mask

    def __getitem__(self, index):
        mask = self.mask[index]
        valid_residue = mask[:, 0] > 0.0
        pos = self.xyz[index][valid_residue]
        geom_s = libcg.BaseClass.get_geometry(
            pos, mask[valid_residue], self.continuous[index][0, valid_residue]
        )
        node_feat = libcg.BaseClass.geom_to_feature(
            geom_s, self.continuous[index][:, valid_residue], dtype=self.dtype
        )
        data = dgl.radius_graph(pos[:, 0], self.radius, self_loop=False)
        data.ndata["pos"] = pos[:, 0]
        data.ndata["node_feat_0"] = node_feat["0"][..., None]  # shape=(N, 16, 1)
        data.ndata["node_feat_1"] = node_feat["1"]  # shape=(N, 4, 3)
        #
        edge_src, edge_dst = data.edges()
        data.edata["rel_pos"] = pos[edge_dst, 0] - pos[edge_src, 0]
        #
        data.ndata["chain_index"] = self.chain_index[index][valid_residue]
        data.ndata["residue_type"] = self.residue_type[index][valid_residue]
        data.ndata["continuous"] = self.continuous[index][0, valid_residue]
        #
        if self.fix_atom and self.cg_model is not None:
            raise NotImplementedError
        #
        ssbond_index = torch.full((data.num_nodes(),), -1, dtype=torch.long, device=self.device)
        for cys_i, cys_j in self.ssbond_s[index]:
            if cys_i < cys_j:  # because of loss_f_atomic_clash
                ssbond_index[cys_j] = cys_i
            else:
                ssbond_index[cys_i] = cys_j
        data.ndata["ssbond_index"] = ssbond_index
        #
        edge_feat = torch.zeros(
            (data.num_edges(), 3), dtype=self.dtype, device=self.device
        )  # bonded / ssbond / space
        #
        # bonded
        pair_s = [
            (i - 1, i) for i, cont in enumerate(self.continuous[index][0, valid_residue]) if cont
        ]
        pair_s = torch.as_tensor(pair_s, dtype=torch.long, device=self.device)
        has_edges = data.has_edges_between(pair_s[:, 0], pair_s[:, 1])
        pair_s = pair_s[has_edges]
        eid = data.edge_ids(pair_s[:, 0], pair_s[:, 1])
        edge_feat[eid, 0] = 1.0
        eid = data.edge_ids(pair_s[:, 1], pair_s[:, 0])
        edge_feat[eid, 0] = 1.0
        #
        # ssbond
        if len(self.ssbond_s[index]) > 0:
            pair_s = torch.as_tensor(self.ssbond_s[index], dtype=torch.long, device=self.device)
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

    def get_geometry(self):
        raise NotImplementedError
        r = self.xyz[:, :, 0]  # BB
        #
        not_defined = self.continuous[:, 0] == 0.0
        geom_s = {}
        #
        # n_neigh
        n_neigh = torch.zeros(r.shape[:2], dtype=DTYPE, device=self.device)
        graph = dgl.radius_graph(r, 1.0)
        n_neigh = graph.in_degrees(graph.nodes())
        geom_s["n_neigh"] = n_neigh[:, None]

        if _r.shape[1] > 1:
            # BB --> SC
            r_sc = _r[:, 1:]  # SC
            geom_s["sc_vector"] = (r_sc - r[:, None, :]) * _mask[:, 1:, None]

        # bond vectors
        geom_s["bond_length"] = {}
        geom_s["bond_vector"] = {}
        for shift in [1, 2]:
            dr = torch.zeros((r.shape[0] + shift, 3), dtype=DTYPE, device=self.device)
            b_len = torch.zeros(r.shape[0] + shift, dtype=DTYPE, device=self.device)
            #
            dr[shift:-shift] = r[:-shift, :] - r[shift:, :]
            b_len[shift:-shift] = v_size(dr[shift:-shift])
            #
            dr = dr / torch.clamp(b_len[:, None], min=EPS)
            b_len = torch.clamp(b_len, max=1.0)
            #
            for s in range(shift):
                dr[s : -shift + s][not_defined] = 0.0
                b_len[s : -shift + s][not_defined] = 1.0
            #
            geom_s["bond_length"][shift] = (b_len[:-shift], b_len[shift:])
            geom_s["bond_vector"][shift] = (dr[:-shift], -dr[shift:])

        # bond angles
        v1 = geom_s["bond_vector"][1][0]
        v2 = geom_s["bond_vector"][1][1]
        cosine = inner_product(v1, v2)
        sine = 1.0 - cosine**2
        mask = torch.ones_like(cosine)
        mask[not_defined] = 0.0
        mask[-1] = 0.0
        mask[:-1][not_defined[1:]] = 0.0
        cosine = cosine * mask
        sine = sine * mask
        geom_s["bond_angle"] = (cosine, sine)

        # dihedral angles
        R = torch.stack([r[0:-3], r[1:-2], r[2:-1], r[3:]]).reshape(-1, 4, 3)
        t_ang = torsion_angle(R)
        cosine = torch.cos(t_ang)
        sine = torch.sin(t_ang)
        for i in range(3):
            cosine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
            sine[: -(i + 1)][not_defined[i + 1 : -3]] = 0.0
        sc = torch.zeros((r.shape[0], 4, 2), device=self.device)
        for i in range(4):
            sc[i : i + cosine.shape[0], i, 0] = cosine
            sc[i : i + sine.shape[0], i, 1] = sine
        geom_s["dihedral_angle"] = sc
        return geom_s


def resSeq_to_number(resSeq_s: np.ndarray):
    resSeq_number_s = []
    resSeqIns_s = []
    for resSeq in resSeq_s:
        if isinstance(resSeq, int):
            resSeq_number_s.append(resSeq)
            resSeqIns_s.append(0)
        else:
            resSeq_number_s.append(int(resSeq[:-1]))
            resSeqIns_s.append(1 + ascii_letters.index(resSeq[-1]))
    return resSeq_number_s, resSeqIns_s


def create_topology_from_data(data: dgl.DGLGraph, write_native: bool = False) -> mdtraj.Topology:
    top = mdtraj.Topology()
    #
    chain_prev = -1
    seg_no = -1
    n_atom = 0
    atom_index = []
    for i_res in range(data.ndata["residue_type"].size(0)):
        chain_index = data.ndata["chain_index"][i_res]
        continuous = data.ndata["continuous"][i_res].cpu().detach().item()
        #
        resNum = data.ndata["resSeq"][i_res].cpu().detach().item()
        resSeqIns = data.ndata["resSeqIns"][i_res].cpu().detach().item()
        if resSeqIns == 0:
            resSeq = resNum
        else:
            resSeq = f"{resNum}{ascii_letters[resSeqIns-1]}"
        #
        if chain_index != chain_prev:
            chain_prev = chain_index
            top_chain = top.add_chain()
        #
        residue_type_index = int(data.ndata["residue_type"][i_res])
        residue_name = AMINO_ACID_s[residue_type_index]
        residue_name_std = AMINO_ACID_REV_s.get(residue_name, residue_name)
        if residue_name == "UNK":
            continue
        ref_res = residue_s[residue_name]
        top_residue = top.add_residue(residue_name_std, top_chain, resSeq)
        #
        if continuous == 0.0:
            seg_no += 1
        top_residue.segment_id = f"P{seg_no:03d}"
        #
        if write_native:
            mask = data.ndata["pdb_atom_mask"][i_res]
            #
            for i_atm, atom_name in enumerate(ref_res.atom_s):
                if mask[i_atm] > 0.0:
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    top.add_atom(atom_name, element, top_residue)
        else:
            mask = data.ndata["output_atom_mask"][i_res]
            #
            for i_atm, atom_name in zip(ref_res.output_atom_index, ref_res.output_atom_s):
                if mask[i_atm] > 0.0:
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    top.add_atom(atom_name, element, top_residue)
                    atom_index.append(n_atom + i_atm)
            n_atom += top_residue.n_atoms
    return top, atom_index


def create_trajectory_from_batch(
    batch: dgl.DGLGraph,
    R: torch.Tensor = None,
    write_native: bool = False,
) -> List[mdtraj.Trajectory]:
    #
    if R is not None:
        R = R.cpu().detach().numpy()
    #
    write_native = write_native or R is None
    #
    start = 0
    traj_s = []
    ssbond_s = []
    for idx, data in enumerate(dgl.unbatch(batch)):
        top, atom_index = create_topology_from_data(data, write_native=write_native)
        #
        xyz = []
        if write_native:
            mask = data.ndata["pdb_atom_mask"].cpu().detach().numpy()
            xyz.append(data.ndata["output_xyz"].cpu().detach().numpy()[mask > 0.0])
        else:
            mask = data.ndata["output_atom_mask"].cpu().detach().numpy()
        #
        ssbond = []
        for cys_i, cys_j in enumerate(data.ndata["ssbond_index"].cpu().detach().numpy()):
            if cys_j != -1:
                ssbond.append((cys_j, cys_i))
        ssbond_s.append(sorted(ssbond))
        #
        if R is not None:
            end = start + data.num_nodes()
            xyz.append(R[start:end][mask > 0.0])
            start = end
        #
        if write_native:
            xyz = np.array(xyz)
        else:
            xyz = np.array(xyz)[:, atom_index]
        #
        traj = mdtraj.Trajectory(xyz=xyz, topology=top)
        traj_s.append(traj)
    return traj_s, ssbond_s


def standardize_atom_name(out: mdtraj.Trajectory):
    PDBTrajectoryFile._loadNameReplacementTables()
    #
    for residue in out.topology.residues:
        atom_name_mapping = PDBTrajectoryFile._atomNameReplacements.get(residue.name, {})
        #
        for atom in residue.atoms:
            atom.name = atom_name_mapping.get(atom.name, atom.name)


def main():
    topology_map = read_coarse_grained_topology("martini3")
    cg_model = libcg.Martini3

    pdbset = PDBset(
        basedir="pdb.29k",
        pdblist="set/targets.pdb.29k",
        cg_model=cg_model,
        topology_map=topology_map,
        radius=1.0,
        use_pt="Martini3",
    )

    dataloader = dgl.dataloading.GraphDataLoader(pdbset, batch_size=1, num_workers=24)

    import tqdm

    for _ in tqdm.tqdm(dataloader):
        pass


if __name__ == "__main__":
    main()
