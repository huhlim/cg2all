#!/usr/bin/env python

import sys
import copy
import random
import numpy as np
import pathlib
import mdtraj
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
    AMINO_ACID_s,
    AMINO_ACID_REV_s,
    ATOM_INDEX_CA,
    residue_s,
    read_martini_topology,
)

torch.multiprocessing.set_sharing_strategy("file_system")


class PDBset(Dataset):
    def __init__(
        self,
        basedir: str,
        pdblist: List[str],
        cg_model,
        topology_file=None,
        radius=1.0,
        self_loop=False,
        augment="",
        min_cg="",
        use_pt=None,
        crop=-1,
        use_md=False,
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
        self.topology_file = topology_file
        self.radius = radius
        self.self_loop = self_loop
        self.augment = augment
        self.min_cg = min_cg
        #
        if use_pt is not None and use_pt.startswith("mixed"):
            mixed_s = {"mixed_0": ["CA_aug_min+FLIP", "min_100", "min_010", "min_001"]}
            self.use_pt = mixed_s[use_pt]
        else:
            self.use_pt = use_pt
        self.crop = crop
        #
        self.use_md = use_md
        self.n_frame = n_frame

    def __len__(self):
        return self.n_pdb * self.n_frame

    def pdb_to_cg(self, *arg, **argv):
        if self.topology_file is None:
            return self.cg_model(*arg, **argv)
        else:
            return self.cg_model(*arg, **argv, martini_top=self.topology_file)

    def __getitem__(self, index):
        pdb_index = index // self.n_frame
        frame_index = index % self.n_frame
        pdb_id = self.pdb_s[pdb_index]
        #
        if self.use_pt is not None:
            if isinstance(self.use_pt, str):
                pt_name = self.use_pt
            else:
                pt_name = random.choice(self.use_pt)
            if self.use_md:
                pt_fn = self.basedir / f"{pdb_id}_{pt_name}.{frame_index}.pt"
            else:
                pt_fn = self.basedir / f"{pdb_id}_{pt_name}.pt"
            #
            if pt_fn.exists():
                data = torch.load(pt_fn)
                # temporary
                if "bfactors" in data.ndata:
                    del data.ndata["bfactors"]
                    torch.save(data, pt_fn)
                if self.crop > 0:
                    return self.get_subgraph(data)
                else:
                    return data
        #
        pdb_fn = self.basedir / f"{pdb_id}.pdb"
        if self.use_md:
            dcd_fn = self.basedir / f"{pdb_id}.dcd"
            cg = self.pdb_to_cg(pdb_fn, dcd_fn=dcd_fn, frame_index=frame_index)
        else:
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
                sys.stderr.write(
                    f"WARNING: augment PDB does NOT exist, {str(pdb_fn_aug)}\n"
                )
        #
        if self.min_cg != "":
            pdb_fn_cg = self.basedir / f"cg/{pdb_id}/{pdb_id}.{self.min_cg}.pdb"
            if pdb_fn_cg.exists():
                cg_min = self.pdb_to_cg(pdb_fn_cg, check_validity=False)
                assert cg_min.R_cg[0].shape == cg.R_cg[0].shape
                r_cg = torch.as_tensor(cg_min.R_cg[0], dtype=self.dtype)
            else:
                sys.stderr.write(
                    f"WARNING: min_cg PDB does NOT exist, {str(pdb_fn_cg)}\n"
                )
        else:
            r_cg = torch.as_tensor(cg.R_cg[0], dtype=self.dtype)
        #
        valid_residue = cg.atom_mask_cg[:, 0] > 0.0
        pos = r_cg[valid_residue, :]
        geom_s = cg.get_geometry(pos, cg.atom_mask_cg, cg.continuous[0])
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
        dcd_fn=None,
        radius=1.0,
        self_loop=False,
        dtype=DTYPE,
    ):
        super().__init__()
        #
        self.pdb_fn = pdb_fn
        self.dcd_fn = dcd_fn
        #
        self.cg_model = cg_model
        #
        self.radius = radius
        self.self_loop = self_loop
        self.dtype = dtype
        #
        if self.dcd_fn is None:
            self.n_frame = 1
        else:
            self.cg = self.pdb_to_cg(self.pdb_fn, dcd_fn=self.dcd_fn)
            self.n_frame = self.cg.n_frame
            # self.n_frame = mdtraj.load(
            #     self.dcd_fn, top=self.pdb_fn, atom_indices=[0]
            # ).n_frames

    def __len__(self):
        return self.n_frame

    def pdb_to_cg(self, *arg, **argv):
        return self.cg_model(*arg, **argv, is_all=False)

    def __getitem__(self, index):
        if self.dcd_fn is None:
            cg = self.pdb_to_cg(self.pdb_fn)
            R_cg = cg.R_cg[0]
        else:
            # cg = self.pdb_to_cg(self.pdb_fn, dcd_fn=self.dcd_fn, frame_index=index)
            cg = self.cg
            R_cg = cg.R_cg[index]
        #
        r_cg = torch.as_tensor(R_cg, dtype=self.dtype)
        #
        valid_residue = cg.atom_mask_cg[:, 0] > 0.0
        pos = r_cg[valid_residue, :]
        geom_s = cg.get_geometry(pos, cg.atom_mask_cg, cg.continuous[0])
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

        return data


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


def create_topology_from_data(
    data: dgl.DGLGraph, write_native: bool = False
) -> mdtraj.Topology:
    top = mdtraj.Topology()
    #
    chain_prev = -1
    seg_no = -1
    n_atom = 0
    atom_index = []
    for i_res in range(data.ndata["residue_type"].size(0)):
        chain_index = data.ndata["chain_index"][i_res]
        resNum = data.ndata["resSeq"][i_res].cpu().detach().item()
        resSeqIns = data.ndata["resSeqIns"][i_res].cpu().detach().item()
        continuous = data.ndata["continuous"][i_res].cpu().detach().item()
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
            for i_atm, atom_name in zip(
                ref_res.output_atom_index, ref_res.output_atom_s
            ):
                # for i_atm, atom_name in enumerate(ref_res.atom_s):
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
        for cys_i, cys_j in enumerate(
            data.ndata["ssbond_index"].cpu().detach().numpy()
        ):
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


def to_pt():
    base_dir = BASE / "pdb.6k"
    pdblist = "set/targets.pdb.6k"
    #
    cg_model = libcg.Martini
    topology_file = read_martini_topology()
    #
    augment = ""
    use_pt = None  # "Martini"
    #
    train_set = PDBset(
        base_dir,
        pdblist,
        cg_model,
        topology_file=topology_file,
        use_pt=use_pt,
        augment=augment,
    )
    #
    train_loader = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=8, shuffle=False, num_workers=16
    )
    for _ in train_loader:
        pass


def test():
    base_dir = BASE / "pdb.6k"
    pdblist = "set/targets.pdb.6k"
    #
    cg_model = libcg.Martini
    topology_file = read_martini_topology()
    #
    augment = ""
    use_pt = None  # "Martini"
    #
    train_set = PDBset(
        base_dir,
        pdblist,
        cg_model,
        topology_file=topology_file,
        use_pt=use_pt,
        augment=augment,
    )
    batch = train_set[0]
    R = torch.zeros((batch.num_nodes(), 24, 3))
    traj_s, ssbond_s = create_trajectory_from_batch(batch, R, write_native=True)
    traj_s[0].save("test.pdb")


if __name__ == "__main__":
    test()
