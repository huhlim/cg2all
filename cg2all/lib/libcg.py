#!/usr/bin/env python

import mdtraj
import numpy as np
import torch
import dgl

from libconfig import DTYPE, EPS
from libpdb import PDB
import numpy_basics
from torch_basics import (
    v_size,
    inner_product,
    torsion_angle,
    one_hot_encoding,
    acos_safe,
)
from residue_constants import (
    MAX_RESIDUE_TYPE,
    AMINO_ACID_s,
    AMINO_ACID_ALT_s,
    residue_s,
    MAX_ATOM,
    ATOM_INDEX_CA,
    ATOM_INDEX_N,
    ATOM_INDEX_C,
    ATOM_INDEX_O,
    read_coarse_grained_topology,
    rigid_groups_dep,
)


class BaseClass(PDB):
    NAME_BEAD = ["CA"]
    WRITE_BEAD = NAME_BEAD
    MAX_BEAD = 1

    n_node_scalar = 17
    n_node_vector = 4
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, is_all=True, **kwarg):
        super().__init__(pdb_fn, dcd_fn, is_all=is_all, **kwarg)
        self.max_bead_type = MAX_RESIDUE_TYPE
        #
        if is_all:
            self.convert_to_cg(**kwarg)
        else:
            self.chain_break_cutoff = kwarg.get("chain_break_cutoff", 1.0)
            self.read_cg()
            self.get_continuity_cg()

    def convert_to_cg(self, **kwarg):
        raise NotImplementedError

    def read_cg(self):
        if len(self.ssbond_s) > 0:
            ssbond_s = np.concatenate(self.ssbond_s, dtype=int)
        else:
            ssbond_s = []
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD, 3))
        self.atom_mask = np.zeros((self.n_residue, MAX_ATOM), dtype=float)
        self.atom_mask_cg = np.zeros((self.n_residue, self.MAX_BEAD), dtype=float)
        self.bfactors_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD), dtype=float)
        #
        for residue in self.top.residues:
            i_res = residue.index
            if residue.name == "HIS":
                residue_name = np.random.choice(["HSD", "HSE"], p=[0.5, 0.5])
                residue.name = residue_name
            else:
                residue_name = AMINO_ACID_ALT_s.get(residue.name, residue.name)
            if residue_name not in AMINO_ACID_s:
                residue_name = "UNK"
            #
            self.residue_name.append(residue_name)
            self.residue_index[i_res] = AMINO_ACID_s.index(residue_name)
            if residue_name == "UNK":
                continue
            ref_res = residue_s[residue_name]
            n_atom = len(ref_res.atom_s)
            self.atom_mask[i_res, :n_atom] = 1.0
            #
            for atom in residue.atoms:
                if atom.name not in self.NAME_BEAD:
                    continue
                i_atm = self.NAME_BEAD.index(atom.name)
                self.R_cg[:, i_res, i_atm] = self.traj.xyz[:, atom.index]
                self.atom_mask_cg[i_res, i_atm] = 1.0
                if not self.is_dcd:
                    self.bfactors_cg[:, i_res, i_atm] = self.traj.bfactors[:, atom.index]
            #
            if i_res in ssbond_s:
                HG1_index = ref_res.atom_s.index("HG1")
                self.atom_mask[i_res, HG1_index] = 0.0

    def write_cg(self, R, pdb_fn=None, dcd_fn=None):
        mask = np.where(self.atom_mask_cg)
        xyz = R[:, mask[0], mask[1], :]
        #
        if pdb_fn is not None:
            traj = mdtraj.Trajectory(xyz[:1], self.top_cg)
            traj.save(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, self.top_cg)
            traj.save(dcd_fn)

    def get_continuity_cg(self):
        self.continuous = np.zeros((2, self.n_residue), dtype=bool)  # prev / next
        #
        # different chains
        same_chain = self.chain_index[1:] == self.chain_index[:-1]
        self.continuous[0, 1:] = same_chain
        self.continuous[1, :-1] = same_chain

        # chain breaks
        dr = self.R_cg[:, 1:, 0] - self.R_cg[:, :-1, 0]
        d = numpy_basics.v_size(dr).mean(axis=0)
        chain_breaks = d > self.chain_break_cutoff
        self.continuous[0, 1:][chain_breaks] = False
        self.continuous[1, :-1][chain_breaks] = False

    @staticmethod
    def get_geometry(_r: torch.Tensor, _mask: torch.Tensor, continuous: torch.Tensor):
        device = _r.device
        r = _r[:, 0]  # BB
        #
        not_defined = continuous == 0.0
        geom_s = {}
        #
        # n_neigh
        n_neigh = torch.zeros(r.shape[0], dtype=DTYPE, device=device)
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
            dr = torch.zeros((r.shape[0] + shift, 3), dtype=DTYPE, device=device)
            b_len = torch.zeros(r.shape[0] + shift, dtype=DTYPE, device=device)
            #
            dr[shift:-shift] = r[:-shift, :] - r[shift:, :]
            b_len[shift:-shift] = v_size(dr[shift:-shift])
            #
            dr = dr / torch.clamp(b_len[:, None], min=EPS)
            # dr[shift:-shift] = dr[shift:-shift] / b_len[shift:-shift, None]
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
        sc = torch.zeros((r.shape[0], 4, 2), device=device)
        for i in range(4):
            sc[i : i + cosine.shape[0], i, 0] = cosine
            sc[i : i + sine.shape[0], i, 1] = sine
        geom_s["dihedral_angle"] = sc
        return geom_s

    @staticmethod
    def geom_to_feature(geom_s, continuous: torch.Tensor, dtype=DTYPE) -> torch.Tensor:
        # features for each residue
        f_in = {"0": [], "1": []}
        #
        # 0d
        f_in["0"].append(geom_s["n_neigh"])  # 1
        f_in["0"].append(torch.as_tensor(continuous.T, dtype=dtype))  # 2
        #
        f_in["0"].append(geom_s["bond_length"][1][0][:, None])  # 4
        f_in["0"].append(geom_s["bond_length"][1][1][:, None])
        f_in["0"].append(geom_s["bond_length"][2][0][:, None])
        f_in["0"].append(geom_s["bond_length"][2][1][:, None])
        #
        f_in["0"].append(geom_s["bond_angle"][0][:, None])  # 2
        f_in["0"].append(geom_s["bond_angle"][1][:, None])
        #
        f_in["0"].append(geom_s["dihedral_angle"].reshape(-1, 8))  # 8
        #
        f_in["0"] = torch.as_tensor(torch.cat(f_in["0"], axis=1), dtype=dtype)  # 17
        #
        # 1d: unit vectors from adjacent residues to the current residue
        f_in["1"].append(geom_s["bond_vector"][1][0][:, None, :])
        f_in["1"].append(geom_s["bond_vector"][1][1][:, None, :])
        f_in["1"].append(geom_s["bond_vector"][2][0][:, None, :])
        f_in["1"].append(geom_s["bond_vector"][2][1][:, None, :])
        if "sc_vector" in geom_s:
            f_in["1"].append(geom_s["sc_vector"])
        f_in["1"] = torch.as_tensor(torch.cat(f_in["1"], axis=1), dtype=dtype)  # 4
        #
        return f_in


class ResidueBasedModel(BaseClass):
    NAME = "ResidueBasedModel"

    def __init__(self, pdb_fn, dcd_fn=None, **kwarg):
        super().__init__(pdb_fn, dcd_fn, **kwarg)

    def convert_to_cg(self, **kwarg):
        self.top_cg = self.top.subset(self.top.select("name CA"))
        #
        mass_weighted_R = self.R * self.atomic_mass[None, ..., None]
        R_cg = mass_weighted_R.sum(axis=-2) / self.atomic_mass.sum(axis=-1)[None, ..., None]
        #
        self.R_cg = R_cg[..., None, :]
        self.atom_mask_cg = self.atom_mask_pdb[:, (ATOM_INDEX_CA,)]
        self.bfactors_cg = self.bfactors[:, :, (ATOM_INDEX_CA,)]


class CalphaBasedModel(BaseClass):
    NAME = "CalphaBasedModel"

    def __init__(self, pdb_fn, dcd_fn=None, **kwarg):
        super().__init__(pdb_fn, dcd_fn, **kwarg)

    def convert_to_cg(self, **kwarg):
        self.top_cg = self.top.subset(self.top.select("name CA"))
        #
        self.R_cg = self.R[:, :, (ATOM_INDEX_CA,), :]
        self.atom_mask_cg = self.atom_mask_pdb[:, (ATOM_INDEX_CA,)]
        self.bfactors_cg = self.bfactors[:, :, (ATOM_INDEX_CA,)]


class CalphaCMModel(BaseClass):
    NAME = "CalphaCMModel"
    NAME_BEAD = ["CA", "CM"]
    MAX_BEAD = 2

    n_node_scalar = 17
    n_node_vector = 5
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, **kwarg):
        super().__init__(pdb_fn, dcd_fn, **kwarg)

    def create_top_cg(self):
        top = mdtraj.Topology()
        #
        serial = 0
        for chain0 in self.top.chains:
            chain = top.add_chain()
            #
            for residue0 in chain0.residues:
                residue = top.add_residue(
                    residue0.name, chain, residue0.resSeq, residue0.segment_id
                )
                #
                atom_s = []
                for atom_name in self.NAME_BEAD:
                    serial += 1
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    atom = top.add_atom(atom_name, element, residue, serial=serial)
                    atom_s.append(atom)
        return top

    def convert_to_cg(self, **kwarg):
        self.top_cg = self.create_top_cg()
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, self.MAX_BEAD), dtype=float)
        #
        self.R_cg[:, :, 0] = self.R[:, :, ATOM_INDEX_CA, :]
        self.atom_mask_cg[:, :] = self.atom_mask_pdb[:, ATOM_INDEX_CA][:, None]
        #
        mass_weighted_R = self.R * self.atomic_mass[None, ..., None]
        R_cm = mass_weighted_R.sum(axis=-2) / self.atomic_mass.sum(axis=-1)[None, ..., None]
        self.R_cg[:, :, 1] = R_cm


class CalphaSCModel(BaseClass):
    NAME = "CalphaSCModel"
    NAME_BEAD = ["CA", "SC"]
    MAX_BEAD = 2

    n_node_scalar = 17
    n_node_vector = 5
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, **kwarg):
        super().__init__(pdb_fn, dcd_fn, **kwarg)

    def create_top_cg(self):
        top = mdtraj.Topology()
        #
        serial = 0
        mask_sc = []
        for chain0 in self.top.chains:
            chain = top.add_chain()
            #
            for residue0 in chain0.residues:
                residue = top.add_residue(
                    residue0.name, chain, residue0.resSeq, residue0.segment_id
                )
                #
                index = AMINO_ACID_s.index(residue0.name)
                mask = rigid_groups_dep[index].copy()
                mask = (mask > 3) & (mask != 8)
                if residue0.name != "GLY":
                    atom_index_CB = residue_s[residue0.name].atom_s.index("CB")
                    mask[atom_index_CB] = True
                mask_sc.append(mask)
                #
                atom_s = []
                for atom_name in self.NAME_BEAD:
                    if residue0.name == "GLY" and atom_name != "CA":
                        continue
                    serial += 1
                    element = mdtraj.core.element.Element.getBySymbol("C")
                    atom = top.add_atom(atom_name, element, residue, serial=serial)
                    atom_s.append(atom)
        return top, np.array(mask_sc, dtype=bool)

    def convert_to_cg(self, **kwarg):
        self.top_cg, mask_sc = self.create_top_cg()
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, self.MAX_BEAD), dtype=float)
        #
        self.R_cg[:, :, 0] = self.R[:, :, ATOM_INDEX_CA, :]
        self.atom_mask_cg[:, 0] = self.atom_mask_pdb[:, ATOM_INDEX_CA]
        #
        for i_res in range(self.n_residue):
            mask = mask_sc[i_res]
            mass = self.atomic_mass[i_res, mask]
            if mass.sum() < EPS:
                continue
            mass_weighted_R = self.R[:, i_res, mask] * mass[None, :, None]
            R_cm = mass_weighted_R.sum(axis=1) / mass.sum()
            self.R_cg[:, i_res, 1] = R_cm
            self.atom_mask_cg[i_res, 1] = 1.0


class SidechainModel(BaseClass):
    NAME = "SidechainModel"
    NAME_BEAD = ["SC"]
    MAX_BEAD = 1

    n_node_scalar = 17
    n_node_vector = 4
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, **kwarg):
        super().__init__(pdb_fn, dcd_fn, **kwarg)

    def create_top_cg(self):
        top = mdtraj.Topology()
        #
        serial = 0
        mask_sc = []
        for chain0 in self.top.chains:
            chain = top.add_chain()
            #
            for residue0 in chain0.residues:
                residue = top.add_residue(
                    residue0.name, chain, residue0.resSeq, residue0.segment_id
                )
                #
                index = AMINO_ACID_s.index(residue0.name)
                mask = rigid_groups_dep[index].copy()
                mask = (mask > 3) & (mask != 8)
                if residue0.name != "GLY":
                    atom_index_CB = residue_s[residue0.name].atom_s.index("CB")
                    mask[atom_index_CB] = True
                mask_sc.append(mask)
                #
                atom_s = []
                for atom_name in self.NAME_BEAD:
                    serial += 1
                    element = mdtraj.core.element.Element.getBySymbol("C")
                    atom = top.add_atom(atom_name, element, residue, serial=serial)
                    atom_s.append(atom)
        return top, np.array(mask_sc, dtype=bool)

    def convert_to_cg(self, **kwarg):
        self.top_cg, mask_sc = self.create_top_cg()
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, self.MAX_BEAD), dtype=float)
        #
        self.R_cg[:, :, 0] = self.R[:, :, ATOM_INDEX_CA, :]
        self.atom_mask_cg[:, 0] = self.atom_mask_pdb[:, ATOM_INDEX_CA]
        #
        for i_res in range(self.n_residue):
            mask = mask_sc[i_res]
            mass = self.atomic_mass[i_res, mask]
            if mass.sum() < EPS:
                self.R_cg[:, i_res, 0] = self.R[:, i_res, ATOM_INDEX_CA, :]
                self.atom_mask_cg[i_res, 0] = self.atom_mask_pdb[i_res, ATOM_INDEX_CA]
            else:
                mass_weighted_R = self.R[:, i_res, mask] * mass[None, :, None]
                R_cm = mass_weighted_R.sum(axis=1) / mass.sum()
                self.R_cg[:, i_res, 0] = R_cm
                self.atom_mask_cg[i_res, 0] = 1.0


class Martini(BaseClass):
    NAME = "Martini"
    NAME_BEAD = ["BB", "SC1", "SC2", "SC3", "SC4"]
    MAX_BEAD = 5

    n_node_scalar = 17
    n_node_vector = 8
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, is_all=True, **kwarg):
        if is_all:
            assert "topology_map" in kwarg
        super().__init__(pdb_fn, dcd_fn, is_all=is_all, **kwarg)

    def create_top_cg(self, topology_map):
        top = self.top.subset(self.top.select("name CA"))
        serial = 0
        for residue in top.residues:
            bb = residue.atom(0)
            for atom in residue.atoms:
                serial += 1
                atom.serial = serial
                atom.name = "BB"
            #
            n_sc = max(topology_map[self.residue_index[residue.index]])
            for i in range(n_sc):
                top.add_atom(f"SC{i+1}", bb.element, residue)
            serial += n_sc
        return top

    def convert_to_cg(self, **kwarg):
        topology_map = kwarg["topology_map"]
        self.top_cg = self.create_top_cg(topology_map)
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, self.MAX_BEAD), dtype=float)
        #
        for i_res in range(self.n_residue):
            index = topology_map[self.residue_index[i_res]]
            #
            mass_weighted_R = self.R[:, i_res] * self.atomic_mass[None, i_res, :, None]
            mass_weighted_R[:, index == -1] = 0.0
            for i_frame in range(self.n_frame):
                np.add.at(self.R_cg[i_frame, i_res], index, mass_weighted_R[i_frame])
                #
            mass_sum = np.zeros(self.MAX_BEAD)
            mass = self.atomic_mass[i_res].copy()
            mass[index == -1] = 0.0
            np.add.at(mass_sum, index, mass)
            #
            self.R_cg[:, i_res] /= np.maximum(EPS, mass_sum[None, :, None])
            self.atom_mask_cg[i_res, mass_sum > EPS] = 1.0


class PRIMO(BaseClass):
    NAME = "PRIMO"
    NAME_BEAD = ["CA", "N", "CO", "SC1", "SC2", "SC3", "SC4", "SC5"]
    WRITE_BEAD = ["N", "CA", "CO", "SC1", "SC2", "SC3", "SC4", "SC5"]
    MAX_BEAD = 8

    n_node_scalar = 17
    n_node_vector = 11
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, is_all=True, **kwarg):
        if is_all:
            assert "topology_map" in kwarg
        super().__init__(pdb_fn, dcd_fn, is_all=is_all, **kwarg)

    def create_top_cg(self, topology_map):
        top = mdtraj.Topology()
        #
        serial = 0
        for chain0 in self.top.chains:
            chain = top.add_chain()
            #
            for residue0 in chain0.residues:
                residue = top.add_residue(
                    residue0.name, chain, residue0.resSeq, residue0.segment_id
                )
                #
                n_bead = len(topology_map[self.residue_index[residue.index]])
                for atom_name in self.WRITE_BEAD[:n_bead]:
                    serial += 1
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    atom = top.add_atom(atom_name, element, residue, serial=serial)
        return top

    def convert_to_cg(self, **kwarg):
        topology_map = kwarg["topology_map"]
        self.top_cg = self.create_top_cg(topology_map)
        #
        self.R_cg = np.zeros((self.n_frame, self.n_residue, self.MAX_BEAD, 3))
        self.atom_mask_cg = np.zeros((self.n_residue, self.MAX_BEAD), dtype=float)
        #
        for i_res in range(self.n_residue):
            index_s = topology_map[self.residue_index[i_res]]
            #
            for k, index in enumerate(index_s):
                is_valid = self.atom_mask[i_res, index].sum() > 0
                self.atom_mask_cg[i_res, k] = is_valid.astype(float)
                if is_valid:
                    self.R_cg[:, i_res, k] = self.R[:, i_res, index].mean(axis=1)

    def write_cg(self, R, pdb_fn=None, dcd_fn=None):
        atom_order = [self.WRITE_BEAD.index(atom_name) for atom_name in self.NAME_BEAD]

        mask = np.where(self.atom_mask_cg[:, atom_order])
        xyz = R[:, :, atom_order][:, mask[0], mask[1]]
        #
        if pdb_fn is not None:
            traj = mdtraj.Trajectory(xyz[:1], self.top_cg)
            traj.save(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, self.top_cg)
            traj.save(dcd_fn)


class BackboneModel(BaseClass):
    NAME = "BackboneModel"
    NAME_BEAD = ["CA", "N", "C"]
    WRITE_BEAD = ["N", "CA", "C"]
    MAX_BEAD = 3

    n_node_scalar = 17
    n_node_vector = 6
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, is_all=True, **kwarg):
        super().__init__(pdb_fn, dcd_fn, is_all=is_all, **kwarg)

    def create_top_cg(self):
        top = mdtraj.Topology()
        #
        serial = 0
        for chain0 in self.top.chains:
            chain = top.add_chain()
            #
            for residue0 in chain0.residues:
                residue = top.add_residue(
                    residue0.name, chain, residue0.resSeq, residue0.segment_id
                )
                #
                for atom_name in self.WRITE_BEAD:
                    serial += 1
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    atom = top.add_atom(atom_name, element, residue, serial=serial)
        return top

    def convert_to_cg(self, **kwarg):
        self.top_cg = self.create_top_cg()
        #
        atom_index = (ATOM_INDEX_CA, ATOM_INDEX_N, ATOM_INDEX_C)
        self.R_cg = self.R[:, :, atom_index, :]
        self.atom_mask_cg = self.atom_mask_pdb[:, atom_index]

    def write_cg(self, R, pdb_fn=None, dcd_fn=None):
        atom_order = [self.WRITE_BEAD.index(atom_name) for atom_name in self.NAME_BEAD]

        mask = np.where(self.atom_mask_cg[:, atom_order])
        xyz = R[:, :, atom_order][:, mask[0], mask[1]]
        #
        if pdb_fn is not None:
            traj = mdtraj.Trajectory(xyz[:1], self.top_cg)
            traj.save(pdb_fn)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, self.top_cg)
            traj.save(dcd_fn)


class MainchainModel(BackboneModel):
    NAME = "MainchainModel"
    NAME_BEAD = ["CA", "N", "C", "O"]
    WRITE_BEAD = ["N", "CA", "C", "O"]
    MAX_BEAD = 4

    n_node_scalar = 17
    n_node_vector = 7
    n_edge_scalar = 3
    n_edge_vector = 0

    def __init__(self, pdb_fn, dcd_fn=None, is_all=True, **kwarg):
        super().__init__(pdb_fn, dcd_fn, is_all=is_all, **kwarg)

    def convert_to_cg(self, **kwarg):
        self.top_cg = self.create_top_cg()
        #
        atom_index = (ATOM_INDEX_CA, ATOM_INDEX_N, ATOM_INDEX_C, ATOM_INDEX_O)
        self.R_cg = self.R[:, :, atom_index, :]
        self.atom_mask_cg = self.atom_mask_pdb[:, atom_index]


def get_residue_center_of_mass(r: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    # r: (n_residue, MAX_ATOM, 3)
    # mass: (n_residue, MAX_ATOM)
    mass_weighted_R = r * mass[..., None]
    cntr = mass_weighted_R.sum(dim=1) / mass.sum(dim=1)[..., None]
    return cntr


def get_backbone_angles(R):
    r_N = R[:, ATOM_INDEX_N]
    r_CA = R[:, ATOM_INDEX_CA]
    r_C = R[:, ATOM_INDEX_C]
    #
    R_phi = torch.stack([r_C[:-1], r_N[1:], r_CA[1:], r_C[1:]], dim=1)
    phi = torsion_angle(R_phi)
    R_psi = torch.stack([r_N[:-1], r_CA[:-1], r_C[:-1], r_N[1:]], dim=1)
    psi = torsion_angle(R_psi)
    #
    tor_s = torch.zeros((R.size(0), 3, 2), device=R.device)
    tor_s[1:, 0, 0] = phi
    tor_s[:-1, 0, 1] = psi
    tor_s[:-1, 1, :] = tor_s[1:, 0]
    tor_s[1:, 2, :] = tor_s[:-1, 0]
    #
    out = torch.cat([torch.cos(tor_s), torch.sin(tor_s)], dim=-1).view(-1, 12)
    return out
