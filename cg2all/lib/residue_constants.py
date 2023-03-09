import os
import json
import pickle
import numpy as np
import torch
from collections import namedtuple

from libconfig import DATA_HOME
from residue_constants_base import *


def get_rigid_group_by_torsion(ss, residue_name, tor_name, index=-1, sub_index=-1):
    rigid_group = [[], []]  # atom_name, coord
    for X in rigid_groups[ss][residue_name]:
        if X[1] == tor_name:
            if (index < 0 or X[2] == index) and (sub_index < 0 or X[3] == sub_index):
                t_ang = X[5]
                rigid_group[0].append(X[0])
                rigid_group[1].append(X[6])
    rigid_group[1] = np.array(rigid_group[1]) / 10.0  # in nm
    if len(rigid_group[0]) == 0:
        raise ValueError(
            "Cannot find rigid group for"
            f" {residue_name} {tor_name} {index} {sub_index}\n"
        )
    return t_ang, rigid_group[0], rigid_group[1]


def get_rigid_transform_by_torsion(ss, residue_name, tor_name, index, sub_index=-1):
    rigid_transform = None
    for X, Y, tR in rigid_group_transformations[ss][residue_name]:
        if (X[0] == tor_name and X[1] == index) and (
            sub_index < 0 or X[2] == sub_index
        ):
            rigid_transform = (np.array(tR[1]), np.array(tR[0]) / 10.0)
            break
    return Y, rigid_transform


residue_constants_pkl_fn = DATA_HOME / "residue_constants.pkl"
if residue_constants_pkl_fn.exists():
    use_compiled = True
    with open(residue_constants_pkl_fn, "rb") as fp:
        data_dict = pickle.load(fp)
else:
    use_compiled = False
    data_dict = {}

if use_compiled:
    ATOM_NAME_ALT_s = data_dict["ATOM_NAME_ALT_s"]
    #
    torsion_s = data_dict["torsion_s"]
else:
    ATOM_NAME_ALT_s = {}
    with open(DATA_HOME / "rename_atoms.dat") as fp:
        for line in fp:
            x = line.strip().split()
            if x[0] == "*":
                for residue_name in AMINO_ACID_s:
                    ATOM_NAME_ALT_s[(residue_name, x[1])] = x[2]
            else:
                ATOM_NAME_ALT_s[(x[0], x[1])] = x[2]
    data_dict["ATOM_NAME_ALT_s"] = ATOM_NAME_ALT_s
    #
    torsion_s = read_torsion(DATA_HOME / "torsion.dat")
    data_dict["torsion_s"] = torsion_s

if use_compiled:
    residue_s = data_dict["residue_s"]
    #
    radius_s = data_dict["radius_s"]
    par_dihed_s = data_dict["par_dihed_s"]
    #
    rigid_groups = data_dict["rigid_groups"]
    rigid_group_transformations = data_dict["rigid_group_transformations"]
    #
    rigid_transforms_tensor = data_dict["rigid_transforms_tensor"]
    rigid_transforms_dep = data_dict["rigid_transforms_dep"]
    #
    rigid_groups_tensor = data_dict["rigid_groups_tensor"]
    rigid_groups_dep = data_dict["rigid_groups_dep"]
else:
    residue_s = read_CHARMM_rtf(DATA_HOME / "toppar/top_all36_prot.rtf")
    radius_s, par_dihed_s = read_CHARMM_prm(DATA_HOME / "toppar/par_all36m_prot.prm")
    for residue_name, torsion in torsion_s.items():
        residue = residue_s[residue_name]
        for tor in torsion:
            if tor is None:
                continue

    for residue_name, residue in residue_s.items():
        residue.add_torsion_info(torsion_s[residue_name])
        residue.add_radius_info(radius_s)

    rigid_groups = {}
    rigid_group_transformations = {}
    for ss in SECONDARY_STRUCTURE_s:
        if ss == "":
            fn0 = DATA_HOME / f"rigid_groups.json"
            fn1 = DATA_HOME / f"rigid_body_transformation_between_frames.json"
        else:
            fn0 = DATA_HOME / f"rigid_groups_{ss}.json"
            fn1 = DATA_HOME / f"rigid_body_transformation_between_frames_{ss}.json"
        if os.path.exists(fn0) and os.path.exists(fn1):
            with open(fn0) as fp:
                rigid_groups[ss] = json.load(fp)
            with open(fn1) as fp:
                rigid_group_transformations[ss] = json.load(fp)
            for residue_name, residue in residue_s.items():
                if residue_name not in rigid_groups[ss]:
                    continue
                if residue_name not in rigid_group_transformations[ss]:
                    continue
                residue.add_rigid_group_info(
                    rigid_groups[ss][residue_name],
                    rigid_group_transformations[ss][residue_name],
                )

    rigid_transforms_tensor = np.zeros(
        (MAX_SS, MAX_RESIDUE_TYPE, MAX_RIGID, 4, 3), dtype=float
    )
    rigid_transforms_tensor[:, :, :3, :3] = np.eye(3)
    rigid_transforms_dep = np.full((MAX_RESIDUE_TYPE, MAX_RIGID), -1, dtype=int)
    for s, ss in enumerate(SECONDARY_STRUCTURE_s):
        for i, residue_name in enumerate(AMINO_ACID_s):
            if residue_name == "UNK":
                continue
            #
            for tor in torsion_s[residue_name]:
                if tor is None or tor.name == "BB":
                    continue
                if tor.name != "XI":
                    prev, (R, t) = get_rigid_transform_by_torsion(
                        ss, residue_name, tor.name, tor.index
                    )
                else:
                    prev, (R, t) = get_rigid_transform_by_torsion(
                        ss, residue_name, tor.name, tor.index, tor.sub_index
                    )
                rigid_transforms_tensor[s, i, tor.i, :3] = R
                rigid_transforms_tensor[s, i, tor.i, 3] = t
                if prev[0] == "CHI":
                    dep = prev[1] + 2
                elif prev[0] == "BB":
                    dep = 0
                if s == 0:
                    rigid_transforms_dep[i, tor.i] = dep

    rigid_groups_tensor = np.zeros((MAX_SS, MAX_RESIDUE_TYPE, MAX_ATOM, 3), dtype=float)
    rigid_groups_dep = np.full((MAX_RESIDUE_TYPE, MAX_ATOM), -1, dtype=int)
    for s, ss in enumerate(SECONDARY_STRUCTURE_s):
        for i, residue_name in enumerate(AMINO_ACID_s):
            if residue_name == "UNK":
                continue
            #
            residue_atom_s = residue_s[residue_name].atom_s
            for tor in torsion_s[residue_name]:
                if tor is None:
                    continue
                if tor.name != "XI":
                    _, atom_names, coords = get_rigid_group_by_torsion(
                        ss, residue_name, tor.name, tor.index
                    )
                else:
                    _, atom_names, coords = get_rigid_group_by_torsion(
                        ss, residue_name, tor.name, tor.index, tor.sub_index
                    )
                index = tuple([residue_atom_s.index(x) for x in atom_names])
                rigid_groups_tensor[s, i, index] = coords
                if s == 0:
                    rigid_groups_dep[i, index] = tor.i
    #
    data_dict["residue_s"] = residue_s
    #
    data_dict["radius_s"] = radius_s
    data_dict["par_dihed_s"] = par_dihed_s
    #
    data_dict["rigid_groups"] = rigid_groups
    data_dict["rigid_group_transformations"] = rigid_group_transformations
    #
    data_dict["rigid_transforms_tensor"] = rigid_transforms_tensor
    data_dict["rigid_transforms_dep"] = rigid_transforms_dep
    #
    data_dict["rigid_groups_tensor"] = rigid_groups_tensor
    data_dict["rigid_groups_dep"] = rigid_groups_dep


ATOM_INDEX_PRO_CD = residue_s["PRO"].atom_s.index("CD")
ATOM_INDEX_CYS_CB = residue_s["CYS"].atom_s.index("CB")
ATOM_INDEX_CYS_SG = residue_s["CYS"].atom_s.index("SG")

RIGID_TRANSFORMS_TENSOR = torch.as_tensor(rigid_transforms_tensor)
RIGID_TRANSFORMS_DEP = torch.as_tensor(rigid_transforms_dep, dtype=torch.long)
RIGID_TRANSFORMS_DEP[RIGID_TRANSFORMS_DEP == -1] = MAX_RIGID - 1
RIGID_GROUPS_TENSOR = torch.as_tensor(rigid_groups_tensor)
RIGID_GROUPS_DEP = torch.as_tensor(rigid_groups_dep, dtype=torch.long)
RIGID_GROUPS_DEP[RIGID_GROUPS_DEP == -1] = MAX_RIGID - 1

MAX_TORSION_ENERGY = 5
MAX_TORSION_ENERGY_TERM = 17
if use_compiled:
    torsion_energy_tensor = data_dict["torsion_energy_tensor"]
    torsion_energy_dep = data_dict["torsion_energy_dep"]
else:
    torsion_energy_tensor = np.zeros(
        (MAX_SS, MAX_RESIDUE_TYPE, MAX_TORSION_ENERGY, MAX_TORSION_ENERGY_TERM, 5),
        dtype=float,
    )
    torsion_energy_dep = np.tile(
        np.array([0, 1, 2, 3]), [MAX_RESIDUE_TYPE, MAX_TORSION_ENERGY, 1]
    )
    for s, ss in enumerate(SECONDARY_STRUCTURE_s):
        if ss == "":
            fn = DATA_HOME / "torsion_energy_terms.json"
        else:
            fn = DATA_HOME / f"torsion_energy_terms_{ss}.json"
        if os.path.exists(fn):
            with open(fn) as fp:
                X = json.load(fp)
            for i, residue_name in enumerate(AMINO_ACID_s):
                if residue_name == "UNK":
                    continue
                for j in range(len(X[residue_name][0])):
                    if s == 0:
                        torsion_energy_dep[i, j] = np.array(X[residue_name][0][j])
                    for k, term in enumerate(X[residue_name][1][j]):
                        torsion_energy_tensor[s, i, j, k] = np.array(term)
    #
    data_dict["torsion_energy_tensor"] = torsion_energy_tensor
    data_dict["torsion_energy_dep"] = torsion_energy_dep

TORSION_ENERGY_TENSOR = torch.as_tensor(torsion_energy_tensor)
TORSION_ENERGY_DEP = torch.as_tensor(torsion_energy_dep, dtype=torch.long)


def read_martini_topology():
    top_s = {}
    with open(DATA_HOME / "martini.top") as fp:
        for line in fp:
            if line.startswith("RESI"):
                resName = line.strip().split()[1]
                top_s[resName] = []
            elif line.startswith("BEAD"):
                atmName_s = line.strip().split()[2:]
                top_s[resName].append(atmName_s)
    #
    martini_map = np.full((MAX_RESIDUE_TYPE, MAX_ATOM), -1, dtype=int)
    for i_res, resName in enumerate(AMINO_ACID_s):
        if resName not in top_s:
            continue
        #
        for k, bead in enumerate(top_s[resName]):
            for atmName in bead:
                i_atm = residue_s[resName].atom_s.index(atmName)
                martini_map[i_res, i_atm] = k
    return martini_map


def read_primo_topology():
    top_s = {}
    with open(DATA_HOME / "primo.top") as fp:
        for line in fp:
            if line.startswith("RESI"):
                resName = line.strip().split()[1]
                top_s[resName] = []
            elif line.startswith("BEAD"):
                atmName_s = line.strip().split()[2:]
                top_s[resName].append(atmName_s)
    #
    primo_map = []
    for i_res, resName in enumerate(AMINO_ACID_s):
        if resName not in top_s:
            primo_map.append(None)
            continue
        #
        index_s = []
        primo_map.append(index_s)
        for k, bead in enumerate(top_s[resName]):
            index = []
            for atmName in bead:
                i_atm = residue_s[resName].atom_s.index(atmName)
                index.append(i_atm)
            index_s.append(index)
    return primo_map


def update_primo_names(pdb):
    for residue in pdb.top.residues:
        if len(residue.name) == 4 and residue.name[-1] == "2":
            residue.name = residue.name[:3]
        for atom in residue.atoms:
            if atom.name in ["N1", "CA1"]:
                atom.name = atom.name[:-1]


def read_coarse_grained_topology(model):
    if model == "martini":
        return read_martini_topology()
    elif model == "primo":
        return read_primo_topology()


if not use_compiled:
    with open(residue_constants_pkl_fn, "wb") as fout:
        pickle.dump(data_dict, fout)
