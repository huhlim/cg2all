"""
This application defines rigid bodies and transformations between frames
"""
import sys
import numpy as np
from typing import List
from collections import namedtuple
from libquat import Quaternion
from numpy_basics import *
from residue_constants import *
import json

from libconfig import DATA_HOME

np.set_printoptions(suppress=True)


def override_ic(residue_s):
    ic_dat_fn = DATA_HOME / "ic.dat"
    ic_s = {}
    with open(ic_dat_fn) as fp:
        for line in fp:
            x = line.strip().split()
            resName = x[0]
            ic_type = {"BOND": 0, "ANGLE": 1, "TORSION": 2}[x[1]]
            atom_s = tuple(x[2 : 4 + ic_type])
            par = np.array(x[4 + ic_type :], dtype=float)
            #
            if resName not in ic_s:
                ic_s[resName] = [{}, {}, {}]
            ic_s[resName][ic_type][atom_s] = par.reshape(3, 4)
    #
    for resName, residue in residue_s.items():
        if resName != "HSE":
            ic = ic_s[resName]
        else:
            ic = ic_s["HSD"]
        #
        for i in range(3):
            for key in residue.ic_s[i]:
                # if i == 2 and key[:3] != ("N", "C", "CA"):
                #     continue
                x = ic[i].get(key, None)
                if x is not None:
                    if i == 2:
                        if x[2][0] > np.deg2rad(15.0):
                            continue
                        delta = np.abs(x[1][0] - residue.ic_s[i][key])
                        delta = np.min([delta, 2 * np.pi - delta])
                        if delta > np.deg2rad(15.0):
                            continue
                    # if i == 2:
                    #     print(resName, key, delta, residue.ic_s[i][key], x[1:, 0])
                    residue.ic_s[i][key] = x[1]


def build_structure_from_ic(residue, ss_index=0):
    def rotate(v, axis=None, angle=0.0):
        # make sure v is normalized
        v = v_norm(v)
        if axis is None:
            axis_tmp = np.cross(v, np.array([0.0, 0.0, 1.0]))
            axis = np.cross(v_norm(axis_tmp), v)
        axis = v_norm(axis)
        q = Quaternion.from_axis_and_angle(axis, angle)
        return q.rotate().dot(v)

    R = {}
    R["-C"] = np.zeros(3)
    #
    # build N, index=0
    atom_name = "N"
    b0 = residue.get_bond_parameter(("-C", atom_name))[ss_index]
    R[atom_name] = R["-C"] + b0 * np.array([1, 0, 0], dtype=float)
    #
    # build CA, index=1
    atom_name = "CA"
    b0 = residue.get_bond_parameter(("N", atom_name))[ss_index]
    a0 = residue.get_angle_parameter(("-C", "N", atom_name))[ss_index]
    v = R["N"] - R["-C"]
    v = rotate(v, angle=np.pi - a0)
    R[atom_name] = R["N"] + b0 * v
    #
    # build the rest
    for atom_s in residue.build_ic:
        atom_name = atom_s[-1]
        b0 = residue.get_bond_parameter(atom_s[-2:])[ss_index]
        a0 = residue.get_angle_parameter(atom_s[-3:])[ss_index]
        t0 = residue.get_torsion_parameter(atom_s)[ss_index]
        r = [R.get(atom_name, None) for atom_name in atom_s[:-1]]
        if True in [np.any(np.isnan(ri)) for ri in r]:
            raise ValueError("Cannot get coordinates for atom", atom_s, r)
        r = np.array(r, dtype=float)
        #
        v21 = v_norm(r[2] - r[1])
        v01 = v_norm(r[0] - r[1])
        axis = v_norm(np.cross(v21, v01))
        v = rotate(v21, axis=axis, angle=np.pi - a0)
        v = rotate(v, axis=v21, angle=t0)
        R[atom_name] = r[-1] + b0 * v
    return R


# define rigid bodies
#  - second atom at the origin
#  - align the rotation axis to the x-axis
#  - last atom on the xy-plane
def get_rigid_groups(residue_s, tor_s, ss_index=0):
    X_axis = np.array([1.0, 0.0, 0.0])
    Y_axis = np.array([0.0, 1.0, 0.0])
    Z_axis = np.array([0.0, 0.0, 1.0])
    #
    rigid_groups = {}
    to_json = {}
    for residue_name, residue in residue_s.items():
        rigid_groups[residue_name] = []
        #
        data = {}
        for tor in tor_s[residue_name]:
            if tor is None:
                continue
            tor_type = tor.name
            atom_s = tor.atom_s

            R = np.array([residue.R.get(atom_name) for atom_name in atom_s], dtype=float)
            t_ang = torsion_angle(R[:4])

            # move the second atom to the origin
            R -= R[2]

            # align the rotation axis to the x-axis
            v = v_norm(R[2] - R[1])
            angle = np.arccos(v.dot(X_axis))
            axis = v_norm(np.cross(v, X_axis))
            q = Quaternion.from_axis_and_angle(axis, angle)
            R = q.rotate().dot(R.T).T

            # last atom on the xy-plane
            v = v_norm(R[3] - R[2])
            n = v_norm(np.cross(v, X_axis))
            angle = np.arccos(n.dot(Z_axis)) * angle_sign(n[1])
            axis = X_axis
            q = Quaternion.from_axis_and_angle(axis, angle)
            R = q.rotate().dot(R.T).T
            if R[3][1] < 0.0:
                q = Quaternion.from_axis_and_angle(axis, np.pi)
                R = q.rotate().dot(R.T).T
            if tor_type == "BB":
                R -= R[1]
            for k, atom_name in enumerate(atom_s):
                if atom_name not in data:
                    data[atom_name] = [tor, t_ang, R[k]]
            #
            # save rigid frames to evaluate rigid body transformation between frames
            rigid_groups[residue_name].append((tor, t_ang, R))

        to_json[residue_name] = []
        for atom_name in residue.atom_s:
            tor, t_ang, R = data[atom_name]
            to_json[residue_name].append(
                [
                    atom_name,
                    tor.name,
                    tor.index,
                    tor.sub_index,
                    tor.index_prev,
                    t_ang,
                    tuple(R.tolist()),
                ]
            )
    ss = ["", "_H", "_E", "_C"][ss_index]
    with open(DATA_HOME / f"rigid_groups{ss}.json", "wt") as fout:
        fout.write(json.dumps(to_json, indent=2))
    return rigid_groups


# define rigid body transformations between frames
# this function evaluates T_{i->j}^{lit} in the Algorithm 24 in the AF2 paper.
#  - OMEGA/PHI/PSI -> BB
#  - CHI1 -> BB
#  - CHI[2-4] -> CHI[1-3]
#  - XI[i] -> CHI[i-1]
def get_rigid_body_transformation_between_frames(rigid_group_s, ss_index=0):
    def get_prev_frame(tor_name, tor_index, rigid_group):
        for tor in rigid_group:
            if tor[0].name == tor_name and tor[0].index == tor_index:
                return tor
        raise Exception(f"{tor_name} {tor_index} not found")

    def get_common_atoms(tor, tor_prev):
        indices = []
        for i, atom_name in enumerate(tor.atom_s[:3]):
            indices.append(tor_prev.atom_s.index(atom_name))
        return tuple(indices)

    to_json = {}
    for residue_name, rigid_group in rigid_group_s.items():
        to_json[residue_name] = []
        for tor, _, R in rigid_group:
            if tor.index_prev < 0:  # backbone do not have a previous frame
                continue
            elif tor.index_prev == 0:  # backbone
                tor_prev, _, R_prev = get_prev_frame("BB", tor.index_prev, rigid_group)
            else:
                tor_prev, _, R_prev = get_prev_frame("CHI", tor.index_prev, rigid_group)
            #
            index = get_common_atoms(tor, tor_prev)
            P = R_prev[index, :].copy()
            Q = R[:3].copy()
            #
            # align the second atoms at the origin
            P0 = P[1].copy()
            Q0 = Q[1].copy()
            P -= P0
            Q -= Q0

            # align the torsion axis
            v21 = v_norm(Q[2] - Q[1])
            torsion_axis = v_norm(P[2] - P[1])
            angle = np.arccos(np.clip(v21.dot(torsion_axis), -1.0, 1.0))
            axis = np.cross(v21, torsion_axis)
            if v_size(axis) > 0.0:
                axis = v_norm(axis)
                q = Quaternion.from_axis_and_angle(axis, angle)
                rotation_1 = q.rotate()
            else:
                rotation_1 = np.eye(3)
            Q = rotation_1.dot(Q.T).T

            # align the dihedral angle
            v01 = v_norm(Q[0] - Q[1])
            u01 = v_norm(P[0] - P[1])
            n0 = v_norm(np.cross(v01, torsion_axis))
            n1 = v_norm(np.cross(u01, torsion_axis))
            angle = np.arccos(n0.dot(n1)) * angle_sign(v01.dot(n1))
            if angle != 0.0:
                q = Quaternion.from_axis_and_angle(torsion_axis, angle)
                rotation_2 = q.rotate()
            else:
                rotation_2 = np.eye(3)
            Q = rotation_2.dot(Q.T).T
            #
            rotation = rotation_2.dot(rotation_1)
            translation = P0 - rotation.dot(Q0)
            #
            R = translate_and_rotate(R, rotation, translation)
            delta = R[:3] - R_prev[index, :]
            delta = np.sqrt(np.mean(np.power(delta, 2).sum(-1)))
            if delta > 1e-5:
                raise ValueError(
                    residue_name,
                    tor,
                    tor_prev,
                    R.round(3),
                    R_prev[index, :].round(3),
                    delta,
                )
            #
            to_json[residue_name].append(
                [
                    (tor.name, tor.index, tor.sub_index),
                    (tor_prev.name, tor_prev.index),
                    (translation.tolist(), rotation.tolist()),
                ]
            )

    ss = ["", "_H", "_E", "_C"][ss_index]
    with open(DATA_HOME / f"rigid_body_transformation_between_frames{ss}.json", "wt") as fout:
        fout.write(json.dumps(to_json, indent=2))


def build_torsion_energy_table(residue_s, par_dihed_s, ss_index=0):
    def get_min_value(p):
        t_ang = np.linspace(-np.pi, np.pi, 36001)
        x = (t_ang[..., None] + p[..., 3]) * p[..., 1] - p[..., 2]
        energy = (p[..., 0] * (1.0 + np.cos(x))).sum(-1)
        return energy.min()

    table_s = {}
    for residue_index, residue_name in enumerate(AMINO_ACID_s):
        if residue_name == "UNK":
            continue
        #
        table_s[residue_name] = [[], []]
        #
        residue = residue_s[residue_name]
        R = residue.R
        #
        axes = {}
        for torsion_atom_index in residue.find_1_N_pair(N=4):
            atom_s = [residue.atom_s[atom_index] for atom_index in torsion_atom_index]
            rigid_group_index = rigid_groups_dep[residue_index, torsion_atom_index]
            if np.all(rigid_group_index < 3):  # is not purely dependent on a residue's atoms
                continue
            rigid_group_unique = np.unique(rigid_group_index, return_counts=True)
            rigid_group_leaf = max(rigid_group_unique[0])
            n_leaf = rigid_group_unique[1][rigid_group_unique[0] == rigid_group_leaf]
            if n_leaf >= 2:
                continue
            if rigid_group_index[0] > rigid_group_index[-1]:
                torsion_atom_index = torsion_atom_index[::-1]
                atom_s = atom_s[::-1]
                rigid_group_index = rigid_group_index[::-1]
            #
            axes_atom = tuple(atom_s[1:3])
            if axes_atom not in axes:
                axes[axes_atom] = []
            #
            type_s = tuple([residue.atom_type_s[atom_index] for atom_index in torsion_atom_index])
            type_rev_s = type_s[::-1]
            type_x = tuple(["X", type_s[1], type_s[2], "X"])
            type_rev_x = type_x[::-1]
            if type_s in par_dihed_s:
                par = par_dihed_s[type_s]
            elif type_rev_s in par_dihed_s:
                par = par_dihed_s[type_rev_s]
            elif type_x in par_dihed_s:
                par = par_dihed_s[type_x]
            elif type_rev_x in par_dihed_s:
                par = par_dihed_s[type_rev_x]
            else:
                raise KeyError(type_s)
            #
            axes[axes_atom].append((atom_s, par))
        #
        for axes_atom, value_s in axes.items():
            tor_axes = None
            for tor in torsion_s[residue_name][3:]:
                if tor.atom_s[1:3] == list(axes_atom):
                    tor_axes = tor.atom_s[:4]
                    break

            r = []
            tor_ref = -1
            for ii, (atom_s, _) in enumerate(value_s):
                if tor_ref < 0 and (tor_axes is None or tor_axes == atom_s):
                    tor_ref = ii
                r.append([R.get(atom) for atom in atom_s])
            r = np.array(r)
            t_ang_s = torsion_angle(r)
            t_ang_s -= t_ang_s[tor_ref]
            #
            _par_s = []
            for t_ang, (_, par) in zip(t_ang_s, value_s):
                for p in par:
                    # p[0] * (1+cos(p[1] * (x + p[-1]) - p[2]))
                    p = p.tolist() + [t_ang]
                    _par_s.append(p)
            #
            torsion_atom_index = [residue.atom_s.index(atom) for atom in value_s[tor_ref][0]]
            #
            energy_min = get_min_value(np.array(_par_s))
            par_s = []
            for ii, par in enumerate(_par_s):
                if ii == 0:
                    par_s.append(par + [energy_min])
                else:
                    par_s.append(par + [0.0])
            #
            table_s[residue_name][0].append(torsion_atom_index)
            table_s[residue_name][1].append(par_s)

    ss = ["", "_H", "_E", "_C"][ss_index]
    with open(DATA_HOME / f"torsion_energy_terms{ss}.json", "wt") as fout:
        fout.write(json.dumps(table_s, indent=2))


def write_residue(pdb_fn, residue, R):
    wrt = [[], [], []]
    for i, (atom, r) in enumerate(R.items()):
        if "-" in atom:
            i_res = 0
            name = atom[1:]
        elif "+" in atom:
            i_res = 2
            name = atom[1:]
        else:
            i_res = 1
            name = atom

        if len(atom) < 4:
            name = f" {name:<3s}"
        if i_res == 1:
            resName = residue.residue_name
        else:
            resName = "ALA"
        line = "ATOM  " + f"{0:5d} {name} {resName} A{i_res:4d}    "
        line += f"{r[0]:8.3f}{r[1]:8.3f}{r[2]:8.3f}\n"
        wrt[i_res].append(line)

    with open(pdb_fn, "wt") as fout:
        fout.writelines(wrt[0])
        fout.writelines(wrt[1])
        fout.writelines(wrt[2])


def main():
    override_ic(residue_s)

    for ss_index in [0, 1, 2, 3]:
        for residue in residue_s.values():
            residue.R = build_structure_from_ic(residue, ss_index=ss_index)
            if not os.path.exists(f"{residue.residue_name}_{ss_index}.pdb"):
                write_residue(f"{residue.residue_name}_{ss_index}.pdb", residue, residue.R)
        #
        rigid_group_s = get_rigid_groups(residue_s, torsion_s, ss_index=ss_index)
        get_rigid_body_transformation_between_frames(rigid_group_s, ss_index=ss_index)
        #
        build_torsion_energy_table(residue_s, par_dihed_s, ss_index=ss_index)


if __name__ == "__main__":
    main()
