#!/usr/bin/env python

import mdtraj
import numpy as np
from numpy_basics import torsion_angle, internal_to_cartesian
from residue_constants import TER_PATCHes


def patch_TER(
    top: mdtraj.core.topology.Topology,
    residue0: mdtraj.core.topology.Residue,
    xyz0: np.ndarray,
    TER_type: str,
) -> np.ndarray:
    patch = TER_PATCHes.get((TER_type, residue0.name), TER_PATCHes[TER_type, "default"])
    index_s = {
        keyword: [None for _ in patch[keyword]] for keyword in ["delete", "define", "t_ang0_atoms"]
    }
    #
    in_atom_s = [atom.name for atom in residue0.atoms]
    out_atom_s = [atom.name for atom in residue0.atoms]
    if len(patch["delete"]) > 0:
        rindex = out_atom_s.index(patch["delete"][0])
        for name in patch["delete"]:
            out_atom_s.remove(name)
    else:
        rindex = patch["append_index"]
    out_atom_s = out_atom_s[:rindex] + patch["append"] + out_atom_s[rindex:]
    n_atom = len(out_atom_s)
    #
    for atom in residue0.atoms:
        for keyword in index_s:
            if atom.name in patch[keyword]:
                ith_atom = patch[keyword].index(atom.name)
                index_s[keyword][ith_atom] = atom.index
    in_index = [atom.index for atom in residue0.atoms if atom.index not in index_s["delete"]]
    in_mask = [name in out_atom_s for name in in_atom_s]
    out_mask = [name in in_atom_s for name in out_atom_s]
    #
    n_frame = xyz0.shape[0]
    xyz = np.zeros((n_frame, n_atom, 3), dtype=float)
    xyz[:, out_mask] = xyz0[:, in_index]
    #
    if len(index_s["t_ang0_atoms"]) > 0:
        index = index_s["t_ang0_atoms"] + index_s["define"]
        t_ang0 = torsion_angle(xyz0[:, index])
    else:
        t_ang0 = np.zeros(n_frame, dtype=float)
    #
    xyz_define = xyz0[:, index_s["define"]]
    for i, atom_name in enumerate(patch["append"]):
        out_index = out_atom_s.index(atom_name)
        for k in range(n_frame):
            t_ang = patch["t_ang"][i] + t_ang0[k]
            xyz[k, out_index] = internal_to_cartesian(
                *xyz_define[k], patch["b_len"] * 0.1, patch["b_ang"], t_ang
            )
    #
    residue = top.add_residue(residue0.name, top.chain(-1), residue0.resSeq)
    if residue0.segment_id != "":
        residue.segment_id = residue0.segment_id
    for atom_name in out_atom_s:
        element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
        top.add_atom(atom_name, element, residue)
    return xyz, (in_mask, out_mask)


def patch_termini(traj: mdtraj.Trajectory, return_mask=False) -> mdtraj.Trajectory:
    top0 = traj.topology.copy()
    xyz0 = traj.xyz.copy()
    #
    top = mdtraj.Topology()
    xyz = []
    mask = [[], []]
    for chain0 in top0.chains:
        chain = top.add_chain()
        #
        residue_nter = chain0.residue(0)
        xyz_nter, mask_nter = patch_TER(top, residue_nter, xyz0, "NTER")
        xyz.append(xyz_nter)
        if return_mask:
            mask[0].append(np.array(mask_nter[0], dtype=bool))
            mask[1].append(np.array(mask_nter[1], dtype=bool))
        #
        for i_res in range(1, chain0.n_residues - 1):
            residue0 = chain0.residue(i_res)
            index = [atom.index for atom in residue0.atoms]
            xyz.append(xyz0[:, index])
            if return_mask:
                mask[0].append(np.ones(len(index), dtype=bool))
                mask[1].append(np.ones(len(index), dtype=bool))
            #
            residue = top.add_residue(residue0.name, chain, residue0.resSeq)
            if residue0.segment_id != "":
                residue.segment_id = residue0.segment_id
            for atom0 in residue0.atoms:
                top.add_atom(atom0.name, atom0.element, residue)
        #
        residue_cter = chain0.residue(-1)
        xyz_cter, mask_cter = patch_TER(top, residue_cter, xyz0, "CTER")
        xyz.append(xyz_cter)
        if return_mask:
            mask[0].append(np.array(mask_cter[0], dtype=bool))
            mask[1].append(np.array(mask_cter[1], dtype=bool))
    #
    xyz = np.concatenate(xyz, axis=1)
    out = mdtraj.Trajectory(
        xyz=xyz,
        topology=top,
        unitcell_lengths=traj.unitcell_lengths,
        unitcell_angles=traj.unitcell_angles,
    )
    if return_mask:
        mask = (np.concatenate(mask[0]), np.concatenate(mask[1]))
        return out, mask
    else:
        return out


def test():
    traj = mdtraj.load("pdb.processed/1ab1_A.all.pdb", standard_names=False)
    out = patch_termini(traj)
    out.save("pdb.processed/termini.pdb")


if __name__ == "__main__":
    test()
