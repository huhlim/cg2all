#!/usr/bin/env python

import os
import sys
import mdtraj
import numpy as np
import pathlib
from libpdb import PDB
from residue_constants import residue_s, AMINO_ACID_s
from numpy_basics import *

import tqdm
from tqdm.contrib.concurrent import process_map

N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))


class IC(object):
    def __init__(self, residue):
        self.atom_s = []
        self.ic_s = []
        self.ic_lit = []
        for ic in residue.ic_s:  # bond/angle/torsion
            atom_s = []
            ic_s = []
            ic_lit = []
            for key, value in ic.items():
                atom_s.append(key)
                ic_s.append([[], [], []])
                ic_lit.append(value)

            self.atom_s.append(atom_s)
            self.ic_s.append(ic_s)
            self.ic_lit.append(ic_lit)


def get_atom_index(residue):
    atom_index_s = []
    for ic in residue.ic_s:
        _atom_index_s = [[], []]
        for key in ic:
            atom_index = [[], []]
            for atom in key:
                if atom.startswith("-"):
                    atom_index[0].append(-1)
                    atom_index[1].append(residue.atom_s.index(atom[1:]))
                elif atom.startswith("+"):
                    atom_index[0].append(+1)
                    atom_index[1].append(residue.atom_s.index(atom[1:]))
                else:
                    atom_index[0].append(0)
                    atom_index[1].append(residue.atom_s.index(atom))
            #
            _atom_index_s[0].append(atom_index[0])
            _atom_index_s[1].append(atom_index[1])
        atom_index_s.append(np.array(_atom_index_s, dtype=int))
    return atom_index_s


def run(_pdb_fn):
    data = {}
    #
    pdb = PDB(_pdb_fn)
    #
    for i_aa, resName in enumerate(AMINO_ACID_s):
        if resName == "UNK":
            continue
        #
        atom_index_s = get_atom_index(residue_s[resName])
        data[resName] = []
        #
        _i_res = np.where(pdb.residue_index == i_aa)[0]
        for j, (_res_shift, _i_atm) in enumerate(atom_index_s):
            out_s = []
            for k, (res_shift, i_atm) in enumerate(zip(_res_shift, _i_atm)):
                j_res = _i_res.copy()
                i_res = j_res[:, None] + res_shift[None, :]  # shape=(n_res, n_atoms)
                #
                # exclude terminal dependencies
                is_valid = np.all((i_res >= 0) & (i_res < pdb.n_residue), axis=-1)
                i_res = i_res[is_valid]
                j_res = j_res[is_valid]
                #
                res_prev = np.any(res_shift == -1)
                if res_prev:
                    i_res_prev = i_res[:, res_prev].min(axis=-1)[:, 0]
                    cont_prev = pdb.continuous[i_res_prev + 1] > 0
                    i_res = i_res[cont_prev]
                    j_res = j_res[cont_prev]
                #
                res_next = np.any(res_shift == +1)
                if res_next:
                    i_res_next = i_res[:, res_next].max(axis=-1)[:, 0]
                    cont_next = pdb.continuous[i_res_next] > 0
                    i_res = i_res[cont_next]
                    j_res = j_res[cont_next]
                #
                n_atom = len(i_atm)
                index = np.arange(n_atom)
                R = pdb.R[:, i_res][:, :, index, i_atm, :]
                mask = np.all(pdb.atom_mask_pdb[i_res][:, index, i_atm] > 0.0, axis=-1)
                j_res = j_res[mask]
                R = R[:, mask]
                #
                if n_atom == 2:
                    x = bond_length(R) * 10.0
                elif n_atom == 3:
                    x = bond_angle(R)
                else:
                    x = torsion_angle(R)
                ss = pdb.ss[:, j_res]
                out = [x[ss == s] for s in range(3)]
                out_s.append(out)

            data[resName].append(out_s)

    return data


def report(ic_s):
    for resName, ic in ic_s.items():
        for i, (ic_name, _ic) in enumerate(zip(["BOND", "ANGLE", "TORSION"], ic.ic_s)):
            for j, (atom_s, data) in enumerate(zip(ic.atom_s[i], _ic)):
                lit = ic.ic_lit[i][j]
                #
                wrt = []
                wrt.append(resName)
                wrt.append(f"{ic_name:<8s}")
                wrt.append(" ".join([f"{atom:<4s}" for atom in atom_s]))
                #
                try:
                    data_all = np.concatenate(data[0] + data[1] + data[2])
                    if i == 2:
                        data_all -= lit
                        data_all = (data_all + np.pi) % (2 * np.pi) - np.pi
                        data_all += lit
                        data_all[data_all > np.pi] = 2 * np.pi - data_all[data_all > np.pi]
                except:
                    data_all = np.zeros((0,))
                #
                n_data = [data_all.shape[0]]
                mean = []
                std = []
                if n_data[0] > 0:
                    mean.append(np.mean(data_all))
                    std.append(np.std(data_all))
                else:
                    mean.append(0.0)
                    std.append(0.0)
                #
                for ss in range(3):
                    try:
                        data_ss = np.concatenate(data[ss]).copy()
                        if i == 2:
                            data_ss -= lit
                            data_ss = (data_ss + np.pi) % (2 * np.pi) - np.pi
                            data_ss += lit
                            data_ss[data_ss > np.pi] = 2 * np.pi - data_ss[data_ss > np.pi]
                    except:
                        data_ss = np.zeros((0,))
                    #
                    n_data.append(data_ss.shape[0])
                    if n_data[-1] > 0:
                        mean.append(np.mean(data_ss))
                        std.append(np.std(data_ss))
                    else:
                        mean.append(0.0)
                        std.append(0.0)
                #
                wrt.append(" ".join([f"{n:7d}" for n in n_data]))
                wrt.append(" ".join([f"{x:7.4f}" for x in mean]))
                wrt.append(" ".join([f"{x:7.4f}" for x in std]))
                #
                sys.stdout.write("  ".join(wrt) + "\n")


def main():
    pdblist = pathlib.Path("top8000/targets")
    #
    pdb_home = pdblist.parents[0]
    pdb_fn_s = []
    with open(pdblist) as fp:
        for line in fp:
            if not line.startswith("#"):
                pdb_fn_s.append(pdb_home / f"{line.strip()}.pdb")
    #
    n_proc = min(N_PROC, len(pdb_fn_s))
    if process_map is None:
        with multiprocessing.Pool(n_proc) as pool:
            out_s = pool.map(run, pdb_fn_s)
    else:
        out_s = process_map(run, pdb_fn_s, max_workers=n_proc)
    #
    ic_s = {}
    for resName, residue in residue_s.items():
        ic_s[resName] = IC(residue)
    #
    for out in tqdm.tqdm(out_s):
        for resName, data in out.items():
            for i, ic in enumerate(data):
                for j, x in enumerate(ic):
                    for k, xk in enumerate(x):
                        ic_s[resName].ic_s[i][j][k].append(xk)
    #
    report(ic_s)


if __name__ == "__main__":
    main()
