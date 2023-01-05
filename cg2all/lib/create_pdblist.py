#!/usr/bin/env python

import os
import glob
import mdtraj


def main():
    pdb_dir = "pdb"
    pdb_fn_s = glob.glob(f"{pdb_dir}/*.pdb")
    pdb_list = []
    for pdb_fn in pdb_fn_s:
        pdb_id = os.path.basename(pdb_fn).split(".")[0]
        #
        pdb = mdtraj.load(pdb_fn)
        protein = pdb.atom_slice(pdb.top.select("protein"))
        calpha = protein.atom_slice(protein.top.select("name CA"))
        #
        n_residues = calpha.top.n_residues
        pdb_list.append((pdb_id, n_residues))
    pdb_list.sort(key=lambda x: x[1], reverse=True)
    #
    with open(f"{pdb_dir}/pdblist", "w") as fp:
        for pdb_id, n_residues in pdb_list:
            fp.write(f"{pdb_id} {n_residues:4d}\n")


if __name__ == "__main__":
    main()
