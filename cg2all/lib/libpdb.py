#!/usr/bin/env python

# load modules
import sys
import numpy as np
import mdtraj
from string import ascii_uppercase as CHAIN_IDs
from string import ascii_uppercase as INSCODEs
from numpy_basics import *
from residue_constants import *

np.set_printoptions(suppress=True)


class PDB(object):
    def __init__(self, pdb_fn, dcd_fn=None, stride=1, frame_index=None, is_all=True, **kwarg):
        self.pdb_name = str(pdb_fn).split("/")[-1][:-4]

        # read protein
        pdb = mdtraj.load(pdb_fn, standard_names=False)
        load_index = pdb.top.select("protein or (resname HSD or resname HSE or resname MSE)")
        if dcd_fn is None:
            self.is_dcd = False
            traj = pdb.atom_slice(load_index)
            traj.bfactors = pdb.bfactors[:, load_index]
        else:
            self.is_dcd = True
            if frame_index is None:
                traj = mdtraj.load(dcd_fn, top=pdb.top, atom_indices=load_index, stride=stride)
            else:
                traj = mdtraj.load_frame(
                    dcd_fn, frame_index, top=pdb.top, atom_indices=load_index, stride=stride
                )
        self.process(traj, pdb_fn, is_all=is_all, **kwarg)

    def process(self, traj, pdb_fn, is_all=True, check_validity=True, compute_dssp=True):
        self.traj = traj
        self.top = traj.top
        #
        self.n_frame = self.traj.n_frames
        self.n_chain = self.top.n_chains
        self.n_residue = self.top.n_residues
        self.chain_index = np.array([r.chain.index for r in self.top.residues], dtype=int)
        #self.resSeq = np.array([r.resSeq for r in self.top.residues])
        self.resSeq = [r.resSeq for r in self.top.residues]
        self.residue_name = []
        self.residue_index = np.zeros(self.n_residue, dtype=int)
        #
        if compute_dssp and is_all:
            ss = mdtraj.compute_dssp(traj, simplified=True)
            self.ss = np.zeros_like(ss, dtype=int)
            for i, s in enumerate(SECONDARY_STRUCTURE_s):
                self.ss[ss == s] = i
        else:
            self.ss = np.zeros((self.n_frame, self.n_residue), dtype=int)
        #
        self.detect_ssbond(pdb_fn, is_all=is_all)
        if not is_all:
            return
        #
        self.to_atom()
        self.get_continuity()
        #
        if not check_validity:
            return
        #
        is_valid, valid_index = self.check_validity()
        if is_valid:
            return
        else:
            traj_valid = traj.atom_slice(valid_index)
            traj_valid.bfactors = traj.bfactors[:, valid_index]
            self.process(traj_valid, pdb_fn)

    def detect_ssbond(self, pdb_fn, is_all=True):
        chain_s = []
        ssbond_from_pdb = []
        with open(pdb_fn) as fp:
            for line in fp:
                if line.startswith("SSBOND"):
                    cys_0 = (line[15], line[17:22].strip())
                    cys_1 = (line[29], line[31:36].strip())
                    if cys_0 == cys_1:
                        sys.stderr.write(f"WARNING: invalid SSBOND found {pdb_fn}\n")
                        continue
                    ssbond_from_pdb.append((cys_0, cys_1))
                elif line.startswith("ATOM"):
                    chain_id = line[21]
                    if chain_id not in chain_s:
                        chain_s.append(chain_id)
        #
        # find residue.index
        ssbond_s = []
        for cys_s in ssbond_from_pdb:
            residue_index = []
            R = []
            for chain_id, resSeq in cys_s:
                chain_index = chain_s.index(chain_id)
                #
                if resSeq[-1] in INSCODEs:
                    selection = f"chainid {chain_index} and resSeq '{resSeq}'"
                else:
                    selection = f"chainid {chain_index} and resSeq {resSeq}"
                if is_all:
                    selection = f"{selection} and name SG"
                index = self.top.select(selection)
                #
                if is_all:
                    if index.shape[0] == 1:
                        residue_index.append(self.top.atom(index[0]).residue.index)
                        R.append(self.traj.xyz[:, index[0]])
                else:
                    residue_index.append(self.top.atom(index[0]).residue.index)
            residue_index = sorted(residue_index)
            #
            if len(residue_index) == 2 and residue_index not in ssbond_s:
                if is_all:
                    dist = np.linalg.norm(R[1] - R[0], axis=-1)
                    if np.any(dist > 1.0):
                        sys.stderr.write(
                            f"WARNING: invalid SSBOND distance between "
                            f"{cys_s[0][0]} {cys_s[0][1]} and {cys_s[1][0]} {cys_s[1][1]} "
                            f"{np.max(dist).round(3)*10.0:6.2f} {self.pdb_name}\n"
                        )
                    else:
                        ssbond_s.append(residue_index)
                else:
                    ssbond_s.append(residue_index)
        #
        if len(ssbond_s) > 1:
            cys_s, n_cys = np.unique(np.concatenate(ssbond_s), return_counts=True)
            if np.any(n_cys > 1):
                sys.exit("ERROR: some of the Cysteins have multiple SSBOND connections")
        #
        self.ssbond_s = ssbond_s

    def to_atom(self, verbose=False):
        self.R = np.zeros((self.n_frame, self.n_residue, MAX_ATOM, 3))
        self.atom_mask = np.zeros((self.n_residue, MAX_ATOM), dtype=float)
        self.atom_mask_pdb = np.zeros((self.n_residue, MAX_ATOM), dtype=float)
        self.atom_mask_heavy = np.zeros((self.n_residue, MAX_ATOM), dtype=float)
        self.atomic_radius = np.zeros((self.n_residue, MAX_ATOM, 2, 2), dtype=float)
        self.atomic_mass = np.zeros((self.n_residue, MAX_ATOM), dtype=float)
        self.bfactors = np.full((self.n_frame, self.n_residue, MAX_ATOM), 100.0, dtype=float)
        #
        if len(self.ssbond_s) > 0:
            ssbond_s = np.concatenate(self.ssbond_s, dtype=int)
        else:
            ssbond_s = []
        #
        for residue in self.top.residues:
            i_res = residue.index
            if residue.name == "HIS":
                residue_name = get_HIS_state(residue)
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
            #
            for atom in residue.atoms:
                atom_name = ATOM_NAME_ALT_s.get((residue_name, atom.name), atom.name)
                if atom_name in ["OXT", "H2", "H3"]:
                    # at this moment, I ignore N/C-terminal specific atoms
                    continue
                if atom_name.startswith("D"):
                    continue
                if atom_name not in ref_res.atom_s:
                    if verbose:
                        sys.stderr.write(f"Unrecognized atom_name: {residue_name} {atom_name}\n")
                    continue
                i_atm = ref_res.atom_s.index(atom_name)
                self.R[:, i_res, i_atm, :] = self.traj.xyz[:, atom.index, :]
                self.atom_mask_pdb[i_res, i_atm] = 1.0
                if atom_name[0] != "H":
                    self.atom_mask_heavy[i_res, i_atm] = 1.0
                self.atomic_mass[i_res, i_atm] = atom.element.mass
                if not self.is_dcd:
                    self.bfactors[:, i_res, i_atm] = self.traj.bfactors[:, atom.index]
            #
            n_atom = len(ref_res.atom_s)
            self.atomic_radius[i_res, :n_atom] = ref_res.atomic_radius[:n_atom]
            self.atom_mask[i_res, :n_atom] = 1.0
            if i_res in ssbond_s:
                HG1_index = ref_res.atom_s.index("HG1")
                self.atom_mask[i_res, HG1_index] = 0.0
                self.atomic_mass[i_res, HG1_index] = 0.0
        self.bfactors = np.clip(self.bfactors, 0.0, 100.0)

    # get continuity information, whether it has a previous residue
    def get_continuity(self):
        self.continuous = np.zeros((2, self.n_residue), dtype=bool)  # prev / next
        #
        # different chains
        same_chain = self.chain_index[1:] == self.chain_index[:-1]
        self.continuous[0, 1:] = same_chain
        self.continuous[1, :-1] = same_chain

        # chain breaks
        dr = self.R[:, 1:, ATOM_INDEX_N] - self.R[:, :-1, ATOM_INDEX_C]
        d = v_size(dr).mean(axis=0)
        chain_breaks = d > BOND_BREAK
        self.continuous[0, 1:][chain_breaks] = False
        self.continuous[1, :-1][chain_breaks] = False
        #
        has_backbone = np.all(self.atom_mask_pdb[:, :4] > 0.0, axis=-1)
        self.continuous[0, 1:][~has_backbone[:-1]] = False
        self.continuous[1, :-1][~has_backbone[1:]] = False

    def check_validity(self):
        valid = np.any(self.continuous, axis=0)
        #
        if np.all(valid):
            return True, None
        else:
            valid_s = []
            for i_res in np.where(valid > 0.0)[0]:
                valid_s.extend([atom.index for atom in self.top.residue(i_res).atoms])
            valid_s = np.array(valid_s)
            return False, valid_s

    # create a new topology based on the standard residue definitions
    def create_new_topology(self):
        top = mdtraj.Topology()
        #
        i_res = -1
        for chain in self.top.chains:
            top_chain = top.add_chain()
            #
            for residue in chain.residues:
                i_res += 1
                if residue.name == "HIS":
                    residue_name = get_HIS_state(residue)
                else:
                    residue_name = AMINO_ACID_ALT_s.get(residue.name, residue.name)
                if residue_name not in AMINO_ACID_s:
                    residue_name = "UNK"
                if residue_name == "UNK":
                    continue
                #
                has_backbone = np.all(self.atom_mask_pdb[i_res, :4] > 0.0)
                if not has_backbone:
                    self.atom_mask_pdb[i_res, :] = 0.0
                    continue
                #
                top_residue = top.add_residue(residue_name, top_chain, residue.resSeq)
                #
                for i_atm, atom_name in enumerate(residue_s[residue_name].atom_s):
                    if self.atom_mask_pdb[i_res, i_atm] == 0.0:
                        continue
                    element = mdtraj.core.element.Element.getBySymbol(atom_name[0])
                    top.add_atom(atom_name, element, top_residue)
        return top

    # get backbone orientations by placing rigid body atoms (N, CA, C)
    def get_backbone_orientation(self, i_res):
        mask = np.all(self.atom_mask_pdb[i_res, :3])
        if not mask:
            return mask, [np.eye(3), np.zeros(3)]

        # find BB orientations
        opr_s = [[], []]
        for k in range(self.n_frame):
            r = self.R[k, i_res, :3, :]
            frame = rigid_from_3points(r)
            opr_s[0].append(frame[0].T)
            opr_s[1].append(frame[1])
        opr_s = [np.array(opr_s[0]), np.array(opr_s[1])]
        return mask, opr_s

    # get torsion angles for a residue
    def get_torsion_angles(self, i_res):
        residue_name = self.residue_name[i_res]
        ref_res = residue_s[residue_name]
        ss = [SECONDARY_STRUCTURE_s[s] for s in self.ss[:, i_res]]
        #
        torsion_mask = np.zeros(MAX_TORSION, dtype=float)
        torsion_angle_s = np.zeros((self.n_frame, MAX_TORSION, MAX_PERIODIC), dtype=float)
        for tor in torsion_s[self.residue_name[i_res]]:
            if tor is None or tor.name in ["BB"]:
                continue
            #
            t_ang0 = []
            for s in ss:
                t_ang0.append(get_rigid_group_by_torsion(s, residue_name, tor.name, tor.index)[0])
            t_ang0 = np.asarray(t_ang0)
            #
            index = [ref_res.atom_s.index(atom) for atom in tor.atom_s[:4]]
            mask = self.atom_mask_pdb[i_res, index]
            if not np.all(mask):  # if any of the atoms are missing, skip this torsion
                continue
            #
            t_ang = torsion_angle(self.R[:, i_res, index, :]) - t_ang0
            torsion_mask[tor.i - 1] = 1.0
            torsion_angle_s[:, tor.i - 1, :] = t_ang[..., None]
            if tor.periodic > 1:
                for p in range(1, tor.periodic):
                    torsion_angle_s[:, tor.i - 1, p] += p / tor.periodic * 2.0 * np.pi
        return torsion_mask, torsion_angle_s

    def get_R_alt(self, i_res):
        residue_name = self.residue_name[i_res]
        ref_res = residue_s[residue_name]
        #
        R_alt = self.R[:, i_res].copy()
        for tor in torsion_s[residue_name]:
            if tor is None or tor.name != "CHI" or tor.periodic == 1:
                continue
            #
            index_s = []
            for atom_s in tor.atom_alt_s:
                index_s.append([ref_res.atom_s.index(atom) for atom in atom_s[3:]])
            before = index_s[0] + index_s[1]
            after = index_s[1] + index_s[0]
            R_alt[:, before] = self.R[:, i_res, after].copy()
        return R_alt

    def get_structure_information(self):
        # get rigid body operations, backbone_orientations and torsion angles
        self.bb_mask = np.zeros(self.n_residue, dtype=float)
        self.bb = np.zeros((self.n_frame, self.n_residue, 4, 3), dtype=float)
        self.torsion_mask = np.zeros((self.n_residue, MAX_TORSION), dtype=float)
        self.torsion = np.zeros(
            (self.n_frame, self.n_residue, MAX_TORSION, MAX_PERIODIC), dtype=float
        )
        self.R_alt = self.R.copy()
        for i_res in range(self.n_residue):
            mask, opr_s = self.get_backbone_orientation(i_res)
            self.bb_mask[i_res] = mask
            self.bb[:, i_res, :3, :] = opr_s[0]  # rotation matrix
            self.bb[:, i_res, 3, :] = opr_s[1]  # translation vector
            #
            mask, tor_s = self.get_torsion_angles(i_res)
            self.torsion_mask[i_res, :] = mask
            self.torsion[:, i_res, :, :] = tor_s
            #
            self.R_alt[:, i_res] = self.get_R_alt(i_res)

    def write(self, R, pdb_fn, dcd_fn=None):
        top = self.create_new_topology()
        mask = np.where(self.atom_mask_pdb)
        xyz = R[:, mask[0], mask[1], :]
        bfactors = self.bfactors[:, mask[0], mask[1]]
        #
        traj = mdtraj.Trajectory(xyz[:1], top)
        traj.save(pdb_fn, bfactors=bfactors[:1])
        if len(self.ssbond_s) > 0:
            write_SSBOND(pdb_fn, self.top, self.ssbond_s)
        #
        if dcd_fn is not None:
            traj = mdtraj.Trajectory(xyz, top)
            traj.save(dcd_fn)


def get_HIS_state(residue):
    atom_s = [atom.name for atom in residue.atoms]
    if "HD1" in atom_s:
        return "HSD"
    elif "HE2" in atom_s:
        return "HSE"
    else:
        return np.random.choice(["HSD", "HSE"], p=[0.5, 0.5])


def write_SSBOND(pdb_fn, top, ssbond_s):
    SSBOND = "SSBOND  %2d CYS %s %5s   CYS %s %5s\n"
    wrt = []
    for disu_no, ssbond in enumerate(ssbond_s):
        cys_0 = top.residue(ssbond[0])
        cys_1 = top.residue(ssbond[1])
        #
        if isinstance(cys_0.resSeq, int):
            cys_0_resSeq = f"{cys_0.resSeq:4d} "
        else:
            cys_0_resSeq = f"{cys_0.resSeq:>5s}"
        if isinstance(cys_1.resSeq, int):
            cys_1_resSeq = f"{cys_1.resSeq:4d} "
        else:
            cys_1_resSeq = f"{cys_1.resSeq:>5s}"
        #
        wrt.append(
            SSBOND
            % (
                disu_no + 1,
                CHAIN_IDs[cys_0.chain.index],
                cys_0_resSeq,
                CHAIN_IDs[cys_1.chain.index],
                cys_1_resSeq,
            )
        )
    #
    model_s = [[]]
    has_model = False
    with open(pdb_fn) as fp:
        for line in fp:
            if line.startswith("MODEL"):
                has_model = True
                if len(model_s[-1]) > 0:
                    model_s.append([])
                continue
            model_s[-1].append(line)

    with open(pdb_fn, "wt") as fout:
        for i, model in enumerate(model_s):
            if has_model:
                fout.write(f"MODEL   {i:5d}\n")
            fout.writelines(wrt)
            fout.writelines(model)


def generate_structure_from_bb_and_torsion(residue_index, ss, bb, torsion):
    # convert from rigid body operations to coordinates
    n_frame = bb.shape[0]
    n_residue = bb.shape[1]
    R = np.zeros((n_frame, n_residue, MAX_ATOM, 3), dtype=float)
    #
    def rotate_matrix(R, X):
        return np.einsum("...ij,...jk->...ik", R, X)

    def rotate_vector(R, X):
        return np.einsum("...ij,...j", R, X)

    def combine_operations(X, Y):
        Y[..., :3, :] = rotate_matrix(X[..., :3, :], Y[..., :3, :])
        Y[..., 3, :] = rotate_vector(X[..., :3, :], Y[..., 3, :]) + X[..., 3, :]
        return Y

    transforms_dep = rigid_transforms_dep[residue_index]
    rigids_dep = rigid_groups_dep[residue_index]
    #
    for frame in range(n_frame):
        transforms = rigid_transforms_tensor[ss[frame], residue_index]
        rigids = rigid_groups_tensor[ss[frame], residue_index]
        #
        opr = np.zeros_like(transforms)
        opr[:, 0] = bb[frame]
        opr[:, 1:, 0, 0] = 1.0
        sine = np.sin(torsion[frame, ..., 0])
        cosine = np.cos(torsion[frame, ..., 0])
        opr[:, 1:, 1, 1] = cosine
        opr[:, 1:, 1, 2] = -sine
        opr[:, 1:, 2, 1] = sine
        opr[:, 1:, 2, 2] = cosine
        #
        opr = combine_operations(transforms, opr)
        #
        for i_tor in range(1, MAX_RIGID):
            prev = np.take_along_axis(opr, transforms_dep[:, i_tor][:, None, None, None], axis=1)
            opr[:, i_tor] = combine_operations(prev[:, 0], opr[:, i_tor])

        opr = np.take_along_axis(opr, rigids_dep[..., None, None], axis=1)
        R[frame] = rotate_vector(opr[:, :, :3], rigids) + opr[:, :, 3]
    return R


if __name__ == "__main__":
    pdb = PDB("pdb.processed/1ab1_A.pdb")
    pdb.get_structure_information()
    generate_structure_from_bb_and_torsion(pdb.residue_index, pdb.ss, pdb.bb, pdb.torsion)
