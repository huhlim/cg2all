import numpy as np
from libconfig import DATA_HOME

AMINO_ACID_s = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HSD",
    "HSE",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "UNK",
)
AMINO_ACID_one_letter = "ACDEFGHHIKLMNPQRSTVWYX"
PROLINE_INDEX = AMINO_ACID_s.index("PRO")
GLYCINE_INDEX = AMINO_ACID_s.index("GLY")

SECONDARY_STRUCTURE_s = ("", "C", "H", "E")
MAX_SS = len(SECONDARY_STRUCTURE_s)

BACKBONE_ATOM_s = ("N", "CA", "C", "O")
ATOM_INDEX_N = BACKBONE_ATOM_s.index("N")
ATOM_INDEX_CA = BACKBONE_ATOM_s.index("CA")
ATOM_INDEX_C = BACKBONE_ATOM_s.index("C")
ATOM_INDEX_O = BACKBONE_ATOM_s.index("O")

BOND_LENGTH0 = 0.1345
BOND_BREAK = 0.2
BOND_LENGTH_PROLINE_RING = 0.1455
BOND_LENGTH_DISULFIDE = 0.2029

BOND_ANGLE0 = (np.deg2rad(116.5), np.deg2rad(120.0))
TORSION_ANGLE0 = (np.deg2rad(0.0), np.deg2rad(180.0))

TER_PATCHes = {}
TER_PATCHes[("NTER", "default")] = {
    "b_len": 1.040,
    "b_ang": np.deg2rad(109.5),
    "t_ang": np.deg2rad([0.0, 120.0, 240.0]),
    "append": ["HT1", "HT2", "HT3"],
    "delete": ["HN"],
    "define": ["N", "CA", "C"],
    "t_ang0_atoms": ["HN"],
}
TER_PATCHes[("NTER", "PRO")] = {
    "b_len": 1.006,
    "b_ang": np.deg2rad(109.5),
    "t_ang": np.deg2rad([0.0, -120.0]),
    "append": ["HN1", "HN2"],
    "delete": [],
    "append_index": 1,
    "define": ["N", "CA", "C"],
    "t_ang0_atoms": [],
}
TER_PATCHes[("CTER", "default")] = {
    "b_len": 1.260,
    "b_ang": np.deg2rad(118.0),
    "t_ang": np.deg2rad([0.0, 180.0]),
    "append": ["OT1", "OT2"],
    "delete": ["O"],
    "define": ["C", "CA", "N"],
    "t_ang0_atoms": ["O"],
}

AMINO_ACID_ALT_s = {"HIS": "HSD", "MSE": "MET"}
AMINO_ACID_REV_s = {"HSD": "HIS", "HSE": "HIS"}

MAX_RESIDUE_TYPE = len(AMINO_ACID_s)
MAX_ATOM = 24
MAX_TORSION_CHI = 4
MAX_TORSION_XI = 2
MAX_TORSION = MAX_TORSION_CHI + MAX_TORSION_XI + 2  # =8 ; 2 for bb/phi/psi
MAX_RIGID = MAX_TORSION + 1
MAX_PERIODIC = 3
MAX_PERIODIC_HEAVY = 2


class Residue(object):
    def __init__(self, residue_name: str) -> None:
        self.residue_name = residue_name
        self.residue_index = AMINO_ACID_s.index(residue_name)

        self.atom_s = [atom for atom in BACKBONE_ATOM_s]
        self.atom_type_s = [None for atom in BACKBONE_ATOM_s]
        self.output_atom_s = []
        self.output_atom_index = []
        self.ic_s = [{}, {}, {}]  # bond, angle, torsion
        self.build_ic = []

        self.torsion_bb_atom = []
        self.torsion_chi_atom = []
        self.torsion_xi_atom = []
        self.torsion_chi_mask = np.zeros(MAX_TORSION_CHI, dtype=float)
        self.torsion_xi_mask = np.zeros(MAX_TORSION_XI, dtype=float)
        self.torsion_chi_periodic = np.zeros((MAX_TORSION_CHI, 3), dtype=float)
        self.torsion_xi_periodic = np.zeros((MAX_TORSION_XI, 3), dtype=float)

        self.atomic_radius = np.zeros((MAX_ATOM, 2, 2), dtype=float)  # (normal/1-4, epsilon/r_min)
        self.atomic_charge = np.zeros(MAX_ATOM, dtype=float)

        self.bonded_pair_s = {2: []}

    def __str__(self):
        return self.residue_name

    def __eq__(self, other):
        if isinstance(other, Residue):
            return self.residue_name == other.residue_name
        else:
            return self.residue_name == other

    def append_atom(self, atom_name, atom_type, atom_charge=0.0):
        if atom_name not in BACKBONE_ATOM_s:
            self.atom_s.append(atom_name)
            self.atom_type_s.append(atom_type)
        else:
            i = BACKBONE_ATOM_s.index(atom_name)
            self.atom_type_s[i] = atom_type
        #
        self.output_atom_s.append(atom_name)
        self.output_atom_index.append(self.atom_s.index(atom_name))
        self.atomic_charge[self.atom_s.index(atom_name)] = atom_charge

    def append_bond(self, pair):
        if not (pair[0][0] == "+" or pair[1][0] == "+"):
            i = self.atom_s.index(pair[0])
            j = self.atom_s.index(pair[1])
            self.bonded_pair_s[2].append(sorted([i, j]))

    def append_ic(self, atom_s, param_s):
        is_improper = atom_s[2][0] == "*"
        param_s[1:4] = np.deg2rad(param_s[1:4])
        if is_improper:
            atom_s[2] = atom_s[2][1:]
            self.ic_s[0][(atom_s[0], atom_s[2])] = param_s[0]
            self.ic_s[0][(atom_s[2], atom_s[0])] = param_s[0]
            self.ic_s[1][(atom_s[0], atom_s[2], atom_s[1])] = param_s[1]
            self.ic_s[1][(atom_s[1], atom_s[2], atom_s[0])] = param_s[1]
        else:
            self.ic_s[0][(atom_s[0], atom_s[1])] = param_s[0]
            self.ic_s[0][(atom_s[1], atom_s[0])] = param_s[0]
            self.ic_s[1][(atom_s[0], atom_s[1], atom_s[2])] = param_s[1]
            self.ic_s[1][(atom_s[2], atom_s[1], atom_s[0])] = param_s[1]
        self.ic_s[2][(atom_s[0], atom_s[1], atom_s[2], atom_s[3])] = param_s[2]
        self.ic_s[1][(atom_s[1], atom_s[2], atom_s[3])] = param_s[3]
        self.ic_s[1][(atom_s[3], atom_s[2], atom_s[1])] = param_s[3]
        self.ic_s[0][(atom_s[2], atom_s[3])] = param_s[4]
        self.ic_s[0][(atom_s[3], atom_s[2])] = param_s[4]
        self.build_ic.append(tuple(atom_s))

    def get_bond_parameter(self, atom_name_s) -> float:
        b0 = self.ic_s[0].get(atom_name_s, None)
        if b0 is None:
            raise ValueError("bond parameter not found", atom_name_s)
        if isinstance(b0, float):
            b0 = np.ones(4) * b0
        return b0

    def get_angle_parameter(self, atom_name_s) -> float:
        a0 = self.ic_s[1].get(atom_name_s, None)
        if a0 is None:
            raise ValueError("angle parameter not found", atom_name_s)
        if isinstance(a0, float):
            a0 = np.ones(4) * a0
        return a0

    def get_torsion_parameter(self, atom_name_s) -> float:
        t0 = self.ic_s[2].get(atom_name_s, None)
        if t0 is None:
            raise ValueError("torsion parameter not found", atom_name_s)
        if isinstance(t0, float):
            t0 = np.ones(4) * t0
        return t0

    def add_torsion_info(self, tor_s):
        for tor in tor_s:
            if tor is None:
                continue
            if tor.name == "CHI":
                index = tor.index - 1
                periodic = tor.periodic - 1
                self.torsion_chi_atom.append(tor.atom_s[:4])
                self.torsion_chi_mask[index] = 1.0
                self.torsion_chi_periodic[index, periodic] = 1.0
            elif tor.name == "XI":
                index = tor.i - 7
                periodic = tor.periodic - 1
                self.torsion_xi_atom.append(tor.atom_s[:4])
                self.torsion_xi_mask[index] = 1.0
                self.torsion_xi_periodic[index, periodic] = 1.0
            else:
                self.torsion_bb_atom.append(tor.atom_s[:4])

    def add_rigid_group_info(self, rigid_group, transform):
        self.rigid_group = []
        for info in rigid_group:
            self.rigid_group.append(info[:-1] + [np.array(info[-1])])
        self.transform = []
        for info in transform:
            self.transform.append(info[:-1] + [(np.array(info[-1][0]), np.array(info[-1][1]))])

    def add_radius_info(self, radius_s):
        for i, atom_type in enumerate(self.atom_type_s):
            self.atomic_radius[i] = radius_s[atom_type]

    def find_1_N_pair(self, N: int):
        if N in self.bonded_pair_s:
            return self.bonded_pair_s[N]
        #
        pair_prev_s = self.find_1_N_pair(N - 1)
        #
        pair_s = []
        for prev in pair_prev_s:
            for bond in self.bonded_pair_s[2]:
                pair = None
                if prev[-1] == bond[0] and prev[-2] != bond[1]:
                    pair = prev + [bond[1]]
                elif prev[-1] == bond[1] and prev[-2] != bond[0]:
                    pair = prev + [bond[0]]
                elif prev[0] == bond[1] and prev[1] != bond[0]:
                    pair = [bond[0]] + prev
                elif prev[0] == bond[0] and prev[1] != bond[1]:
                    pair = [bond[1]] + prev
                #
                if pair is not None:
                    if pair not in pair_s and (pair[::-1] not in pair_s):
                        pair_s.append(pair)

        self.bonded_pair_s[N] = pair_s
        return pair_s


class Torsion(object):
    def __init__(self, i, name, index, sub_index, index_prev, atom_s, periodic=1):
        self.i = i
        self.name = name
        self.index = index
        self.index_prev = index_prev
        self.sub_index = sub_index
        self.atom_s = [atom.replace("*", "") for atom in atom_s]
        self.periodic = periodic
        if self.periodic > 1:  # always tip atoms
            self.atom_alt_s = self.generate_atom_alt_s(atom_s, periodic)
        else:
            self.atom_alt_s = [atom_s]

    def __repr__(self):
        return f"{self.name} {self.index} {'-'.join(self.atom_s)}"

    def generate_atom_alt_s(self, atom_s, periodic):
        alt_s = [[] for _ in range(periodic)]
        for atom in atom_s[:3]:
            for i in range(periodic):
                alt_s[i].append(atom)
        i = 0
        for atom in atom_s[3:]:
            if "*" in atom:
                atom_name = atom.replace("*", "")
                for k in range(periodic):
                    alt_s[k].append(atom_name)
            else:
                k = i % periodic
                alt_s[k].append(atom)
                i += 1
        return alt_s


# read TORSION.dat file
def read_torsion(fn):
    backbone_rigid = ["N", "CA", "C"]
    tor_s = {}
    with open(fn) as fp:
        for line in fp:
            if line.startswith("RESI"):
                residue_name = line.strip().split()[1]
                xi_index = -1
                tor_s[residue_name] = []
                if residue_name not in ["GLY"]:
                    atom_s = backbone_rigid + ["N", "CB", "HA"]
                    tor_s[residue_name].append(Torsion(0, "BB", 0, -1, -1, atom_s, 1))
                else:
                    atom_s = backbone_rigid + ["N", "HA1", "HA2"]
                    tor_s[residue_name].append(Torsion(0, "BB", 0, -1, -1, atom_s, 1))
                if residue_name not in ["PRO"]:
                    atom_s = backbone_rigid[::-1] + ["HN"]
                    tor_s[residue_name].append(Torsion(1, "PHI", 1, -1, 0, atom_s, 1))
                else:
                    tor_s[residue_name].append(None)
                atom_s = backbone_rigid + ["O"]
                tor_s[residue_name].append(Torsion(2, "PSI", 1, -1, 0, atom_s, 1))
            elif line.startswith("CHI"):
                x = line.strip().split()
                tor_no = int(x[1])
                periodic = int(x[2])
                atom_s = x[3:]
                tor_s[residue_name].append(
                    Torsion(tor_no + 2, "CHI", tor_no, -1, tor_no - 1, atom_s, periodic)
                )
            elif line.startswith("XI"):
                xi_index += 1
                x = line.strip().split()
                tor_no, sub_index = x[1].split(".")
                periodic = int(x[2])
                tor_no = int(tor_no)
                sub_index = int(sub_index)
                atom_s = x[3:]
                tor_s[residue_name].append(
                    Torsion(
                        xi_index + 7,
                        "XI",
                        tor_no,
                        sub_index,
                        tor_no - 1,
                        atom_s,
                        periodic,
                    )
                )
            elif line.startswith("#"):
                continue
    return tor_s


def read_CHARMM_rtf(fn):
    residue_s = {}
    with open(fn) as fp:
        read = False
        for line in fp:
            if line.startswith("RESI"):
                residue_name = line.split()[1]
                if residue_name not in AMINO_ACID_s:
                    continue
                residue = Residue(residue_name)
                residue_s[residue_name] = residue
                read = True
            elif not read:
                continue
            elif line.strip() == "":
                read = False
            elif line.startswith("ATOM"):
                x = line.strip().split()
                atom_name = x[1]
                atom_type = x[2]
                atom_charge = float(x[3])
                residue.append_atom(atom_name, atom_type, atom_charge=atom_charge)
            elif line.startswith("BOND") or line.startswith("DOUBLE"):
                x = line.strip().split()[1:]
                for i in range(len(x) // 2):
                    residue.append_bond([x[2 * i], x[2 * i + 1]])

            elif line.startswith("IC"):
                x = line.strip().split()
                atom_s = x[1:5]
                param_s = np.array(x[5:10], dtype=float)
                residue.append_ic(atom_s, param_s)
    return residue_s


# read CHARMM parameter file
def read_CHARMM_prm(fn):
    with open(fn) as fp:
        content_s = []
        for line in fp:
            line = line.strip()
            if line.startswith("!"):
                continue
            if len(content_s) > 0 and content_s[-1].endswith("-"):
                content_s[-1] += line
            else:
                content_s.append(line)
    #
    par_dihedrals = {}
    read = False
    for line in content_s:
        if line.startswith("DIHEDRALS"):
            read = True
            continue
        if line.startswith("END") or line.startswith("IMPROPER"):
            read = False
        if not read:
            continue
        x = line.strip().split()
        if len(x) == 0:
            continue
        atom_type_s = tuple(x[:4])
        par = np.array([x[4], x[5], np.deg2rad(float(x[6]))], dtype=float)
        if atom_type_s not in par_dihedrals:
            par_dihedrals[atom_type_s] = []
        par_dihedrals[atom_type_s].append(par)
    #
    radius_s = {}
    read = False
    for line in content_s:
        if line.startswith("NONBONDED"):
            read = True
            continue
        if line.strip() == "" or line.startswith("END") or line.startswith("NBFIX"):
            read = False
        if not read:
            continue
        x = line.strip().split()
        if "!" in x:
            x = x[: x.index("!")]
        epsilon = float(x[2])
        r_min = float(x[3]) * 0.1
        if len(x) >= 6:
            epsilon_14 = float(x[5])
            r_min_14 = float(x[6]) * 0.1
        else:
            epsilon_14 = float(x[2])
            r_min_14 = float(x[3]) * 0.1
        radius_s[x[0]] = np.array([[epsilon, r_min], [epsilon_14, r_min_14]])
    return radius_s, par_dihedrals
