#!/usr/bin/env python

import cg2all
import mdtraj
import torch
import dgl
from cg2all.lib.libconfig import DTYPE
from cg2all.lib.residue_constants import AMINO_ACID_s
from cg2all.lib.libpdb import get_HIS_state
from cg2all.lib.libdata import XYZData
from cg2all import load_model

pdb_fn = "1jni.calpha.pdb"
dcd_fn = "1jni.calpha.dcd"

device = torch.device("cpu")
pdb = mdtraj.load(dcd_fn, top=pdb_fn)
pdb = pdb.atom_slice(pdb.top.select("name CA"))
xyz = torch.as_tensor(pdb.xyz[:, :, None, :], dtype=DTYPE, device=device)
xyz.requires_grad_(True)
#
residue_type = []
for residue in pdb.top.residues:
    if residue.name == "HIS":
        residue.name = get_HIS_state(residue)
    residue_type.append(AMINO_ACID_s.index(residue.name))
residue_type = torch.as_tensor(residue_type, dtype=torch.long, device=device)
mask = torch.ones(xyz.shape[:-1], dtype=DTYPE, device=device)
mask[0, 50:, 0] = 0.0
#
xyzdata = XYZData(xyz, residue_type, mask=mask)
dataloader = dgl.dataloading.GraphDataLoader(xyzdata, batch_size=xyzdata.batch_size, shuffle=False)
batch = next(iter(dataloader))
# #
model, _, _ = load_model(device=device)
model.eval()
out = model(batch)[0]["R"]
print(out.shape)
