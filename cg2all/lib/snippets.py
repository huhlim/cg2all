#!/usr/bin/env python

import os
import sys
import mdtraj
import pathlib
import numpy as np
import functools

import dgl
import torch

import cg2all

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)
from libconfig import MODEL_HOME
from libdata import (
    PredictionData,
    create_trajectory_from_batch,
    create_topology_from_data,
)
from residue_constants import read_coarse_grained_topology
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
from libpdb import write_SSBOND
from libter import patch_termini
import libmodel

import warnings

warnings.filterwarnings("ignore")


def convert_cg2all(
    in_pdb_fn,
    out_fn,
    model_type="CalphaBasedModel",
    in_dcd_fn=None,
    ckpt_fn=None,
    device=None,
    n_proc=int(os.getenv("OMP_NUM_THREADS", 1)),
):
    # set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # load model ckpt file
    if ckpt_fn is None:
        ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
    ckpt = torch.load(ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]

    # configure model
    if config["cg_model"] == "CalphaBasedModel":
        cg_model = CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = ResidueBasedModel
    elif config["cg_model"] == "Martini":
        cg_model = Martini
    config = libmodel.set_model_config(config, cg_model)
    model = libmodel.Model(config, cg_model, compute_loss=False)

    # update state_dict
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()

    # prepare input data
    input_s = PredictionData(in_pdb_fn, cg_model, dcd_fn=in_dcd_fn, radius=config.globals.radius)
    if in_dcd_fn is not None:
        unitcell_lengths = input_s.cg.unitcell_lengths
        unitcell_angles = input_s.cg.unitcell_angles
    if len(input_s) > 1 and n_proc > 1:
        input_s = dgl.dataloading.GraphDataLoader(
            input_s, batch_size=1, num_workers=n_proc, shuffle=False
        )

    if in_dcd_fn is None:  # PDB file
        batch = input_s[0].to(device)
        #
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        #
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        output = patch_termini(traj_s[0])
        output.save(out_fn)
        if len(ssbond_s[0]) > 0:
            write_SSBOND(out_fn, output.top, ssbond_s[0])

    else:  # DCD file
        xyz = []
        for batch in input_s:
            batch = batch.to(device)
            #
            with torch.no_grad():
                R = model.forward(batch)[0]["R"].cpu().detach().numpy()
                mask = batch.ndata["output_atom_mask"].cpu().detach().numpy()
                xyz.append(R[mask > 0.0])
        #
        top, atom_index = create_topology_from_data(batch)
        xyz = np.array(xyz)[:, atom_index]
        traj = mdtraj.Trajectory(
            xyz=xyz,
            topology=top,
            unitcell_lengths=unitcell_lengths,
            unitcell_angles=unitcell_angles,
        )
        output = patch_termini(traj)
        output.save(out_fn)

    return output


def convert_all2cg(in_pdb_fn, out_fn, model_type="CalphaBasedModel", in_dcd_fn=None):
    if model_type in ["CA", "ca", "CalphaBasedModel"]:
        cg_model = CalphaBasedModel
    elif model_type in ["RES", "res", "ResidueBasedModel"]:
        cg_model = ResidueBasedModel
    elif model_type in ["Martini", "martini"]:
        cg_model = functools.partial(Martini, martini_top=read_coarse_grained_topology("martini"))
    else:
        raise KeyError(f"Unknown CG model, {model_type}\n")
    #
    cg = cg_model(in_pdb_fn, dcd_fn=in_dcd_fn)
    if in_dcd_fn is None:
        cg.write_cg(cg.R_cg, pdb_fn=out_fn)
        if len(cg.ssbond_s) > 0:
            write_SSBOND(out_fn, cg.top, cg.ssbond_s)
    else:
        cg.write_cg(cg.R_cg, dcd_fn=out_fn)
