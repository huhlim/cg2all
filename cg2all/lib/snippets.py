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

from libconfig import MODEL_HOME
from libdata import (
    PredictionData,
    create_trajectory_from_batch,
    create_topology_from_data,
)
from residue_constants import read_coarse_grained_topology
import libcg
from libpdb import write_SSBOND
from libter import patch_termini
import libmodel

import warnings

warnings.filterwarnings("ignore")


def load_model(model_type="CalphaBasedModel", ckpt_fn=None, fix_atom=False, device=None):
    # set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    #
    # set ckpt_fn path
    if ckpt_fn is None:
        if model_type not in [
            "CalphaBasedModel",
            "ResidueBasedModel",
            "CalphaCMModel",
            "CalphaSCModel",
            "SidechainModel",
            "BackboneModel",
            "MainchainModel",
            "Martini",
            "PRIMO",
        ]:
            raise ValueError(f"Invalid model_type, {model_type}")
        #
        if fix_atom:
            ckpt_fn = MODEL_HOME / f"{model_type}-FIX.ckpt"
        else:
            ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
        #
        if not os.path.exists(ckpt_fn):
            libmodel.download_ckpt_file(model_type, ckpt_fn, fix_atom=fix_atom)

    else:
        if not os.path.exists(ckpt_fn):
            raise FileNotFoundError(ckpt_fn)
    #
    # load the ckpt file
    ckpt = torch.load(ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]

    if config["cg_model"] == "CalphaBasedModel":
        cg_model = libcg.CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = libcg.ResidueBasedModel
    elif config["cg_model"] == "Martini":
        cg_model = libcg.Martini
    elif config["cg_model"] == "PRIMO":
        cg_model = libcg.PRIMO
    elif config["cg_model"] == "CalphaCMModel":
        cg_model = libcg.CalphaCMModel
    elif config["cg_model"] == "CalphaSCModel":
        cg_model = libcg.CalphaSCModel
    elif config["cg_model"] == "SidechainModel":
        cg_model = libcg.SidechainModel
    elif config["cg_model"] == "BackboneModel":
        cg_model = libcg.BackboneModel
    elif config["cg_model"] == "MainchainModel":
        cg_model = libcg.MainchainModel
    #
    config = libmodel.set_model_config(config, cg_model, flattened=False)
    model = libmodel.Model(config, cg_model, compute_loss=False)
    #
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)

    return model, cg_model, config


def convert_cg2all(
    in_pdb_fn,
    out_fn=None,
    model_type="CalphaBasedModel",
    in_dcd_fn=None,
    ckpt_fn=None,
    fix_atom=False,
    device=None,
    batch_size=1,
    n_proc=int(os.getenv("OMP_NUM_THREADS", 1)),
):
    model, cg_model, config = load_model(
        model_type=model_type, ckpt_fn=ckpt_fn, fix_atom=fix_atom, device=device
    )
    model.eval()

    # prepare input data
    input_s = PredictionData(
        in_pdb_fn,
        cg_model,
        dcd_fn=in_dcd_fn,
        radius=config.globals.radius,
        fix_atom=config.globals.fix_atom,
    )
    if in_dcd_fn is not None:
        n_frame0 = input_s.n_frame0
        unitcell_lengths = input_s.cg.unitcell_lengths
        unitcell_angles = input_s.cg.unitcell_angles
    if len(input_s) > 1 and (n_proc > 1 or batch_size > 1):
        input_s = dgl.dataloading.GraphDataLoader(
            input_s, batch_size=batch_size, num_workers=n_proc, shuffle=False
        )
    else:
        input_s = dgl.dataloading.GraphDataLoader(
            input_s, batch_size=1, num_workers=1, shuffle=False
        )

    if in_dcd_fn is None:  # PDB file
        batch = next(iter(input_s)).to(device)
        #
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        #
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        output = patch_termini(traj_s[0])
        if out_fn is not None:
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
        if batch_size > 1:
            batch = dgl.unbatch(batch)[0]
            xyz = np.concatenate(xyz, axis=0)
            xyz = xyz.reshape((n_frame0, -1, 3))
        else:
            xyz = np.array(xyz)
        top, atom_index = create_topology_from_data(batch)
        xyz = np.array(xyz)[:, atom_index]
        traj = mdtraj.Trajectory(
            xyz=xyz,
            topology=top,
            unitcell_lengths=unitcell_lengths,
            unitcell_angles=unitcell_angles,
        )
        output = patch_termini(traj)
        if out_fn is not None:
            output.save(out_fn)

    return output


def convert_all2cg(in_pdb_fn, out_fn, model_type="CalphaBasedModel", in_dcd_fn=None):
    if model_type in ["CA", "ca", "CalphaBasedModel"]:
        cg_model = libcg.CalphaBasedModel
    elif model_type in ["RES", "res", "ResidueBasedModel"]:
        cg_model = libcg.ResidueBasedModel
    elif model_type in ["Martini", "martini"]:
        cg_model = functools.partial(
            libcg.Martini, topology_map=read_coarse_grained_topology("martini")
        )
    elif model_type in ["PRIMO", "primo"]:
        cg_model = functools.partial(
            libcg.PRIMO, topology_map=read_coarse_grained_topology("primo")
        )
    elif model_type in ["CACM", "cacm", "CalphaCM", "CalphaCMModel"]:
        cg_model = libcg.CalphaCMModel
    elif model_type in ["BB", "bb", "backbone", "Backbone", "BackboneModel"]:
        cg_model = libcg.BackboneModel
    elif model_type in ["MC", "mc", "mainchain", "Mainchain", "MainchainModel"]:
        cg_model = libcg.MainchainModel
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
