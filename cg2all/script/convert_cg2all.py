#!/usr/bin/env python

import os
import sys
import json
import time
import tqdm
import mdtraj
import pathlib
import argparse

import numpy as np

import torch
import dgl

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from libdata import PredictionData, create_trajectory_from_batch, create_topology_from_data
from libcg import ResidueBasedModel, CalphaBasedModel, Martini
from libpdb import write_SSBOND
from libter import patch_termini
import libmodel

import warnings

warnings.filterwarnings("ignore")


def main():
    arg = argparse.ArgumentParser(prog="cg2all")
    arg.add_argument("-p", "--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("-d", "--dcd", dest="in_dcd_fn", default=None)
    arg.add_argument("-o", "--out", "--output", dest="out_fn", required=True)
    arg.add_argument("--ckpt", dest="ckpt_fn", default=None, required=True)
    arg.add_argument("--time", dest="time_json", default=None)
    arg.add_argument("--device", dest="device", default=None)
    arg = arg.parse_args()
    #
    timing = {}
    #
    timing["loading_ckpt"] = time.time()
    if arg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(arg.device)
    ckpt = torch.load(arg.ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]
    timing["loading_ckpt"] = time.time() - timing["loading_ckpt"]
    #
    timing["model_configuration"] = time.time()
    if config["cg_model"] == "CalphaBasedModel":
        cg_model = CalphaBasedModel
    elif config["cg_model"] == "ResidueBasedModel":
        cg_model = ResidueBasedModel
    elif config["cg_model"] == "Martini":
        cg_model = Martini
    config = libmodel.set_model_config(config, cg_model)
    model = libmodel.Model(config, cg_model, compute_loss=False)
    #
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()
    timing["model_configuration"] = time.time() - timing["model_configuration"]
    #
    timing["loading_input"] = time.time()
    input_s = PredictionData(
        arg.in_pdb_fn, cg_model, dcd_fn=arg.in_dcd_fn, radius=config.globals.radius
    )
    timing["loading_input"] = time.time() - timing["loading_input"]
    #
    if arg.in_dcd_fn is None:
        t0 = time.time()
        batch = input_s[0].to(device)
        timing["loading_input"] += time.time() - t0
        #
        t0 = time.time()
        with torch.no_grad():
            R = model.forward(batch)[0]["R"]
        timing["forward_pass"] = time.time() - t0
        #
        timing["writing_output"] = time.time()
        traj_s, ssbond_s = create_trajectory_from_batch(batch, R)
        output = patch_termini(traj_s[0])
        output.save(arg.out_fn)
        if len(ssbond_s[0]) > 0:
            write_SSBOND(arg.out_fn, output.top, ssbond_s[0])
        timing["writing_output"] = time.time() - timing["writing_output"]
    else:
        timing["forward_pass"] = 0.0
        xyz = []
        t0 = time.time()
        for batch in tqdm.tqdm(input_s, total=len(input_s)):
            batch = batch.to(device)
            timing["loading_input"] += time.time() - t0
            #
            t0 = time.time()
            with torch.no_grad():
                R = model.forward(batch)[0]["R"].cpu().detach().numpy()
                mask = batch.ndata["output_atom_mask"].cpu().detach().numpy()
                xyz.append(R[mask > 0.0])
            timing["forward_pass"] += time.time() - t0
            t0 = time.time()
        #
        timing["writing_output"] = time.time()
        top = create_topology_from_data(batch)
        traj = mdtraj.Trajectory(xyz=np.array(xyz), topology=top)
        output = patch_termini(traj)
        output.save(arg.out_fn)
        timing["writing_output"] = time.time() - timing["writing_output"]

    time_total = 0.0
    for step, t in timing.items():
        time_total += t
    timing["total"] = time_total
    #
    print(timing)
    if arg.time_json is not None:
        with open(arg.time_json, "wt") as fout:
            fout.write(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
