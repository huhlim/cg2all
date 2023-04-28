#!/usr/bin/env python

import os
import sys
import json
import time
import tqdm
import pathlib
import argparse

import torch
import dgl

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"
import mdtraj

from cg2all.lib.libconfig import MODEL_HOME, DTYPE
from cg2all.lib.libdata import create_topology_from_data
import cg2all.lib.libcg
from cg2all.lib.libpdb import write_SSBOND
from cg2all.lib.libter import patch_termini
import cg2all.lib.libmodel
from cg2all.lib.torch_basics import v_norm_safe, inner_product, rotate_vector
from cg2all.lib.libcryoem import CryoEMLossFunction, MinimizableData

import warnings

warnings.filterwarnings("ignore")


def rotation_matrix_from_6D(v):
    v0 = v[0]
    v1 = v[1]
    e0 = v_norm_safe(v0, index=0)
    u1 = v1 - e0 * inner_product(e0, v1)
    e1 = v_norm_safe(u1, index=1)
    e2 = torch.cross(e0, e1)
    rot = torch.stack([e0, e1, e2], dim=1).mT
    return rot


def rigid_body_move(r, trans, rotation):
    center_of_mass = r.mean(dim=(0, 1))
    rotation_matrix = rotation_matrix_from_6D(rotation)
    #
    r_cg = r - center_of_mass
    r_cg = rotate_vector(rotation_matrix, r_cg) + center_of_mass + trans
    #
    return r_cg


def main():
    arg = argparse.ArgumentParser(prog="cryo_em_minimizer")
    arg.add_argument("-p", "--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("-m", "--map", dest="in_map_fn", required=True)
    arg.add_argument("-o", "--out", "--output", dest="out_dir", required=True)
    arg.add_argument("-a", "--all", "--is_all", dest="is_all", default=False, action="store_true")
    arg.add_argument("-n", "--step", dest="n_step", default=1000, type=int)
    arg.add_argument("--freq", "--output_freq", dest="output_freq", default=100, type=int)
    arg.add_argument("--restraint", dest="restraint", default=100.0, type=float)
    arg.add_argument(
        "--cg",
        dest="cg_model",
        default="ResidueBasedModel",
        choices=["CalphaBasedModel", "CA", "ca", "ResidueBasedModel", "RES", "res"],
    )
    arg = arg.parse_args()
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()

    if arg.cg_model in ["CalphaBasedModel", "CA", "ca"]:
        model_type = "CalphaBasedModel"
    elif arg.cg_model in ["ResidueBasedModel", "RES", "res"]:
        model_type = "ResidueBasedModel"
    ckpt_fn = MODEL_HOME / f"{model_type}.ckpt"
    if not ckpt_fn.exists():
        cg2all.lib.libmodel.download_ckpt_file(model_type, ckpt_fn)
    ckpt = torch.load(ckpt_fn, map_location=device)
    config = ckpt["hyper_parameters"]
    #
    if model_type == "CalphaBasedModel":
        cg_model = cg2all.lib.libcg.CalphaBasedModel
    elif model_type == "ResidueBasedModel":
        cg_model = cg2all.lib.libcg.ResidueBasedModel
    config = cg2all.lib.libmodel.set_model_config(config, cg_model)
    model = cg2all.lib.libmodel.Model(config, cg_model, compute_loss=False)
    #
    state_dict = ckpt["state_dict"]
    for key in list(state_dict):
        state_dict[".".join(key.split(".")[1:])] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.set_constant_tensors(device)
    model.eval()
    #
    data = MinimizableData(arg.in_pdb_fn, cg_model, is_all=arg.is_all)
    output_dir = pathlib.Path(arg.out_dir)
    output_dir.mkdir(exist_ok=True)
    #
    trans = torch.zeros(3, dtype=DTYPE, requires_grad=True)
    rotation = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=DTYPE, requires_grad=True)
    #
    optimizer = torch.optim.Adam([data.r_cg, trans, rotation], lr=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0005)
    #
    r_cg = rigid_body_move(data.r_cg, trans, rotation)
    batch = data.convert_to_batch(r_cg).to(device)
    ret = model.forward(batch)[0]
    #
    out_top, out_atom_index = create_topology_from_data(batch)
    out_mask = batch.ndata["output_atom_mask"].cpu().detach().numpy()
    #
    ssbond = []
    for cys_i, cys_j in enumerate(batch.ndata["ssbond_index"].cpu().detach().numpy()):
        if cys_j != -1:
            ssbond.append((cys_j, cys_i))
    ssbond.sort()
    #
    out_fn = output_dir / f"min.{0:04d}.pdb"
    xyz = ret["R"].cpu().detach().numpy()[out_mask > 0.0][None, out_atom_index]
    output = mdtraj.Trajectory(xyz=xyz, topology=out_top)
    output = patch_termini(output)
    output.save(out_fn)
    if len(ssbond) > 0:
        write_SSBOND(out_fn, output.top, ssbond)
    #
    loss_f = CryoEMLossFunction(
        arg.in_map_fn, data, device, model, model_type=model_type, restraint=arg.restraint
    )
    for i in range(arg.n_step):
        loss_sum, loss = loss_f.eval(batch, ret)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_([data.r_cg, trans, rotation], max_norm=1e4)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        #
        print("STEP", i, loss_sum.detach().cpu().item(), time.time() - time_start)
        print(
            {
                name: value.detach().cpu().item() * loss_f.weight[name]
                for name, value in loss.items()
            }
        )
        #
        r_cg = rigid_body_move(data.r_cg, trans, rotation)
        batch = data.convert_to_batch(r_cg).to(device)
        ret = model.forward(batch)[0]
        #
        if (i + 1) % arg.output_freq == 0:
            out_fn = output_dir / f"min.{i+1:04d}.pdb"
            xyz = ret["R"].cpu().detach().numpy()[out_mask > 0.0][None, out_atom_index]
            output = mdtraj.Trajectory(xyz=xyz, topology=out_top)
            output = patch_termini(output)
            output.save(out_fn)
            if len(ssbond) > 0:
                write_SSBOND(out_fn, output.top, ssbond)


if __name__ == "__main__":
    main()
