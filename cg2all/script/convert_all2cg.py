#!/usr/bin/env python

import os
import sys
import argparse
import pathlib
import functools

os.environ["OPENMM_PLUGIN_DIR"] = "/dev/null"

import cg2all.lib.libcg
from cg2all.lib.libpdb import write_SSBOND
from cg2all.lib.residue_constants import read_coarse_grained_topology


def main():
    arg = argparse.ArgumentParser(prog="convert_all2cg")
    arg.add_argument("-p", "--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("-d", "--dcd", dest="in_dcd_fn", default=None)
    arg.add_argument("-o", "--out", "--output", dest="out_fn", required=True)
    arg.add_argument(
        "--cg",
        dest="cg_model",
        default="CalphaBasedModel",
        # fmt:off
        choices=["CalphaBasedModel", "CA", "ca", \
                "ResidueBasedModel", "RES", "res", \
                "Martini", "martini", \
                "PRIMO", "primo", \
                "CACM", "cacm", "CalphaCM", "CalphaCMModel",\
                "BB", "bb", "backbone", "Backbone", "BackboneModel", \
                "MC", "mc", "mainchain", "Mainchain", "MainchainModel",
                ]
        # fmt:on
    )
    arg = arg.parse_args()
    #
    if arg.cg_model in ["CA", "ca", "CalphaBasedModel"]:
        cg_model = cg2all.lib.libcg.CalphaBasedModel
    elif arg.cg_model in ["RES", "res", "ResidueBasedModel"]:
        cg_model = cg2all.lib.libcg.ResidueBasedModel
    elif arg.cg_model in ["Martini", "martini"]:
        cg_model = functools.partial(
            cg2all.lib.libcg.Martini, topology_map=read_coarse_grained_topology("martini")
        )
    elif arg.cg_model in ["PRIMO", "primo"]:
        cg_model = functools.partial(
            cg2all.lib.libcg.PRIMO, topology_map=read_coarse_grained_topology("primo")
        )
    elif arg.cg_model in ["CACM", "cacm", "CalphaCM", "CalphaCMModel"]:
        cg_model = cg2all.lib.libcg.CalphaCMModel
    elif arg.cg_model in ["BB", "bb", "backbone", "Backbone", "BackboneModel"]:
        cg_model = cg2all.lib.libcg.BackboneModel
    elif arg.cg_model in ["MC", "mc", "mainchain", "Mainchain", "MainchainModel"]:
        cg_model = cg2all.lib.libcg.MainchainModel
    else:
        raise KeyError(f"Unknown CG model, {arg.cg_model}\n")
    #
    cg = cg_model(arg.in_pdb_fn, dcd_fn=arg.in_dcd_fn)
    if arg.in_dcd_fn is None:
        cg.write_cg(cg.R_cg, pdb_fn=arg.out_fn)
        if len(cg.ssbond_s) > 0:
            write_SSBOND(arg.out_fn, cg.top, cg.ssbond_s)
    else:
        cg.write_cg(cg.R_cg, dcd_fn=arg.out_fn)


if __name__ == "__main__":
    main()
