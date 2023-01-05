#!/usr/bin/env python

import os
import sys
import argparse
import pathlib
import functools

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from libcg import ResidueBasedModel, CalphaBasedModel, Martini
from libpdb import write_SSBOND
from residue_constants import read_martini_topology


def main():
    arg = argparse.ArgumentParser(prog="all2cg")
    arg.add_argument("-p", "--pdb", dest="in_pdb_fn", required=True)
    arg.add_argument("-d", "--dcd", dest="in_dcd_fn", default=None)
    arg.add_argument("-o", "--out", "--output", dest="out_fn", required=True)
    arg.add_argument(
        "--cg",
        dest="cg_model",
        default="CalphaBasedModel",
        #fmt:off
        choices=["CalphaBasedModel", "CA", "ca", \
                "ResidueBasedModel", "RES", "res", \
                "Martini", "martini"]
        #fmt:on
    )
    arg = arg.parse_args()
    #
    if arg.cg_model in ["CA", "ca", "CalphaBasedModel"]:
        cg_model = CalphaBasedModel
    elif arg.cg_model in ["RES", "res", "ResidueBasedModel"]:
        cg_model = ResidueBasedModel
    elif arg.cg_model in ["Martini", "martini"]:
        cg_model = functools.partial(Martini, martini_top=read_martini_topology())
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
