#!/usr/bin/env python

import os
import sys
import pathlib
import torch

BASE = pathlib.Path(__file__).parents[1].resolve()
LIB_HOME = BASE / "lib"
DATA_HOME = BASE / "data"

DTYPE = torch.get_default_dtype()
EPS = 1e-6
