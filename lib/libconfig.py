# %%
import os
import sys
import pathlib
import torch

BASE = pathlib.Path(__file__).parents[1]
LIB_HOME = BASE / "lib"
DATA_HOME = BASE / "data"

DTYPE = torch.get_default_dtype()
EPS = 1e-6
EQUIVARIANT_TOLERANCE = 1e-3
USE_EQUIVARIANCE_TEST = False
# EPS = torch.finfo(DTYPE).eps

# %%
