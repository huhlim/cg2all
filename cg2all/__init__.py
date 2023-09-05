import sys
import pathlib

BASE = pathlib.Path(__file__).parents[0].resolve()
LIB_HOME = str(BASE / "lib")
sys.path.insert(0, LIB_HOME)

from cg2all.lib.snippets import convert_cg2all, convert_all2cg, load_model
