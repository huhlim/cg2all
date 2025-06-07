import importlib.util
import sys
import torch

# Load libmetric without triggering package __init__
sys.path.insert(0, "cg2all/lib")
if not hasattr(torch, "power"):
    torch.power = torch.pow
spec = importlib.util.spec_from_file_location("libmetric", "cg2all/lib/libmetric.py")
libmetric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(libmetric)

def test_rmsd_torsion_angle_runs():
    sc0 = torch.zeros(4, 5, 2)
    sc_ref = torch.zeros(4, 5)
    mask = torch.ones(4, 5)
    libmetric.rmsd_torsion_angle(sc0, sc_ref, mask)
