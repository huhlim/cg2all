import torch
import numpy as np
from libconfig import EPS

pi = torch.tensor(np.pi)

# some basic functions
v_size = lambda v: torch.linalg.norm(v, dim=-1)
v_norm = lambda v: v / v_size(v)[..., None]


def v_norm_safe_prev(v, index=0):
    u = v.clone()
    u[..., index] = u[..., index] + EPS
    return v_norm(u)


def v_nonzero(v, index=0):
    safe = torch.zeros(v.size(-1), device=v.device, dtype=v.dtype)
    safe[index] = 1.0
    #
    size = v_size(v)[..., None]
    u = torch.where(size > EPS, v, safe)
    return u


def v_norm_safe(v, index=0):
    return v_norm(v_nonzero(v, index=index))


def inner_product(v1, v2):
    return torch.sum(v1 * v2, dim=-1)


def rotate_matrix(R, X):
    return R @ X


def rotate_vector(R, X):
    return (X[..., None, :] @ R.mT)[..., 0, :]


def rotate_vector_inv(R, X):
    R_inv = torch.inverse(R)
    return rotate_vector(R_inv, X)


def angle_sign(x):
    s = torch.sign(x)
    s[s == 0] = 1.0
    return s


def acos_safe(x, eps=EPS):
    # torch.acos is unstable around -1 and 1 -> added EPS
    return torch.acos(torch.clamp(x, -1.0 + eps, 1.0 - eps))


def torsion_angle_prev(R: torch.Tensor) -> torch.Tensor:
    torsion_axis = v_norm(R[..., 2, :] - R[..., 1, :])
    v0 = v_norm_safe(R[..., 0, :] - R[..., 1, :], index=0)
    v1 = v_norm_safe(R[..., 3, :] - R[..., 2, :], index=1)
    n0 = v_norm_safe(torch.linalg.cross(v0, torsion_axis, dim=-1), index=0)
    n1 = v_norm_safe(torch.linalg.cross(v1, torsion_axis, dim=-1), index=1)
    angle = acos_safe(inner_product(n0, n1))
    sign = angle_sign(inner_product(v0, n1))
    return angle * sign


def torsion_angle(R: torch.Tensor) -> torch.Tensor:
    b1 = v_norm_safe(R[..., 1, :] - R[..., 0, :])
    b2 = v_norm_safe(R[..., 2, :] - R[..., 1, :])
    b3 = v_norm_safe(R[..., 3, :] - R[..., 2, :])
    #
    c1 = v_nonzero(torch.linalg.cross(b2, b3, dim=-1))
    c2 = torch.linalg.cross(b1, b2, dim=-1)
    #
    p1 = inner_product(b1, c1)
    p2 = inner_product(c1, c2)
    return torch.atan2(p1, p2)


def one_hot_encoding(X, X_min, X_max, nX) -> torch.Tensor:
    dX = (X_max - X_min) / nX
    index = ((X - X_min) / dX).type(torch.long)
    index = torch.clip(index, min=0, max=nX - 1)
    return torch.eye(nX)[index]
