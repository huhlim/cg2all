#!/usr/bin/env python

import numpy as np

EPS = 1e-6

# Some basic functions
v_size = lambda v: np.linalg.norm(v, axis=-1)
v_norm = lambda v: v / v_size(v)[..., None]


def v_nonzero(v, index=0):
    safe = np.zeros(3, dtype=v.dtype)
    safe[index] = 1.0
    #
    if len(v.shape) > 1:
        s = np.where(v_size(v) < EPS)
        u = v.copy()
        u[s] = safe
    else:
        if v_size(v) < EPS:
            u = safe.copy()
        else:
            u = v
    return u


def v_norm_safe_np(v, index=0):
    return v_norm(v_nonzero(v, index=index))


def inner_product(v1, v2):
    return np.sum(v1 * v2, axis=-1)


def angle_sign(x):
    if isinstance(x, np.ndarray):
        s = np.sign(x)
        s[s == 0] = 1.0
        return s
    elif x >= 0:
        return 1.0
    else:
        return -1.0


# Some geometry functions
def bond_length(R) -> float:
    return v_size(R[..., 1, :] - R[..., 0, :])


# bond_angle: returns the angle consist of three atoms
def bond_angle(R) -> float:
    v1 = R[..., 0, :] - R[..., 1, :]
    v2 = R[..., 2, :] - R[..., 1, :]
    return np.arccos(np.clip(inner_product(v_norm(v1), v_norm(v2)), -1.0, 1.0))


# torsion_angle: returns the torsion angle consist of four atoms
def torsion_angle_old(R) -> float:
    torsion_axis = v_norm(R[..., 2, :] - R[..., 1, :])
    v0 = v_norm(R[..., 0, :] - R[..., 1, :])
    v1 = v_norm(R[..., 3, :] - R[..., 2, :])
    n0 = v_norm(np.cross(v0, torsion_axis))
    n1 = v_norm(np.cross(v1, torsion_axis))
    angle = np.arccos(np.clip(inner_product(n0, n1), -1.0, 1.0))
    sign = angle_sign(inner_product(v0, n1))
    return angle * sign


def torsion_angle(R) -> float:
    b1 = v_norm_safe_np(R[..., 1, :] - R[..., 0, :])
    b2 = v_norm_safe_np(R[..., 2, :] - R[..., 1, :])
    b3 = v_norm_safe_np(R[..., 3, :] - R[..., 2, :])
    #
    c1 = v_nonzero(np.cross(b2, b3))
    c2 = np.cross(b1, b2)
    #
    p1 = inner_product(b1, c1)
    p2 = inner_product(c1, c2)
    return np.arctan2(p1, p2)


# Algorithm 21. Rigid from 3 points using the Gram-Schmidt process
def rigid_from_3points(x):
    v0 = x[2] - x[1]
    v1 = x[0] - x[1]
    e0 = v_norm(v0)
    u1 = v1 - e0 * e0.T.dot(v1)
    e1 = v_norm(u1)
    e2 = np.cross(e0, e1)
    R = np.vstack((e0, e1, e2))
    t = x[1]
    return (R, t)


# Translate and rotate a set of coordinates
def translate_and_rotate(x, R, t):
    if len(t.shape) > 1:
        return x @ np.moveaxis(R, -1, -2) + t[:, None, :]
    else:
        return x @ np.moveaxis(R, -1, -2) + t


# Rotate around the x-axis
def rotate_x(t_ang):
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(t_ang), -np.sin(t_ang)],
            [0.0, np.sin(t_ang), np.cos(t_ang)],
        ],
        dtype=float,
    )
    t = np.zeros(3, dtype=float)
    return (R, t)


# internal_to_cartesian: X -- r0 -- r1 -- r2
def internal_to_cartesian(
    r0: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    b_len: float,
    b_ang: float,
    t_ang: float,
) -> np.ndarray:
    v1 = r0 - r1
    v2 = r0 - r2

    n1 = np.cross(v1, v2)
    n2 = np.cross(v1, n1)

    n1 = v_norm(n1)
    n2 = v_norm(n2)

    n1 *= -np.sin(t_ang)
    n2 *= np.cos(t_ang)

    v3 = v_norm(n1 + n2)
    v3 *= b_len * np.sin(b_ang)

    v1 = v_norm(v1)
    v1 *= b_len * np.cos(b_ang)

    x = r0 + v3 - v1
    return x
