#!/usr/bin/env python

import torch

import dgl
from typing import Optional, List

from libconfig import DTYPE, EPS, DATA_HOME
from residue_constants import (
    MAX_RESIDUE_TYPE,
    AMINO_ACID_s,
    PROLINE_INDEX,
    ATOM_INDEX_N,
    ATOM_INDEX_CA,
    ATOM_INDEX_C,
    ATOM_INDEX_PRO_CD,
    ATOM_INDEX_CYS_CB,
    ATOM_INDEX_CYS_SG,
    BOND_LENGTH0,
    BOND_LENGTH_PROLINE_RING,
    BOND_LENGTH_DISULFIDE,
    BOND_ANGLE0,
)

from libcg import get_residue_center_of_mass
from torch_basics import (
    v_size,
    v_norm,
    v_norm_safe,
    inner_product,
    rotate_vector_inv,
    acos_safe,
    pi,
    torsion_angle,
)


def loss_f(
    batch,
    ret,
    loss_weight,
    use_alt=True,
    loss_prev=None,
    RIGID_OPs=None,
    TORSION_PARs=None,
):
    R = ret["R"]
    opr_bb = ret["opr_bb"]
    #
    if use_alt:
        # at this point, it is used ONLY for FAPE_all/mse_R
        # because the others do NOT depend on symmetric sidechain coordinates
        batch.ndata["output_xyz_ref"] = get_output_xyz_ref(batch, R)
    else:
        batch.ndata["output_xyz_ref"] = batch.ndata["output_xyz"]
    #
    loss = {}
    if loss_weight.get("rigid_body", 0.0) > 0.0:
        loss["rigid_body"] = loss_f_rigid_body(batch, R) * loss_weight.rigid_body
    if loss_weight.get("mse_R", 0.0) > 0.0:
        loss["mse_R"] = loss_f_mse_R(batch, R) * loss_weight.mse_R
    if loss_weight.get("v_cntr", 0.0) > 0.0:
        loss["v_cntr"] = loss_f_v_cntr(batch, R) * loss_weight.v_cntr
    if loss_weight.get("FAPE_CA", 0.0) > 0.0:
        loss["FAPE_CA"] = (
            loss_f_FAPE_CA(batch, R, opr_bb, d_clamp=loss_weight.FAPE_d_clamp) * loss_weight.FAPE_CA
        )
    if loss_weight.get("FAPE_all", 0.0) > 0.0:
        loss["FAPE_all"] = (
            loss_f_FAPE_all(batch, R, opr_bb, d_clamp=loss_weight.FAPE_d_clamp)
            * loss_weight.FAPE_all
        )
    if loss_weight.get("rotation_matrix", 0.0) > 0.0:
        loss["rotation_matrix"] = (
            loss_f_rotation_matrix(batch, ret["bb"], ret.get("bb0", None))
            * loss_weight.rotation_matrix
        )
    if loss_weight.get("bonded_energy", 0.0) > 0.0:
        loss["bonded_energy"] = (
            loss_f_bonded_energy(batch, R, weight_s=(1.0, 0.5)) + loss_f_bonded_energy_aux(batch, R)
        ) * loss_weight.bonded_energy
    if loss_weight.get("backbone_torsion", 0.0) > 0.0:
        loss["backbone_torsion"] = loss_f_backbone_torsion(batch, R) * loss_weight.backbone_torsion
    if loss_weight.get("torsion_angle", 0.0) > 0.0:
        loss["torsion_angle"] = (
            loss_f_torsion_angle(batch, ret["sc"], sc0=ret.get("sc0", None))
        ) * loss_weight.torsion_angle
    if loss_weight.get("torsion_energy", 0.0) > 0.0:
        loss["torsion_energy"] = (
            loss_f_torsion_energy(
                batch,
                R,
                ret["ss"],
                TORSION_PARs,
                energy_clamp=loss_weight.get("torsion_energy_clamp", 0.0),
            )
            * loss_weight.torsion_energy
        )
    if loss_weight.get("atomic_clash", 0.0) > 0.0:
        loss["atomic_clash"] = (
            loss_f_atomic_clash(
                batch,
                R,
                RIGID_OPs,
                vdw_scale=loss_weight.get("atomic_clash_vdw", 1.0),
                energy_clamp=loss_weight.get("atomic_clash_clamp", 0.0),
            )
            * loss_weight.atomic_clash
        )
    if loss_weight.get("ss", 0.0) > 0.0:
        loss["ss"] = loss_f_ss(batch, ret["ss0"]) * loss_weight.ss
    #
    if loss_prev is not None:
        for k, v in loss_prev.items():
            if k in loss:
                loss[k] += v
            else:
                loss[k] = v
    return loss


def get_output_xyz_ref(batch: dgl.DGLGraph, R: torch.Tensor) -> torch.Tensor:
    mask = batch.ndata["heavy_atom_mask"]
    d = torch.sum(torch.pow(R - batch.ndata["output_xyz"], 2).sum(dim=-1) * mask, dim=-1)
    d_alt = torch.sum(torch.pow(R - batch.ndata["output_xyz_alt"], 2).sum(dim=-1) * mask, dim=-1)
    #
    xyz = torch.where(
        (d <= d_alt)[:, None, None],
        batch.ndata["output_xyz"],
        batch.ndata["output_xyz_alt"],
    ).detach()
    return xyz


# MSE loss for comparing coordinates
def loss_f_mse_R(batch: dgl.DGLGraph, R: torch.Tensor, use_alt=True):
    mask = batch.ndata["heavy_atom_mask"][..., None]
    d = torch.sum(torch.pow((R - batch.ndata["output_xyz_ref"]) * mask, 2))
    return d / mask.sum()


def loss_f_v_cntr(batch: dgl.DGLGraph, R: torch.Tensor):
    r_cntr = get_residue_center_of_mass(R, batch.ndata["atomic_mass"])
    v_cntr = r_cntr - R[:, ATOM_INDEX_CA]
    loss_angle = torch.mean(
        torch.abs(1.0 - inner_product(v_norm_safe(v_cntr), v_norm(batch.ndata["v_cntr"])))
    )
    #
    loss_distance = torch.mean(torch.abs(v_size(v_cntr) - v_size(batch.ndata["v_cntr"])))
    return loss_angle + loss_distance * 10.0


def loss_f_rigid_body(batch: dgl.DGLGraph, R: torch.Tensor) -> torch.Tensor:
    R_ref = batch.ndata["output_xyz"]
    #
    # deviation of the backbone rigids, N, CA, C
    loss_translation = torch.mean(torch.abs(R[:, :3, :] - R_ref[:, :3, :]))
    #
    v0 = v_norm(R[:, ATOM_INDEX_C, :] - R[:, ATOM_INDEX_CA, :])
    v0_ref = v_norm(R_ref[:, ATOM_INDEX_C, :] - R_ref[:, ATOM_INDEX_CA, :])
    v1 = v_norm(R[:, ATOM_INDEX_N, :] - R[:, ATOM_INDEX_CA, :])
    v1_ref = v_norm(R_ref[:, ATOM_INDEX_N, :] - R_ref[:, ATOM_INDEX_CA, :])
    loss_rotation_0 = torch.mean(torch.abs(1.0 - inner_product(v0, v0_ref)))
    loss_rotation_1 = torch.mean(torch.abs(1.0 - inner_product(v1, v1_ref)))
    #
    return loss_translation + loss_rotation_0 + loss_rotation_1


def loss_f_rotation_matrix(
    batch: dgl.DGLGraph, bb: torch.Tensor, bb0: torch.Tensor, norm_weight: float = 0.01
) -> torch.Tensor:
    loss_bb = torch.mean(torch.abs(bb[:, :2] - batch.ndata["correct_bb"][:, :2]))
    #
    if bb0 is not None and norm_weight > 0.0:
        loss_bb_norm_1 = torch.mean(torch.abs(v_size(bb0[:, 0]) - 1.0))
        loss_bb_norm_2 = torch.mean(torch.abs(v_size(bb0[:, 1]) - 1.0))
        return loss_bb + (loss_bb_norm_1 + loss_bb_norm_2) * norm_weight
    else:
        return loss_bb


def loss_f_FAPE_CA(
    batch: dgl.DGLGraph, R: torch.Tensor, bb: torch.Tensor, d_clamp: float = 1.0
) -> torch.Tensor:
    # time: ~30% vs. loss_f_FAPE_CA_old
    # memory: O(N^2) vs. O(N) for loss_f_FAPE_CA_old
    #   (e.g., 51 MB vs. 0.2 MB for 855 aa.)
    #
    first = 0
    loss = 0.0
    for batch_index, data in enumerate(dgl.unbatch(batch)):
        n_residue = data.num_nodes()
        last = first + n_residue
        #
        _R = R[first:last, ATOM_INDEX_CA]
        _bb = bb[first:last][:, None]
        R_ref = data.ndata["output_xyz"][:, ATOM_INDEX_CA]
        bb_ref = data.ndata["correct_bb"][:, None]
        #
        r = rotate_vector_inv(_bb[:, :, :3], _R[None, :] - _bb[:, :, 3])
        r_ref = rotate_vector_inv(bb_ref[:, :, :3], R_ref[None, :] - bb_ref[:, :, 3])
        dr = r - r_ref
        d = torch.clamp(torch.sqrt(torch.pow(dr, 2).sum(dim=-1) + EPS**2), max=d_clamp)
        loss = loss + torch.mean(d)
        #
        first = last

    return loss / batch.batch_size


def loss_f_FAPE_all(
    batch: dgl.DGLGraph, R: torch.Tensor, bb: torch.Tensor, d_clamp: float = 1.0
) -> torch.Tensor:
    # time: ~30% vs. loss_f_FAPE_all_old
    # memory: O(N^2) vs. O(N) for loss_f_FAPE_all_old
    #   (e.g., 1,200 MB vs. 1.2 MB for 855 aa.)
    #
    first = 0
    loss = 0.0
    for batch_index, data in enumerate(dgl.unbatch(batch)):
        n_residue = data.num_nodes()
        last = first + n_residue
        #
        mask = data.ndata["heavy_atom_mask"] > 0.0
        _R = R[first:last]
        _bb = bb[first:last][:, None]  # shape=(N, 1, 4, 3)
        bb_ref = data.ndata["correct_bb"][:, None]
        R_ref = data.ndata["output_xyz_ref"]  # use symmetry-corrected reference structure
        #
        r = rotate_vector_inv(_bb[:, :, None, :3], _R[None, :] - _bb[..., 3, :][:, None])
        r_ref = rotate_vector_inv(
            bb_ref[:, :, None, :3], R_ref[None, :] - bb_ref[..., 3, :][:, None]
        )
        d = torch.pow((r - r_ref)[:, mask], 2).sum(dim=-1)
        fape = torch.clamp(torch.sqrt(d + EPS**2), max=d_clamp)
        loss = loss + torch.mean(fape)
        #
        first = last
        #
    return loss / batch.batch_size


# Bonded energy penalties
def loss_f_bonded_energy(batch: dgl.DGLGraph, R: torch.Tensor, weight_s=(1.0, 0.5)):
    if weight_s[0] == 0.0:
        return 0.0

    R_ref = batch.ndata["output_xyz"]
    bonded = batch.ndata["continuous"][1:]
    n_bonded = torch.sum(bonded)

    # vector: -C -> N
    v1 = R[1:, ATOM_INDEX_N, :] - R[:-1, ATOM_INDEX_C, :]
    v1_ref = R_ref[1:, ATOM_INDEX_N, :] - R_ref[:-1, ATOM_INDEX_C, :]
    #
    # bond lengths
    d1 = v_size(v1)
    d1_ref = v_size(v1_ref)
    bond_energy = torch.sum(torch.abs(d1 - d1_ref) * bonded) / n_bonded
    if weight_s[1] == 0.0:
        return bond_energy * weight_s[0]
    #
    # vector: -CA -> -C
    v0 = R[:-1, ATOM_INDEX_C, :] - R[:-1, ATOM_INDEX_CA, :]
    v0_ref = R_ref[:-1, ATOM_INDEX_C, :] - R_ref[:-1, ATOM_INDEX_CA, :]
    # vector: N -> CA
    v2 = R[1:, ATOM_INDEX_CA, :] - R[1:, ATOM_INDEX_N, :]
    v2_ref = R_ref[1:, ATOM_INDEX_CA, :] - R_ref[1:, ATOM_INDEX_N, :]
    #
    d0 = v_size(v0)
    d2 = v_size(v2)
    #
    # bond angles
    def bond_angle(v1, v2):
        return acos_safe(inner_product(v1, v2))

    v0 = v_norm(v0)
    v1 = v1 / d1[..., None]
    v2 = v_norm(v2)
    a01 = bond_angle(-v0, v1)
    a12 = bond_angle(-v1, v2)
    #
    v0_ref = v_norm(v0_ref)
    v1_ref = v1_ref / d1_ref[..., None]
    v2_ref = v_norm(v2_ref)
    a01_ref = bond_angle(-v0_ref, v1_ref)
    a12_ref = bond_angle(-v1_ref, v2_ref)
    #
    angle_energy = torch.abs(a01 - a01_ref) + torch.abs(a12 - a12_ref)
    angle_energy = torch.sum(angle_energy * bonded) / n_bonded
    #
    return bond_energy * weight_s[0] + angle_energy * weight_s[1]


def loss_f_bonded_energy_aux(batch: dgl.DGLGraph, R: torch.Tensor):
    # proline ring closure
    proline = batch.ndata["residue_type"] == PROLINE_INDEX
    if torch.any(proline):
        R_pro_N = R[proline, ATOM_INDEX_N]
        R_pro_CD = R[proline, ATOM_INDEX_PRO_CD]
        d_pro = v_size(R_pro_N - R_pro_CD)
        bond_energy_pro = torch.sum(torch.abs(d_pro - BOND_LENGTH_PROLINE_RING)) / R.size(0)
    else:
        bond_energy_pro = torch.zeros(1, dtype=DTYPE, device=R.device)

    # disulfide bond
    bond_energy_ssbond = torch.zeros(1, dtype=DTYPE, device=R.device)
    for batch_index in range(batch.batch_size):
        data = dgl.slice_batch(batch, batch_index, store_ids=True)
        if not torch.any(data.ndata["ssbond_index"] >= 0):
            continue
        #
        _R = R[data.ndata["_ID"]]
        cys_1_index = data.ndata["ssbond_index"]
        disu = cys_1_index >= 0
        cys_0_index = data.nodes()[disu]
        cys_1_index = cys_1_index[disu]
        R_cys_0 = _R[cys_0_index, ATOM_INDEX_CYS_SG]
        R_cys_1 = _R[cys_1_index, ATOM_INDEX_CYS_SG]
        d_ssbond = v_size(R_cys_1 - R_cys_0)
        bond_energy_ssbond = bond_energy_ssbond + torch.sum(
            torch.abs(d_ssbond - BOND_LENGTH_DISULFIDE)
        ) / R.size(0)

    return bond_energy_pro.sum() + bond_energy_ssbond.sum()


def loss_f_backbone_torsion(batch: dgl.DGLGraph, R: torch.Tensor):
    bonded = batch.ndata["continuous"][1:] > 0.0
    #
    r_N = R[:, (ATOM_INDEX_N,), :]
    r_CA = R[:, (ATOM_INDEX_CA,), :]
    r_C = R[:, (ATOM_INDEX_C,), :]
    r = torch.stack(
        [
            torch.cat([r_C[:-1], r_N[1:], r_CA[1:], r_C[1:]], dim=1),
            torch.cat([r_N[:-1], r_CA[:-1], r_C[:-1], r_N[1:]], dim=1),
            torch.cat([r_CA[:-1], r_C[:-1], r_N[1:], r_CA[1:]], dim=1),
        ],
        dim=1,
    )[bonded]
    angle = torsion_angle(r)
    #
    R_ref = batch.ndata["output_xyz"]
    r_N_ref = R_ref[:, (ATOM_INDEX_N,), :]
    r_CA_ref = R_ref[:, (ATOM_INDEX_CA,), :]
    r_C_ref = R_ref[:, (ATOM_INDEX_C,), :]
    r_ref = torch.stack(
        [
            torch.cat([r_C_ref[:-1], r_N_ref[1:], r_CA_ref[1:], r_C_ref[1:]], dim=1),
            torch.cat([r_N_ref[:-1], r_CA_ref[:-1], r_C_ref[:-1], r_N_ref[1:]], dim=1),
            torch.cat([r_CA_ref[:-1], r_C_ref[:-1], r_N_ref[1:], r_CA_ref[1:]], dim=1),
        ],
        dim=1,
    )[bonded]
    angle_ref = torsion_angle(r_ref)
    #
    loss = torch.sum(
        1.0 - (torch.cos(angle) * torch.cos(angle_ref)) - (torch.sin(angle) * torch.sin(angle_ref))
    )
    loss = loss / angle.size(0)
    return loss


def loss_f_torsion_angle(
    batch: dgl.DGLGraph,
    sc: torch.Tensor,
    sc0: Optional[torch.Tensor] = None,
    norm_weight: Optional[float] = 0.01,
):
    sc_ref = batch.ndata["correct_torsion"]  # shape=(N, MAX_TORSION, MAX_PERIODIC)
    mask = batch.ndata["torsion_mask"]  # shape=(N, MAX_TORSION)
    #
    sc_cos = torch.cos(sc_ref)
    sc_sin = torch.sin(sc_ref)
    dot = torch.max(sc[..., 0, None] * sc_cos + sc[..., 1, None] * sc_sin, dim=2)[0]
    loss = torch.sum((1.0 - dot) * mask)
    #
    if sc0 is not None and norm_weight > 0.0:
        norm = v_size(sc0)
        loss_norm = torch.sum(torch.abs(norm - 1.0) * mask)
        loss = loss + loss_norm * norm_weight
    loss = loss / mask.sum()
    return loss


def loss_f_atomic_clash(
    batch: dgl.DGLGraph,
    R: torch.Tensor,
    RIGID_OPs,
    vdw_scale=1.0,
    energy_clamp=0.0,
    g_radius=1.4,
):
    # c ~ average number of edges per node
    # time: O(Nxc) vs. O(N^2) for loss_f_atomic_clash_old
    # memory: O(Nxc) vs. O(N) for loss_f_atomic_clash_old
    #   (e.g., 70 MB vs. 20 MB for 855 aa.)
    # this can be approximate if radius is small
    #
    _RIGID_GROUPS_DEP = RIGID_OPs[1][1]
    #
    def get_pairs(data, R, g_radius):
        g = dgl.radius_graph(R, g_radius, self_loop=False)
        #
        edges = g.edges()
        ssbond = torch.zeros_like(edges[0], dtype=bool)
        #
        cys_i = torch.nonzero(data.ndata["ssbond_index"] >= 0)[:, 0]
        if cys_i.size(0) > 0:
            cys_j = data.ndata["ssbond_index"][cys_i]
            try:
                eids = g.edge_ids(cys_i, cys_j)
                ssbond[eids] = True
            except:
                pass
        #
        subset = edges[0] > edges[1]
        edges = (edges[0][subset], edges[1][subset])
        ssbond = ssbond[subset]
        #
        return edges, ssbond

    #
    energy = 0.0
    for batch_index in range(batch.batch_size):
        data = dgl.slice_batch(batch, batch_index, store_ids=True)
        _R = R[data.ndata["_ID"]]
        (i, j), ssbond = get_pairs(data, _R[:, ATOM_INDEX_CA], g_radius=g_radius)
        #
        mask_i = data.ndata["output_atom_mask"][i] > 0.0
        mask_j = data.ndata["output_atom_mask"][j] > 0.0
        mask = mask_j[:, :, None] & mask_i[:, None, :]
        #
        # find consecutive residue pairs (i == j + 1)
        y = i == j + 1
        curr_residue_type = data.ndata["residue_type"][i[y]]
        prev_residue_type = data.ndata["residue_type"][j[y]]
        curr_bb = _RIGID_GROUPS_DEP[curr_residue_type] < 3
        curr_bb[curr_residue_type == PROLINE_INDEX, :7] = True
        prev_bb = _RIGID_GROUPS_DEP[prev_residue_type] < 3
        bb_pair = prev_bb[:, :, None] & curr_bb[:, None, :]
        mask[y] = mask[y] & (~bb_pair)
        #
        mask[ssbond, ATOM_INDEX_CYS_CB:, ATOM_INDEX_CYS_CB:] = False
        mask[ssbond, ATOM_INDEX_CYS_SG, ATOM_INDEX_CA] = False
        mask[ssbond, ATOM_INDEX_CA, ATOM_INDEX_CYS_SG] = False
        #
        dr = _R[j][:, :, None] - _R[i][:, None, :]
        dist = v_size(dr)
        #
        epsilon_i = data.ndata["atomic_radius"][i, :, 0, 0]
        epsilon_j = data.ndata["atomic_radius"][j, :, 0, 0]
        epsilon = torch.sqrt(epsilon_j[:, :, None] * epsilon_i[:, None, :]) * mask
        #
        radius_i = data.ndata["atomic_radius"][i, :, 0, 1]
        radius_j = data.ndata["atomic_radius"][j, :, 0, 1]
        radius_sum = radius_j[:, :, None] + radius_i[:, None, :]
        radius_sum = radius_sum * vdw_scale
        #
        x = -torch.clamp(dist - radius_sum, max=0.0)
        energy_ij = 10.0 * (epsilon * torch.pow(x, 2))
        energy_ij = torch.clamp(energy_ij.sum(dim=(1, 2)) - energy_clamp, min=0.0)
        #
        energy = energy + energy_ij.sum()
        #
    energy = energy / R.size(0)
    return energy


def loss_f_torsion_energy(
    batch: dgl.DGLGraph,
    R: torch.Tensor,
    ss: torch.Tensor,
    TORSION_PARs,
    energy_clamp=0.0,
):
    residue_type = batch.ndata["residue_type"]
    n_residue = residue_type.size(0)
    par = TORSION_PARs[0][ss, residue_type]
    atom_index = TORSION_PARs[1][residue_type]
    #
    r = torch.take_along_dim(R, atom_index.view(n_residue, -1, 1), 1).view(n_residue, -1, 4, 3)
    t_ang = torsion_angle(r)
    #
    t_ang = (t_ang[..., None] + par[..., 3]) * par[..., 1] - par[..., 2]
    energy = (par[..., 0] * (1.0 + torch.cos(t_ang))).sum(-1) - par[..., 0, 4]
    energy = torch.clamp(energy - energy_clamp, min=0.0)
    return torch.sum(energy) / n_residue


def loss_f_ss(batch: dgl.DGLGraph, ss0: torch.Tensor):
    loss = torch.nn.functional.cross_entropy(ss0, batch.ndata["ss"], reduction="mean")
    return loss


def find_atomic_clash(
    batch: dgl.DGLGraph, R: torch.Tensor, RIGID_OPs, vdw_scale=1.0, energy_clamp=0.0
):
    # CAUTION: this function finds atomic clashes for EDGES
    # it evaluates ONLY for edges, so it CANNOT detect clashes between nodes if they are not connected
    #
    _RIGID_GROUPS_DEP = RIGID_OPs[1][1]

    feat = []
    for batch_index in range(batch.batch_size):
        data = dgl.slice_batch(batch, batch_index, store_ids=True)
        _R = R[data.ndata["_ID"]]
        i, j = data.edges()
        #
        mask_i = data.ndata["output_atom_mask"][i] > 0.0
        mask_j = data.ndata["output_atom_mask"][j] > 0.0
        mask = mask_j[:, :, None] & mask_i[:, None, :]
        #
        x = i == j + 1
        curr_residue_type = data.ndata["residue_type"][i[x]]
        prev_residue_type = data.ndata["residue_type"][j[x]]
        curr_bb = _RIGID_GROUPS_DEP[curr_residue_type] < 3
        curr_bb[curr_residue_type == PROLINE_INDEX, :7] = True
        prev_bb = _RIGID_GROUPS_DEP[prev_residue_type] < 3
        bb_pair = prev_bb[:, :, None] & curr_bb[:, None, :]
        mask[x] = mask[x] & (~bb_pair)
        #
        x = i + 1 == j
        curr_residue_type = data.ndata["residue_type"][i[x]]
        next_residue_type = data.ndata["residue_type"][j[x]]
        curr_bb = _RIGID_GROUPS_DEP[curr_residue_type] < 3
        next_bb = _RIGID_GROUPS_DEP[next_residue_type] < 3
        next_bb[next_residue_type == PROLINE_INDEX, :7] = True
        bb_pair = next_bb[:, :, None] & curr_bb[:, None, :]
        mask[x] = mask[x] & (~bb_pair)
        #
        ssbond = data.edata["edge_feat_0"][:, 1, 0] == 1.0
        mask[ssbond, ATOM_INDEX_CYS_CB:, ATOM_INDEX_CYS_CB:] = False
        mask[ssbond, ATOM_INDEX_CYS_SG, ATOM_INDEX_CA] = False
        mask[ssbond, ATOM_INDEX_CA, ATOM_INDEX_CYS_SG] = False
        mask = mask.type(torch.float)
        #
        dr = _R[j][:, :, None] - _R[i][:, None, :]
        dist = v_size(dr)
        #
        epsilon_i = data.ndata["atomic_radius"][i, :, 0, 0]
        epsilon_j = data.ndata["atomic_radius"][j, :, 0, 0]
        epsilon = torch.sqrt(epsilon_j[:, :, None] * epsilon_i[:, None, :]) * mask
        #
        radius_i = data.ndata["atomic_radius"][i, :, 0, 1]
        radius_j = data.ndata["atomic_radius"][j, :, 0, 1]
        radius_sum = radius_j[:, :, None] + radius_i[:, None, :]
        radius_sum = radius_sum * vdw_scale
        #
        x = -torch.clamp(dist - radius_sum, max=0.0)
        energy_ij = 10.0 * (epsilon * torch.pow(x, 2)).sum(dim=(1, 2))
        energy_ij = torch.clamp(energy_ij - energy_clamp, min=0.0)
        feat.append(energy_ij)

    feat = torch.cat(feat, dim=0)
    return feat


class CoarseGrainedGeometryEnergy(object):
    def __init__(self, cg_model_name, device, use_aa_specific=False):
        self.cg_model_name = cg_model_name
        if cg_model_name == "ResidueBasedModel":
            self.use_aa_specific = True
        else:
            self.use_aa_specific = use_aa_specific
        self.set_param(device)

    def set_param(self, device):
        if self.cg_model_name == "CalphaBasedModel":
            data_fn = DATA_HOME / "calpha_geometry_params.dat"
        elif self.cg_model_name == "ResidueBasedModel":
            data_fn = DATA_HOME / "residue_geometry_params.dat"
        else:
            raise ValueError(self.cg_model_name)
        #
        self.angle_aa_map = torch.zeros((MAX_RESIDUE_TYPE, MAX_RESIDUE_TYPE), dtype=torch.long)
        for i, aa_i in enumerate(AMINO_ACID_s):
            ii = {"PRO": 1, "GLY": 2}.get(aa_i, 0)
            for j, aa_j in enumerate(AMINO_ACID_s):
                jj = {"PRO": 1, "GLY": 2}.get(aa_j, 0)
                self.angle_aa_map[i, j] = ii * 3 + jj
        #
        with open(data_fn) as fp:
            self.b_len0 = torch.zeros(
                (MAX_RESIDUE_TYPE, MAX_RESIDUE_TYPE, 2), dtype=DTYPE, device=device
            )
            self.b_ang0 = torch.zeros((MAX_RESIDUE_TYPE, 9, 2), dtype=DTYPE, device=device)
            self.vdw = torch.zeros((MAX_RESIDUE_TYPE, MAX_RESIDUE_TYPE), dtype=DTYPE, device=device)
            #
            for line in fp:
                x = line.strip().split()
                if x[0] == "BOND_LENGTH":
                    aa0, aa1 = x[1], x[2]
                    p = torch.as_tensor([float(x[3]), float(x[4])])
                    if aa0 == "ANY":
                        self.b_len0[:, :] = p[None, None, :]
                    elif self.use_aa_specific:
                        i = AMINO_ACID_s.index(aa0)
                        j = AMINO_ACID_s.index(aa1)
                        self.b_len0[i, j] = p
                elif x[0] == "BOND_ANGLE":
                    aa0, aa1, aa2 = x[1], x[2], x[3]
                    p = torch.as_tensor([float(x[4]), float(x[5])])
                    if aa0 == "ANY":
                        self.b_ang0[:, :] = p[None, None, :]
                    elif self.use_aa_specific:
                        i = AMINO_ACID_s.index(aa0)
                        j = ["XXX", "PRO", "GLY"].index(aa1)
                        k = ["XXX", "PRO", "GLY"].index(aa2)
                        jk = j * 3 + k
                        self.b_ang0[i, jk] = p
                else:
                    i = AMINO_ACID_s.index(x[1])
                    j = AMINO_ACID_s.index(x[2])
                    self.vdw[i, j] = float(x[3])

    def eval(self, batch):
        return self.eval_bonded(batch) + self.eval_vdw(batch)

    def eval_bonded(self, batch, weight=[1.0, 1.0]):
        r_cg = batch.ndata["pos"]
        residue_type = batch.ndata["residue_type"]
        #
        bonded = batch.ndata["continuous"][1:]
        n_bonded = torch.sum(bonded)
        b_len0 = self.b_len0[residue_type[1:], residue_type[:-1]]
        v1 = r_cg[1:] - r_cg[:-1]
        d = (v_size(v1) - b_len0[:, 0]) / b_len0[:, 1]
        bond_energy = torch.sum(torch.square(d) * bonded)
        #
        angled = bonded[1:] * bonded[:-1]
        n_angled = torch.sum(angled)
        angle_type = self.angle_aa_map[residue_type[:-2], residue_type[2:]]
        b_ang0 = self.b_ang0[residue_type[1:-1], angle_type]
        v1 = v_norm(v1)
        v0 = -v1
        angle = acos_safe(inner_product(v0[:-1], v1[1:]))
        angle = (angle - b_ang0[:, 0]) / b_ang0[:, 1]
        #
        angle_energy = torch.square(angle)
        angle_energy = torch.sum(torch.square(angle) * angled)
        #
        return bond_energy * weight[0] + angle_energy * weight[1]

    def eval_vdw(self, batch):
        r_cg = batch.ndata["pos"]
        #
        g = dgl.radius_graph(r_cg, 0.5, self_loop=False)
        edges = g.edges()
        sequence_separation = edges[1] - edges[0]
        valid = sequence_separation > 2
        edges = (edges[0][valid], edges[1][valid])
        #
        dij = v_size(r_cg[edges[0]] - r_cg[edges[1]])
        #
        index = batch.ndata["residue_type"]
        i = index[edges[0]]
        j = index[edges[1]]
        vdw_sum = self.vdw[i, j]
        #
        x = -torch.clamp(dij - vdw_sum, max=0.0)
        energy = torch.square(x).sum()
        return energy


def test():
    import time
    from libconfig import BASE
    import libcg
    import functools
    from libdata import PDBset, create_trajectory_from_batch

    base_dir = BASE / "pdb.processed"
    pdblist = base_dir / "loss_test"
    cg_model = libcg.CalphaBasedModel
    #
    train_set = PDBset(
        base_dir,
        pdblist,
        cg_model,
        radius=0.8,
    )
    train_loader = dgl.dataloading.GraphDataLoader(
        train_set, batch_size=2, shuffle=False, num_workers=1
    )
    cuda = torch.cuda.device_count() > 0
    batch = next(iter(train_loader))
    if cuda:
        batch = batch.to("cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #
    native = dgl.slice_batch(batch, 0)
    model = dgl.slice_batch(batch, 1)
    #
    R_ref = native.ndata["output_xyz"].clone()
    bb_ref = native.ndata["correct_bb"].clone()
    R_model = model.ndata["output_xyz"].clone()
    bb_model = model.ndata["correct_bb"].clone()

    from residue_constants import (
        RIGID_TRANSFORMS_TENSOR,
        RIGID_TRANSFORMS_DEP,
        RIGID_GROUPS_TENSOR,
        RIGID_GROUPS_DEP,
        TORSION_ENERGY_TENSOR,
        TORSION_ENERGY_DEP,
    )

    RIGID_OPs = (
        (RIGID_TRANSFORMS_TENSOR.to(device), RIGID_GROUPS_TENSOR.to(device)),
        (RIGID_TRANSFORMS_DEP.to(device), RIGID_GROUPS_DEP.to(device)),
    )
    TORSION_PARs = (TORSION_ENERGY_TENSOR.to(device), TORSION_ENERGY_DEP.to(device))
    #
    import time


if __name__ == "__main__":
    test()
