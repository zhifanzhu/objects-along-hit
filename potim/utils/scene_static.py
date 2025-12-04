import os
import os.path as osp
import os.path as osp
import numpy as np
import glob
import torch
from pytorch3d.transforms import rotation_6d_to_matrix
from einops import rearrange, repeat
from potim.defs.sim3 import Sim3
from potim.utils.propagation_helpers import make_o2w_mapping
from potim.utils.scene_static_commons import (
    get_xy_rot,
    initialise_from_posed_masks)
from potim.utils.scene_static_epic import get_R_o2w_upright_epic


def get_R_o2w_upright_hot3d(cat: str, num_rots: int) -> torch.Tensor:
    """
    Get the sole rotation that makes the object
    standing on surface.

    Assumption:
        - Object's Y-axis is pointing upwards. hence need rotation
        - Worlds' Z-axis is pointing upwards.
    Returns:
        R_o2w: (1, 3, 3)
    """
    # rotate along x counter-clock 90 to make z-axis up
    base_rot = torch.tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=torch.float32)

    rots = rearrange(
        [get_xy_rot(2*np.pi*i / num_rots) for i in range(num_rots)],
        'n d e -> n d e', n=num_rots)
    rots = rots @ base_rot.view(1, 3, 3)

    return rots.view(num_rots, 3, 3)

def get_R_o2w_upright_arctic(cat: str) -> torch.Tensor:
    """
    Get the sole rotation that makes the object
    standing on surface.

    Assumption:
        - Object's Z-axis is pointing upwards. hence not a per-category rotation
        - Worlds' Z-axis is pointing upwards.
    Returns:
        R_o2w: (1, 3, 3)
    """
    raise NotImplementedError
    if cat == 'notebook':
        rot = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=torch.float32)
        return rot.view(1, 3, 3)
    elif cat == 'phone':
        rot = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=torch.float32)
    elif cat == 'scissors':
        rot = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=torch.float32)
    else:
        rot = torch.eye(3)
    return rot.view(1, 3, 3)



""" Source: temporal/utils.py:estimate_obj_depth()
Becareful about the order of N and T.
"""
def estimate_obj_scale(bboxes: torch.Tensor,
                       vo: torch.Tensor,
                       local_cam_mat: torch.Tensor):
    """ Find scale(s) s.t.
        sqrt( (fx*(X_diam/z))**2 + (fy*(Y_diam/z))**2 ) = bboxes_diag
    Note here z is object-to-camera.
    Further, to find best "z" so that V/z = diag,
        we use: exp{ Avg(log(V) - log(diag)) }

    Args:
        bboxes: (T, 4) xywh, T stands for Time. Target boxes to match
        vo: (N_init, T, V, 3) allow N_init verts in camera space.
        cam_mat: (T, 3, 3)

    Returns:
        estimated_scale: (N_init, 1)
    """
    local_cam_mat = local_cam_mat.to(vo.device)
    bboxes = bboxes.to(vo.device)
    diag = (bboxes[:, 2]**2 + bboxes[:, 3]**2).sqrt().unsqueeze(0)  # (1, T)

    fx = local_cam_mat[:, 0, 0].unsqueeze(0)
    fy = local_cam_mat[:, 1, 1].unsqueeze(0)
    z_avg = vo[:, :, :, 2].mean(dim=-1)  # (N_init, T)
    vo_xmax_3d = vo[:, :, :, 0].max(dim=-1).values  # (N_init, T)
    vo_xmin_3d = vo[:, :, :, 0].min(dim=-1).values
    vo_ymax_3d = vo[:, :, :, 1].max(dim=-1).values
    vo_ymin_3d = vo[:, :, :, 1].min(dim=-1).values
    vo_proj_diameter = ((fx*(vo_xmax_3d - vo_xmin_3d)/z_avg)**2 + (fy*(vo_ymax_3d - vo_ymin_3d)/z_avg)**2).sqrt()
    est_scale = diag / vo_proj_diameter  # (N_init, T)
    est_scale = est_scale.mean(dim=-1, keepdim=True)  # (N_init, 1)
    return est_scale


def setup_inits_o2w_via_inhand(D) -> Sim3:
    # extract R_o2c priors using R_o2h prior and hamer's R_h2c
    R_h2c = rotation_6d_to_matrix(D.hamers[1].rot6d)[[0]].view(1, 1, 3, 3)
    R_o2h_priors = D.inits_o2h.rot.view(-1, 1, 3, 3)
    R_o2c = R_h2c @ R_o2h_priors

    c2ws = torch.inverse(D.w2cs[0])
    static_inds = torch.arange(len(c2ws))
    static_inds = static_inds[(D.segments[0].st <= static_inds) & (static_inds <= D.segments[0].ed)]
    static_masks = D.obj.mask[static_inds, ...]
    static_fx = D.global_cam.fx[static_inds]
    static_fy = D.global_cam.fy[static_inds]

    last_index = len(c2ws) - 1
    R_o2w = c2ws[last_index, :3, :3].view(1, 1, 3, 3) @ R_o2c
    inits_o2w = initialise_from_posed_masks(
        static_masks, c2ws, static_fx, static_fy,
        R_o2w=R_o2w)

    # Estimate scale
    T_o2c = c2ws @ inits_o2w.to_matrix()
    static_bboxes = D.obj.bboxes[static_inds]
    vo_obj = D.obj.verts.view(1, 1, -1, 3)
    vo_cam = vo_obj @ T_o2c[..., :3, :3].permute(0, 1, 3, 2) + T_o2c[..., :3, [-1]].permute(0, 1, 3, 2)
    K_static = D.global_cam.get_K()[static_inds]
    scale_est = estimate_obj_scale(static_bboxes, vo_cam, K_static)

    inits_o2w = Sim3(inits_o2w.rot, inits_o2w.t, scale_est)
    return inits_o2w


def setup_inits_o2w_upright(D, 
                            index_of_static: int,
                            num_rots: int,
                            estimate_scale: bool) -> Sim3:
    """
    Required from D:
        w2cs,
        segments,
        obj.
            mask
            bboxes
            verts
        global_cam

    Returns:
        inits_o2w for one static segment.
    """
    if D.dataset_name == 'hot3d':
        R_o2w = get_R_o2w_upright_hot3d(D['cat'], num_rots).view(num_rots, 1, 3, 3)
    elif D.dataset_name == 'arctic':
        R_o2w = get_R_o2w_upright_arctic(D['cat']).view(1, 1, 3, 3)
    elif D.dataset_name == 'epic':
        raise ValueError("Use multi_upright()")
    else:
        raise ValueError("Unknown dataset")
    c2ws = torch.inverse(D.w2cs[index_of_static])

    if len(c2ws) <= 0:
        raise ValueError("No frames")
    # elif len(c2ws) == 1: # Special case of single frame, estimate z
    #     raise ValueError("Single frame should be skipped")
    #     # vo_obj = D.obj.verts.view(1, 1, -1, 3)
    #     # static_bboxes = D.obj.bboxes[static_inds]
    #     # K_static = D.global_cam.get_K()[static_inds]
    #     # depth_est = estimate_obj_depth(static_bboxes, vo_cam, K_static)
    #     # depth_est = depth_est.view(1, 1)
    #     # inits_o2w = initialise_from_posed_masks(
    #     #     static_masks, c2ws, static_fx, static_fy,
    #     #     R_o2w=R_o2w)
    # else:
    sta_seg = D.segments[index_of_static]
    static_inds = torch.arange(sta_seg.st, sta_seg.ed + 1)
    static_masks = D.obj.mask[static_inds, ...]
    static_fx = D.global_cam.fx[static_inds]
    static_fy = D.global_cam.fy[static_inds]

    if D.dataset_name == 'hot3d':
        location_estimation_method = 'fixed_50cm'
    elif D.dataset_name == 'epic':
        location_estimation_method = 'estimate_from_masks'
        # location_estimation_method = 'fixed_50cm'
    inits_o2w = initialise_from_posed_masks(
        static_masks, c2ws, static_fx, static_fy,
        R_o2w=R_o2w, 
        location_estimation_method=location_estimation_method)

    # Estimate scale
    if estimate_scale:
        T_o2c = c2ws @ inits_o2w.to_matrix()
        static_bboxes = D.obj.bboxes[static_inds]
        vo_obj = D.obj.verts.view(1, 1, -1, 3)
        vo_cam = vo_obj @ T_o2c[..., :3, :3].permute(0, 1, 3, 2) + T_o2c[..., :3, [-1]].permute(0, 1, 3, 2)
        K_static = D.global_cam.get_K()[static_inds]
        scale_est = estimate_obj_scale(static_bboxes, vo_cam, K_static)
    else:
        scale_est = torch.ones([1, 1], device=R_o2w.device)

    inits_o2w = Sim3(inits_o2w.rot, inits_o2w.t, scale_est)
    return inits_o2w


def setup_inits_o2w_priors(D, 
                           index_of_static: int,
                           estimate_scale: bool) -> Sim3:
    """
    Required from D:
        w2cs,
        segments,
        obj.
            mask
            bboxes
            verts
        global_cam

    Returns:
        inits_o2w for one static segment.
    """
    if D.dataset_name == 'hot3d':
        # R_o2w = get_R_o2w_upright_hot3d(D['cat']).view(1, 1, 3, 3)
        o2w_priors = torch.load('./weights/pose_priors/hot3d/o2w_priors_10.pth')
        R_o2w = o2w_priors[D.cat]['R_o2w'].clone().view(-1, 1, 3, 3)
    else:
        raise ValueError("Unknown dataset")
    c2ws = torch.inverse(D.w2cs[index_of_static])

    if len(c2ws) <= 0:
        raise ValueError("No frames")

    sta_seg = D.segments[index_of_static]
    static_inds = torch.arange(sta_seg.st, sta_seg.ed + 1)
    static_masks = D.obj.mask[static_inds, ...]
    static_fx = D.global_cam.fx[static_inds]
    static_fy = D.global_cam.fy[static_inds]

    inits_o2w = initialise_from_posed_masks(
        static_masks, c2ws, static_fx, static_fy,
        R_o2w=R_o2w, 
        location_estimation_method='estimate_from_masks')

    # Estimate scale
    if estimate_scale:
        T_o2c = c2ws @ inits_o2w.to_matrix()
        static_bboxes = D.obj.bboxes[static_inds]
        vo_obj = D.obj.verts.view(1, 1, -1, 3)
        vo_cam = vo_obj @ T_o2c[..., :3, :3].permute(0, 1, 3, 2) + T_o2c[..., :3, [-1]].permute(0, 1, 3, 2)
        K_static = D.global_cam.get_K()[static_inds]
        scale_est = estimate_obj_scale(static_bboxes, vo_cam, K_static)
    else:
        scale_est = torch.ones([1, 1], device=R_o2w.device)

    inits_o2w = Sim3(inits_o2w.rot, inits_o2w.t, scale_est)
    return inits_o2w


def setup_inits_o2w_extra_disk_results(D, 
                                        frames_seg: list,
                                        load_from: str, 
                                        use_transl: bool,
                                        use_scale: bool,
                                        accept_thr=6) -> Sim3:
    """ Augment "upright" with optimised R_o2w.
    This includes: 
        1. neighbor inhand-only
        2. previously saved joint_static
        3. previously optimised and saved current frame results

    A Static segments have two endpoints, hence two R_o2w.
        we find nearest inhand frame to the start and end of the segment.
        we can also supply the translation with the inhand results.

    Args:
        D: dict
        load_from: str. Path to inhand rundir.
        accept_thr: only segments within `accept_thr` frames away will be 
            taken for passing.
    """
    # List potential inhand seginds. We directly read existing pth on the disk.
    o2w_mapping = dict()
    pred_o2w_list = glob.glob(osp.join(load_from, D.timeline_name, 'pred_o2w_*.pt'))
    for pred_o2w_pt in pred_o2w_list:
        pred_o2w = torch.load(pred_o2w_pt)
        _o2w_mapping = make_o2w_mapping(pred_o2w)
        o2w_mapping.update(_o2w_mapping)
    
    # First do upright because inhand maybe empty
    assert D.dataset_name == 'epic'
    raise NotImplementedError("num_rots_asym?")
    R_o2w_upright = get_R_o2w_upright_epic(D['cat'], num_rots_asym=1).view(1, 1, 3, 3)
    c2ws = torch.inverse(D.w2cs[0])  # assuming single segment
    static_inds = torch.arange(len(c2ws))
    static_inds = static_inds[
        (D.segments[0].st <= static_inds) & (static_inds <= D.segments[0].ed)]
    static_masks = D.obj.mask[static_inds, ...]
    static_fx = D.global_cam.fx[static_inds]
    static_fy = D.global_cam.fy[static_inds]
    _, t_o2w_upright, _ = initialise_from_posed_masks(
        static_masks, c2ws, static_fx, static_fy,
        R_o2w=R_o2w_upright, location_estimation_method='estimate_from_masks')

    T_o2w = []
    if len(o2w_mapping) > 0:
        for f in [frames_seg[0], frames_seg[-1]]:
            f0 = min(o2w_mapping.keys(), key=lambda k: abs(k-f))  # nearest frame
            if abs(f0 - f) <= 6: # 6 frames away
                T_o2w.append(o2w_mapping[f0])
    if len(T_o2w) > 0:
        T_o2w = rearrange(T_o2w, 'n d e -> n 1 d e', d=4, e=4)
        R_o2w = T_o2w[..., :3, :3]  # (N, 1, 3, 3)
        t_o2w_avg = T_o2w[..., :3, -1].mean(dim=0, keepdim=True)  # (1, 3)
        R_o2w = torch.cat([R_o2w, R_o2w_upright], dim=0)
        if use_transl:
            transl = t_o2w_avg
        else:
            transl = t_o2w_upright.view(1, 1, 3)
        transl = repeat(transl, '1 1 d -> n 1 d', n=len(R_o2w))  # (N, 1, 3)
        transl = transl.to(R_o2w.device)
    else:
        R_o2w = R_o2w_upright
        transl = t_o2w_upright.view(1, 1, 3)
    inits_o2w = Sim3(R_o2w, transl, None)

    if use_scale:
        raise NotImplementedError
    else:
        # # Estimate scale
        # T_o2c = c2ws @ inits_o2w.to_matrix()
        # static_bboxes = D.obj.bboxes[static_inds]
        # vo_obj = D.obj.verts.view(1, 1, -1, 3)
        # vo_cam = vo_obj @ T_o2c[..., :3, :3].permute(0, 1, 3, 2) + T_o2c[..., :3, [-1]].permute(0, 1, 3, 2)
        # K_static = D.global_cam.get_K()[static_inds]
        # scale_est = estimate_obj_scale(static_bboxes, vo_cam, K_static)

        scale_est = torch.ones([len(inits_o2w), 1], device=inits_o2w.rot.device)

    inits_o2w = Sim3(inits_o2w.rot, inits_o2w.t, scale_est)
    return inits_o2w