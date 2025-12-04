import os
import os.path as osp
import numpy as np
import torch
from potim.utils.scene_static_commons import (
    get_xy_rot, initialise_from_posed_masks
)
from potim.utils.scene_static import initialise_from_posed_masks
from einops import rearrange, repeat
from potim.defs.sim3 import Sim3


def get_R_o2w_upright_epic(cat: str, num_rots_asym: int) -> torch.Tensor:
    """
    Get the sole rotation that makes the object
    standing on surface.

    Assumption:
        - Object's Z-axis is pointing upwards. hence not a per-category rotation
        - Worlds' Z-axis is pointing upwards.
    Returns:
        R_o2w: (N, 3, 3)
            N = 4 for cup, mug, saucepan and pan
            N = 1 otherwise
    """
    rots = torch.eye(3).view(1, 3, 3)
    if cat in {'cup', 'mug', 'saucepan', 'pan'}:  # do xy-plane rotation
        rots = rearrange(
            [get_xy_rot(i*np.pi/2) for i in range(num_rots_asym)],
            'n d e -> n d e', n=num_rots_asym)

    return rots


def get_transl_offset_epic(cat: str) -> torch.Tensor:
    """ For certain cetegories, 
    the coord origin is not at the CAD's centroid, 
    we attempt to ``cancel'' those centroids before apply rotations.
    """
    has_offset = False
    transl = None
    if cat == 'pan':
        has_offset = True
        transl = torch.Tensor([-0.25031435, -0.00079351, -0.01111899]).view(1, 3)
    elif cat == 'saucepan':
        has_offset = True
        transl = torch.Tensor([-1.85283498e-01, -2.25700928e-03,  1.71012711e-04]).view(1, 3)
    return has_offset, transl


def setup_inits_o2w_multi_upright(D, 
                                 index_of_static: int,
                                 estimate_scale: bool,
                                 num_rots_asym: int) -> Sim3:
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
        raise NotImplementedError
    elif D.dataset_name == 'arctic':
        raise NotImplementedError
    elif D.dataset_name == 'epic':
        R_o2w = get_R_o2w_upright_epic(D['cat'], num_rots_asym=num_rots_asym).view(-1, 1, 3, 3)
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

    if D.dataset_name == 'hot3d':
        location_estimation_method = 'fixed_50cm'
    elif D.dataset_name == 'epic':
        location_estimation_method = 'estimate_from_masks'
        # location_estimation_method = 'fixed_50cm'
    inits_o2w = initialise_from_posed_masks(
        static_masks, c2ws, static_fx, static_fy,
        R_o2w=R_o2w, 
        location_estimation_method=location_estimation_method)
    transl = inits_o2w.t  # (N, 1, 3)
    
    has_offset, t_offset = get_transl_offset_epic(D['cat'])
    if has_offset:
        t_offset = inits_o2w.rot.float() @ t_offset.view(1, 1, 3, 1)
        t_offset = t_offset.view(-1, 1, 3)
        transl = transl - t_offset

    # Estimate scale
    assert not estimate_scale
    if estimate_scale:
        raise NotImplementedError
        # T_o2c = c2ws @ inits_o2w.to_matrix()
        # static_bboxes = D.obj.bboxes[static_inds]
        # vo_obj = D.obj.verts.view(1, 1, -1, 3)
        # vo_cam = vo_obj @ T_o2c[..., :3, :3].permute(0, 1, 3, 2) + T_o2c[..., :3, [-1]].permute(0, 1, 3, 2)
        # K_static = D.global_cam.get_K()[static_inds]
        # scale_est = estimate_obj_scale(static_bboxes, vo_cam, K_static)
    else:
        scale_est = torch.ones([1, 1], device=R_o2w.device)

    inits_o2w = Sim3(inits_o2w.rot, transl, scale_est)
    return inits_o2w
