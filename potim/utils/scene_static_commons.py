import numpy as np
import torch
from potim.utils.ray_closest import n_ray_closest_point
from potim.utils.mask_functions import compute_mask_center
from potim.defs.sim3 import Sim3


def get_xy_rot(theta: float) -> torch.Tensor:
    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.asarray([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]])
    return torch.from_numpy(mat).float()


def initialise_from_posed_masks(
        masks, c2ws, fx, fy, R_o2w,
        location_estimation_method: str) -> Sim3:
    """
    Args:
        masks: list of mask
        c2ws: (T, 4, 4)
        fx: (T,)
        fy: (T,)
        rots: o2w. Initialised rotations of (N, 3, 3),
            this will be compounded with the locations
            to make Sim3.
        location_estimation_method: str
            {'fixed_50cm', 'estimate_from_masks'}

    Returns:
        o2w: Sim3 of (N, 1, 4, 4), where scale is None.
    """
    def apply_pose(pose, pts):
        """ pose: (N, 4, 4), pts: (N, 3) """
        out = pose[..., :3, :3] @ pts.view(-1, 3, 1) + pose[..., :3, [-1]]
        return out.view(-1, 3)

    def apply_pose_vec(pose, vec):
        """ pose: (N, 4, 4), vec: (N, 3) """
        out = pose[..., :3, :3] @ vec.view(-1, 3, 1)
        return out.view(-1, 3)

    xys = []
    for mask in masks:
        xy = compute_mask_center(mask)
        xys.append(xy)
    xys = torch.vstack(xys)
    xys[:, 0] /= fx
    xys[:, 1] /= fy
    _ones = xys.new_ones([len(fx), 1])
    dir_cam = torch.hstack([xys, _ones])
    dir_cam /= dir_cam.norm(dim=-1, keepdim=True)
    dir_world = apply_pose_vec(c2ws, dir_cam)
    s_world = apply_pose(c2ws, torch.zeros_like(dir_cam))

    if location_estimation_method == 'fixed_50cm':
        # Just put object away be from the last camera 50cms away (assuming static's ed is dynamic's stat)
        # TODO: add the case for the reverse direction
        p = s_world[-1] + 0.5 * dir_world[-1]
    elif location_estimation_method == 'estimate_from_masks':
        p = n_ray_closest_point(s_world.numpy(), dir_world.numpy()).closest_point  # future: resolve nan
        p = torch.from_numpy(p).to(s_world.device)
        p_to_cam_dist = (s_world.mean(dim=0) - p).norm()
        print(f"{p_to_cam_dist=}")
        is_too_close = p_to_cam_dist < 0.05 # 5cm
        if is_too_close:
            print("Warning: object too close to the camera, place to 50cm away")
            p = s_world.mean(dim=0) + 0.5 * dir_world.mean(dim=0)
        if torch.isnan(p).any():
            print("Warning: NaN in p, place to 50cm away")
            p = s_world.mean(dim=0) + 0.5 * dir_world.mean(dim=0)
    else:
        raise ValueError("Unknown location_estimation_method")

    n_rots = len(R_o2w)
    rot = R_o2w.view(n_rots, 1, 3, 3)
    transl = p.view(1, 1, 3).repeat(n_rots, 1, 1)
    transl = transl.to(rot.device)
    return Sim3(rot, transl, None)
