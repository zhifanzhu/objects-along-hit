import torch
import torch.nn.functional as F
from collections import namedtuple
from homan.math import avg_matrix_approx
# from einops import rearrange

def calc_front_viewing_cam(ego_c2ws: torch.Tensor,
                           dist_towards_ego_z=0.6,
                           xy_away_dist=1.8,
                           view_height=1.8):
    """
    Given multi ego camera poses, preferably focus on inhand segments,
    we want automatically obtain a view from the front.
    The lookat point is the mean ego camera location plus a distance, 60cm by default,
      towards the +Z direction of ego.

    This assume the camera pose gravity-aligned (gravity points to -Z).

    Args:
        c2ws: (N, 4, 4) camera-to-world matrix.
        xy_away_dist: distance in XY-plane.
        view_height: the height of the camera in meters. E.g. 1.8m
    
    Returns:
        view_cam_loc: (3,) camera location in world space.
        view_cam_rot: (3, 3) camera rotation in world space.
        view_cam_c2w: (4, 4) camera-to-world matrix.
    """
    Up_dir = torch.Tensor([0, 0, 1])
    ViewCamType = namedtuple('ViewCamType', ['loc', 'rot', 'c2w', 'lookat', 'eye', 'up'])
    cam_loc = ego_c2ws[:, :3, 3].mean(dim=0)
    cam_rot = avg_matrix_approx(ego_c2ws[:, :3, :3])
    X_dir = cam_rot[:3, 0]  # @ torch.Tensor([1, 0, 0])
    Z_dir = cam_rot[:3, 2]  # @ torch.Tensor([0, 0, 1])
    xy_dir = torch.zeros(3)
    xy_dir[0] = - X_dir[1]
    xy_dir[1] = X_dir[0]
    xy_dir[2] = 0.0
    lookat = cam_loc + Z_dir * dist_towards_ego_z

    view_cam_loc = cam_loc + xy_dir * xy_away_dist
    view_cam_loc[2] = view_height

    view_cam_z_dir = F.normalize(lookat - view_cam_loc, dim=0)
    view_cam_x_dir = F.normalize(torch.cross(view_cam_z_dir, Up_dir), dim=0)
    view_cam_y_dir = F.normalize(torch.cross(view_cam_z_dir, view_cam_x_dir), dim=0)
    view_cam_rot = torch.stack([view_cam_x_dir, view_cam_y_dir, view_cam_z_dir], dim=1)
    view_cam_c2w = torch.eye(4)
    view_cam_c2w[:3, :3] = view_cam_rot
    view_cam_c2w[:3, 3] = view_cam_loc

    # put open3d stuff to numpy
    lookat = lookat.detach().cpu().numpy()
    eye = view_cam_loc.detach().cpu().numpy()
    up = Up_dir.detach().cpu().numpy()
    return ViewCamType(
        view_cam_loc, view_cam_rot, view_cam_c2w,
        lookat, eye, up)


def calc_scene_viewing_cam(ego_c2w: torch.Tensor,
                           dist_towards_ego_z=0.6,
                           xy_away_dist=1.5,
                           view_height=1.8):
    """
    Given an ego camera pose, 
    we want automatically obtain a over-the-shoulder view from the back.
    The lookat point is the ego camera location plus a distance (60cm by default) towards the +Z direction of ego.

    This assume the camera pose gravity-aligned (gravity points to -Z).

    Args:
        c2w: (4, 4) camera-to-world matrix.
        xy_away_dist: distance in XY-plane.
        view_height: the height of the camera in meters. E.g. 1.8m
    
    Returns:
        view_cam_loc: (3,) camera location in world space.
        view_cam_rot: (3, 3) camera rotation in world space.
        view_cam_c2w: (4, 4) camera-to-world matrix.
    """
    Up_dir = torch.Tensor([0, 0, 1])
    ViewCamType = namedtuple('ViewCamType', ['loc', 'rot', 'c2w', 'lookat', 'eye', 'up'])
    cam_loc = ego_c2w[:3, 3]
    X_dir = ego_c2w[:3, 0]  # @ torch.Tensor([1, 0, 0])
    Z_dir = ego_c2w[:3, 2]  # @ torch.Tensor([0, 0, 1])
    xy_dir = torch.zeros(3)
    xy_dir[0] = - X_dir[1]
    xy_dir[1] = X_dir[0]
    xy_dir[2] = 0.0
    lookat = cam_loc + Z_dir * dist_towards_ego_z

    view_cam_loc = cam_loc - xy_dir * xy_away_dist
    view_cam_loc[2] = view_height

    view_cam_z_dir = F.normalize(lookat - view_cam_loc, dim=0)
    view_cam_x_dir = F.normalize(torch.cross(view_cam_z_dir, Up_dir), dim=0)
    view_cam_y_dir = F.normalize(torch.cross(view_cam_z_dir, view_cam_x_dir), dim=0)
    view_cam_rot = torch.stack([view_cam_x_dir, view_cam_y_dir, view_cam_z_dir], dim=1)
    view_cam_c2w = torch.eye(4)
    view_cam_c2w[:3, :3] = view_cam_rot
    view_cam_c2w[:3, 3] = view_cam_loc

    # put open3d stuff to numpy
    lookat = lookat.detach().cpu().numpy()
    eye = view_cam_loc.detach().cpu().numpy()
    up = Up_dir.detach().cpu().numpy()
    return ViewCamType(
        view_cam_loc, view_cam_rot, view_cam_c2w,
        lookat, eye, up)