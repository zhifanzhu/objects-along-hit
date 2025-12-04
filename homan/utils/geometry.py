# Copyright (c) Facebook, Inc. and its affiliates.
# Edited by: Zhifan Zhu @ University of Bristol
import math
import warnings

import torch
from torch.nn import functional as F
from pytorch3d.transforms import euler_angles_to_matrix


def rot6d_to_matrix(rot_6d):
    raise ValueError("Use pytorch3d instead")


def matrix_to_rot6d(rotmat):
    raise ValueError("Use pytorch3d instead")

def combine_verts(verts_list):
    batch_size = verts_list[0].shape[0]
    all_verts_list = [v.reshape(batch_size, -1, 3) for v in verts_list]
    verts_combined = torch.cat(all_verts_list, 1)
    return verts_combined

def combine_meshes(verts_list, faces_list):
    """
    Args:
        verts_list: [ (B, V1, 3), (B, V2, 3), ... ]
        faces_list: [ (B, F1, 3), (B, F2, 3), ... ]
    Returns:
        verts_combined: (B, V1+V2+..., 3)
        faces_combined: (B, F1+F2+..., 3)
    """
    batch_size = verts_list[0].shape[0]
    all_verts_list = [v.reshape(batch_size, -1, 3) for v in verts_list]
    verts_combined = torch.cat(all_verts_list, 1)
    num_verts = torch.as_tensor([v.shape[1] for v in all_verts_list], dtype=torch.long)
    all_faces_list = []
    for i, f in enumerate(faces_list):
        f = f + num_verts[:i].sum()
        all_faces_list.append(f)
    faces_combined = torch.cat(all_faces_list, 1)
    return verts_combined, faces_combined


def center_vertices(vertices, faces, flip_y=True):
    """
    Centroid-align vertices.

    Args:
        vertices (V x 3): Vertices.
        faces (F x 3): Faces.
        flip_y (bool): If True, flips y verts to keep with image coordinates convention.

    Returns:
        vertices, faces
    """
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces


def compute_dist_z(verts1, verts2):
    """
    Computes distance between sets of vertices only in Z-direction.

    Args:
        verts1 (V x 3).
        verts2 (V x 3).

    Returns:
        tensor
    """
    a = verts1[:, 2].min()
    b = verts1[:, 2].max()
    c = verts2[:, 2].min()
    d = verts2[:, 2].max()
    if d >= a and b >= c:
        return 0.0
    return torch.min(torch.abs(c - b), torch.abs(a - d))


def generate_rotations_o2h(rot_init: dict,
                           base_rotations,
                           device='cuda'):
    """ Generate rotations w.r.t base rotations (hand-to-camera)

    Args:
        rot_init: dict
            -generate_on: 'camera' or 'hand'
        base_rotations: apply to col-vec. hand-to-camera rotations.
            (1, 3, 3) or (num_inits, 3, 3)

    Returns:
        (rot_o2h, rot_cam): (B, 3, 3) apply to col-vec.
        if generate_on == 'object':
            rot_cam is None
    """
    generate_on = rot_init['generate_on']
    rots = generate_rotations(rot_init, device)
    if generate_on == 'camera':
        assert base_rotations.shape == (1, 3, 3) or base_rotations.shape == rots.shape
        # Assuming col-vec
        # Solution to: R_camera = R_{base to camera} @ R_{object to base}
        # is R_o2h = R_base.T @ R_world
        R_o2h = base_rotations.permute(0, 2, 1).matmul(rots)
        return R_o2h, rots
    elif generate_on == 'hand':
        return rots, None
    else:
        raise ValueError("generate_on must be 'camera' or 'object'")


def generate_rotations(rot_init: dict, device='cuda'):
    """
    Future: Add soft upright, i.e. mostly upright but allow some rotation on equator.

    Args:
        rot_init: dict

    Returns:
        rot: (B, 3, 3) apply to col-vec
    """
    method = rot_init['method']
    if method == 'spiral':
        num_sphere_pts = rot_init['num_sphere_pts']
        num_sym_rots = rot_init['num_sym_rots']
        R_o2h = spiral_rotations(
            num_sphere_pts, num_sym_rots, sym_axis=rot_init['sym_axis']).to(device)
    elif method == 'random':
        num_inits = rot_init['num_inits']
        return random_avro_rotations(num_inits, device)
    elif method == 'upright':
        # Z-axis is up direction
        num_sphere_pts = rot_init['num_sphere_pts']
        num_sym_rots = rot_init['num_sym_rots']
        sym_axis = rot_init['sym_axis']
        upright_lim = rot_init['upright_lim']
        R_upright = spiral_rotations(
            num_sphere_pts, num_sym_rots, sym_axis,
            lim_ratio=upright_lim).to(device)
        R_o2h = R_upright
        # R_o2h = upright_spiral(
        #     num_sphere_pts, num_sym_rots, to_axis=to_axis,
        #     lim_ratio=upright_lim, from_axis='y').to(device)
    else:
        raise ValueError(f"Unknown method: {method}")
    return R_o2h


def compute_random_rotations(B=10, upright=False, device='cuda'):
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.
        upright (bool): If True, samples rotations that are mostly upright. Otherwise,
            samples uniformly from rotation space.

    Returns:
        rotation_matrices (B x 3 x 3) apply to col-vec
    """
    if upright:
        return upright_random(
            B, from_axis='+y', to_axis='-z', device=device)
    else:
        return random_avro_rotations(B, device)


def upright_random(B: int, from_axis, to_axis, device='cuda'):
    # Note we need to return col-vec, hence R = [Rx; Ry; Rz]
    # https://quaternions.online/
    assert from_axis == '+y'
    # absolute_upright = torch.tensor([
    #     [1, 0, 0],
    #     [0, 0, 1],
    #     [0, -1, 0]
    # ], device=device, dtype=torch.float32).repeat(B, 1, 1)
    delta = math.pi / 6  # 30 degrees
    x = torch.FloatTensor(B, 1).uniform_(-delta, delta)
    y = torch.FloatTensor(B, 1).uniform_(-math.pi/2 - delta, -math.pi/2 + delta)
    z = torch.FloatTensor(B, 1).uniform_(-delta, delta)

    angles = torch.cat((y, x, z), 1).cuda()
    rotation_matrices = euler_angles_to_matrix(angles, "XYZ")
    return rotation_matrices


def upright_spiral(num_sphere_pts, num_sym_rots,
                   to_axis,
                   lim_ratio=0.3,
                   from_axis='z'):
    """
    Args:
        to_axis: '-z'
        lim_ratio: float in [0, 1]
    """
    R_upright = spiral_rotations(num_sphere_pts, num_sym_rots, from_axis,
                                 lim_ratio=lim_ratio)
    # if to_axis == '+x':
    #     # x -> -y, y -> x, z -> z
    #     R_axis = R_upright.new_tensor([
    #         [0, 1, 0],
    #         [-1, 0, 0],
    #         [0, 0, 1]
    #     ]).view(-1, 3, 3)
    # elif to_axis == '+y':
    #     # Trivial becaues from_axis is +y
    #     R_axis = R_upright.new_tensor([
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1]
    #     ]).view(-1, 3, 3)
    # elif to_axis == '-y':
    #     # x -> x, y -> -y, z -> -z
    #     R_axis = R_upright.new_tensor([
    #         [1, 0, 0],
    #         [0, -1, 0],
    #         [0, 0, -1]
    #     ]).view(-1, 3, 3)
    # elif to_axis == '-z':
    #     # x -> x, y -> -z, z -> y
    #     R_axis = R_upright.new_tensor([
    #         [1, 0, 0],
    #         [0, 0, 1],
    #         [0, -1, 0]
    #     ]).view(-1, 3, 3)
    if to_axis == '+z':
        # Identity
        R_axis = R_upright.new_tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).view(-1, 3, 3)
    else:
        raise ValueError(f"Unsupported to_axis: {to_axis}")

    R_final = torch.matmul(R_axis, R_upright)
    return R_final


def random_avro_rotations(B: int, device='cuda'):
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.

    Returns:
        rotation_matrices (B x 3 x 3) apply to col-vec
    """
    # Reference: J Avro. "Fast Random Rotation Matrices." (1992)
    x1, x2, x3 = torch.split(torch.rand(3 * B, device=device), B)
    tau = 2 * math.pi
    R = torch.stack(
        (  # B x 3 x 3
            torch.stack((torch.cos(tau * x1), torch.sin(
                tau * x1), torch.zeros_like(x1)), 1),
            torch.stack((-torch.sin(tau * x1), torch.cos(
                tau * x1), torch.zeros_like(x1)), 1),
            torch.stack((torch.zeros_like(x1), torch.zeros_like(x1),
                            torch.ones_like(x1)), 1),
        ),
        1,
    )
    v = torch.stack(
        (  # B x 3
            torch.cos(tau * x2) * torch.sqrt(x3),
            torch.sin(tau * x2) * torch.sqrt(x3),
            torch.sqrt(1 - x3),
        ),
        1,
    )
    identity = torch.eye(3, device=device).repeat(B, 1, 1)
    H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
    rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices


def spiral_rotations(num_sphere_pts: int,
                     num_sym_rots: int,
                     sym_axis: str,
                     lim_ratio: float = 1.0) -> torch.Tensor:
    """ Generalized spiral_rotations_z to allow for other symmetric axis.

    Args:
        See spiral_rotations_z()
        sym_axis: 'x', 'y', 'z'
            - 'x': x->z y->y z->-x
            - 'y': x->x y->z z->-y
            - 'z': default
        lim_ratio: float in [0, 1], the smaller the closer to the upright

    Returns:
        (num_sphere_pts * num_sym_rots, 3, 3) apply to col-vec
    """
    # Apply sym rotations to axes other than z, is equivalent to
    # i) axis transform (e.g. rotate x-axis to z-axis), R_axis
    # ii) apply spiral rotations, R_spiral
    # iii) apply inverse axis transform, R_axis^{-1}
    # hence for col-vec, R_final = R_axis * R_{axis transform}
    R_spiral = spiral_rotations_z(num_sphere_pts, num_sym_rots, lim_ratio)
    if sym_axis == '+x':
        R_axis = R_spiral.new_tensor([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ]).view(-1, 3, 3)
    elif sym_axis == '+y':
        R_axis = R_spiral.new_tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]).view(-1, 3, 3)
    elif sym_axis == '+z':
        R_axis = R_spiral.new_tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).view(-1, 3, 3)
    else:
        raise ValueError(f"Unknown sym_axis: {sym_axis}")
    R_final = R_axis.permute(0, 2, 1).matmul(R_spiral).matmul(R_axis)
    return R_final


def spiral_rotations_z(num_sphere_pts: int,
                       num_xy_rots: int,
                       zlim_ratio: float = 1.0) -> torch.Tensor:
    """ Deterministic algorithm.

    First distribute num_sphere_pts points, set z-axis of these points to the center;
    Then from each point on the sphere, divide the xy rotation into num_xy_rots, i.e. each of 2*pi/num_xy_rots angles.

    [Spiral Ref]: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    Args:
        num_sphere_pts: number of rotations of z-axis
        num_xy_rots: number of rotations around z-axis
        zlim_ratio: [0, 1], by limiting the range of z,
            we can achieve more upright rotations.

    Return:
        (num_sphere_pts * num_xy_rots, 3, 3) apply to col-vec
    """
    if num_sphere_pts == 1:
        warnings.warn("num_sphere_pts == 1, use identity rotation for sphere_points")
        rads = 2*math.pi / num_xy_rots * torch.arange(num_xy_rots)
        Rxy = torch.stack([
            torch.stack([torch.cos(rads), torch.sin(rads), torch.zeros_like(rads)], 1),
            torch.stack([-torch.sin(rads), torch.cos(rads), torch.zeros_like(rads)], 1),
            torch.stack([torch.zeros_like(rads), torch.zeros_like(rads), torch.ones_like(rads)], 1),
        ], 1)  # (num_xy_rots, 3, 3)

        num_rots = num_sphere_pts * num_xy_rots
        Rxy = Rxy.unsqueeze(1).tile(1, num_sphere_pts, 1, 1).view(num_rots, 3, 3)
        return Rxy
    elif num_sphere_pts == 3:
        raise ValueError("Degenerated case for 3 points, use another number")

    n = num_sphere_pts
    if n >= 600000:
        epsilon = 214
    elif n>= 400000:
        epsilon = 75
    elif n>= 11000:
        epsilon = 27
    elif n>= 890:
        epsilon = 10
    elif n>= 177:
        epsilon = 3.33
    elif n>= 24:
        epsilon = 1.33
    else:
        epsilon = 0.33

    goldenRatio = (1 + 5**0.5)/2
    i = torch.arange(0, n)
    theta = 2 * math.pi * i / goldenRatio
    phi = torch.acos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    phi = phi * zlim_ratio
    x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
    z_vecs = torch.stack([x, y, z], 1)

    up_vecs = torch.zeros_like(z_vecs)
    up_vecs[:, 1] = 1.0  # (x=0, y=1, z=0)
    y_vecs = torch.nn.functional.normalize(torch.cross(z_vecs, up_vecs), p=2, dim=1)  # (num_sphere_pts, 3)
    x_vecs = torch.nn.functional.normalize(torch.cross(y_vecs, z_vecs), p=2, dim=1)
    Rz = torch.stack([x_vecs, y_vecs, z_vecs], dim=2)

    rads = 2*math.pi / num_xy_rots * torch.arange(num_xy_rots)
    Rxy = torch.stack([
        torch.stack([torch.cos(rads), torch.sin(rads), torch.zeros_like(rads)], 1),
        torch.stack([-torch.sin(rads), torch.cos(rads), torch.zeros_like(rads)], 1),
        torch.stack([torch.zeros_like(rads), torch.zeros_like(rads), torch.ones_like(rads)], 1),
    ], 1)  # (num_xy_rots, 3, 3)

    num_rots = num_sphere_pts * num_xy_rots
    Rxy = Rxy.unsqueeze(1).tile(1, num_sphere_pts, 1, 1).view(num_rots, 3, 3)
    Rz = Rz.unsqueeze(0).tile(num_xy_rots, 1, 1, 1).view(num_rots, 3, 3)
    rot_mats = Rz.matmul(Rxy)
    return rot_mats
