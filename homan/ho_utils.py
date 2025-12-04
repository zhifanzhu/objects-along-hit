import numpy as np
import torch

from libzhifan.numeric import check_shape


def batch_weakcam2persptrans(weak_cams, K, focal_scale):
    check_shape(K, (-1, 3, 3), "K")
    check_shape(weak_cams, (-1, 3), "weak_cams")
    tz = focal_scale * (K[:, 1, 1] + K[:, 0, 0]) / weak_cams[:, 0] / 2
    tx = (weak_cams[:, 1] - K[:, 0, 2]) * tz / K[:, 0, 0]
    ty = (weak_cams[:, 2] - K[:, 1, 2]) * tz / K[:, 1, 1]
    return torch.stack([tx, ty, tz], -1)


def weakcam2persptrans(weak_cam, K, focal_scale=1):
    check_shape(K, (3, 3), "K")
    check_shape(weak_cam, (3,), "weak_cams")
    tz = focal_scale * (K[1, 1] + K[0, 0]) / weak_cam[0] / 2
    tx = (weak_cam[1] - K[0, 2]) * tz / K[0, 0]
    ty = (weak_cam[2] - K[1, 2]) * tz / K[1, 1]
    return np.array([tx, ty, tz])


def compute_transformation_ortho(meshes,
                                 cams,
                                 rotations=None,
                                 intrinsic_scales=None,
                                 K=None,
                                 img=None,
                                 image_size=640):
    """
    Computes the 3D transformation from a scaled orthographic camera model.

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        cams (B x 3): Scaled orthographic camera [s, tx, ty].
        rotations (B x 3 x 3): Rotation matrices.
        intrinsic_scales (B).
        focal_length (float): Should be 2x object focal length due to scaling.
        K (B x 3 x 3)

    Returns:
        vertices (B x V x 3).
    """
    B = len(cams)
    device = cams.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.eye(3).repeat(B, 1, 1).to(device)
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)
    persp_scale = (cams[:, :1] / 2 * image_size
                   )  # scale of verts_pixel / verts_origin
    persp_trans = (cams[:, 1:] + 1 / cams[:, :1]) * persp_scale
    # perspective camera in pixel space
    orthocams_pixels = torch.cat([persp_scale, persp_trans], 1)
    # K in pixel coordinates
    K_pixels = K.clone()
    K_pixels[:, :2] = K_pixels[:, :2] * image_size
    trans = batch_weakcam2persptrans(orthocams_pixels, K_pixels,
                                              1).unsqueeze(1)  # B x 1 x 3
    verts_rot = torch.matmul(meshes, rotations)  # B x V x 3
    verts_trans = verts_rot + trans
    verts_final = intrinsic_scales.view(-1, 1, 1) * verts_trans
    return verts_final


def compute_transformation_persp(meshes: torch.Tensor,
                                 translations,
                                 rotations=None,
                                 intrinsic_scales=None) -> torch.Tensor:
    """
    Computes the 3D transformation.

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        translations (B x 1 x 3).
        rotations (B x 3 x 3) apply to col-vec
        intrinsic_scales (B).

    Returns:
        vertices (B x V x 3).
    """
    B = translations.shape[0]
    device = meshes.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotations = rotations.to(device)
    rotations = rotations.permute(0, 2, 1)  # Col => Row
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)
    return (meshes * intrinsic_scales) @ rotations + (intrinsic_scales * translations)
