""" 
Created:
    Zhifan 2022 Sep 27 
"""
import torch
import torch.nn.functional as F


def compute_face_angles(verts: torch.Tensor,
                        faces: torch.Tensor) -> torch.Tensor:
    """ compute face areas 

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        face_angles: (B, F, 3) or (F, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    bs = verts.size(0)
    nf = faces.size(0)
    vfaces = verts[:, faces]  # (B, F, 3, 3)

    v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
    e12 = v2 - v1
    e23 = v3 - v2
    e31 = v1 - v3
    l12 = e12.norm(p=2, dim=-1)
    l23 = e23.norm(p=2, dim=-1)
    l31 = e31.norm(p=2, dim=-1)
    va1 = torch.sum(e12 * -e31, dim=-1) / (l12 * l31)
    va2 = torch.sum(e23 * -e12, dim=-1) / (l23 * l12)
    va3 = torch.sum(e31 * -e23, dim=-1) / (l31 * l23)
    face_cosines = torch.cat([va1, va2, va3], dim=2).view(bs, nf, 3)
    face_angles = face_cosines.acos_()
    if ndim_old == 2:
        face_angles = face_angles.squeeze_(0)
    return face_angles


def compute_face_areas(verts: torch.Tensor,
                       faces: torch.Tensor) -> torch.Tensor:
    """ compute face areas 
    TODO: Verify

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        face_areas: (B, F) or (F)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    vfaces = verts[:, faces]  # (B, F, 3, 3)

    v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
    e12 = v2 - v1
    e31 = v1 - v3
    face_areas = torch.sum(e12 * -e31, dim=-1)  # (B, F, 1)
    face_areas = face_areas.squeeze_(2)
    if ndim_old == 2:
        face_areas = face_areas.squeeze_(0)
    return face_areas


def compute_face_normals(verts: torch.Tensor,
                         faces: torch.Tensor) -> torch.Tensor:
    """ compute face normals 
        (it doesn't matter which vertex to use as they are co-planar)
            fn[i] = e12 x e13

    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
    Returns:
        face_normals: (B, F, 3) or (F, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    vfaces = verts[:, faces]  # (B, F, 3, 3)

    v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
    e12 = v2 - v1
    e31 = v1 - v3
    vn1 = torch.cross(e12, -e31, dim=-1)  # (B, F, 1, 3)
    vn1 = F.normalize(vn1, p=2, dim=-1)
    vn1 = vn1.squeeze_(2)
    if ndim_old == 2:
        vn1 = vn1.squeeze_(0)
    return vn1


def compute_vert_normals(verts: torch.Tensor,
                         faces: torch.Tensor,
                         method: str = 'f') -> torch.Tensor:
    """
    Args:
        verts: (B, V, 3) or (V, 3)
        faces: (F, 3) LongTensor
        method: one of {'v', 'f'}
            if method == 'v', compute as 
                normed mean of all connected VERTICES,
                i.e. vn[i] = mean( eij x eik )

            elif method == 'f', compte as
                meaned weighted mean of all connected FACES, as Trimesh,
                where weights are computed according to angles,
                i.e. more flat faces contribute larger

    Returns:
        vert_normals: (B, V, 3) or (V, 3)
    """
    ndim_old = verts.ndim
    if verts.ndim == 2:
        verts = verts.unsqueeze(0)
    bs = verts.size(0)
    nf = faces.size(0)

    if method == 'v':
        vfaces = verts[:, faces]  # (B, F, 3, 3)
        v1, v2, v3 = torch.split(vfaces, 1, dim=2)  # each (B, F, 1, 3)
        e12 = v2 - v1
        e23 = v3 - v2
        e31 = v1 - v3
        vn1 = torch.cross(e12, -e31, dim=-1)  # e12 x e13
        vn2 = torch.cross(e23, -e12, dim=-1)
        vn3 = torch.cross(e31, -e23, dim=-1)
        src = torch.cat([vn1, vn2, vn3], dim=2).view(bs, nf*3, 3)
        src = F.normalize(src, p=2, dim=2)

        index = faces.view(1, nf*3, 1).expand(bs, nf*3, 3)
        vn = torch.zeros_like(verts).scatter_add_(dim=1, index=index, src=src)
    elif method == 'f':
        face_normals = compute_face_normals(verts, faces).unsqueeze_(2)  # (B, F, 1, 3)
        face_angles = compute_face_angles(verts, faces).unsqueeze_(3)  # (B, F, 3, 1)
        src = face_normals * face_angles  # (B, F, 3, 3)
        src = src.view(bs, nf*3, 3)
        index = faces.view(1, nf*3, 1).expand(bs, nf*3, 3)
        vn = torch.zeros_like(verts).scatter_add_(dim=1, index=index, src=src)
    else:
        raise ValueError(f"method {method} not understood.")

    vn = F.normalize(vn, p=2, dim=2)
    if ndim_old == 2:
        vn = vn.squeeze_(0)
    return vn