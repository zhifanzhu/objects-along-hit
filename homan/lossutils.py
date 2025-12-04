from collections import namedtuple
from typing import List
import numpy as np
import torch

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
# from homan.interactions import contactloss, scenesdf
from homan.interactions import scenesdf

from libzhifan.numeric import gather_ext


# MANO_CLOSED_FACES = np.array(
#     trimesh.load("extra_data/mano/closed_fmano.obj", process=False).faces)
MANO_CLOSED_FACES = np.load("homan/local_data/closed_fmano.npy")


def compute_pca_loss(mano_pca_comps):
    return {"pca_mean": (mano_pca_comps**2).mean()}


def compute_collision_loss(verts_hand,
                           verts_object,
                           faces_object):
    """
    Args:
        verts_hand: (B, Vh, 3)
        verts_object: (B, Vo, 3)

    Returns:
        sdf_loss: (B,)
    """
    if torch.isnan(verts_object.max()) or torch.isnan(verts_hand.max()) or \
        torch.isnan(verts_object.min()) or torch.isnan(verts_hand.min()):
        invalid = verts_object.new_ones(verts_object.shape[0]) * 1e6
        return invalid
    hand_nb = verts_hand.shape[0] // verts_object.shape[0]
    mano_faces = faces_object[0].new(MANO_CLOSED_FACES)
    if hand_nb > 1:
        mano_faces = faces_object[0].new(MANO_CLOSED_FACES[:, ::-1].copy())
        sdfl = scenesdf.SDFSceneLoss(
            [mano_faces, mano_faces, faces_object[0]])
        hand_verts = [
            verts_hand[hand_idx::2] for hand_idx in range(hand_nb)
        ]
        sdf_loss, sdf_meta = sdfl(hand_verts + [verts_object])
    else:
        sdfl = scenesdf.SDFSceneLoss([mano_faces, faces_object[0]])
        sdf_loss, sdf_meta = sdfl([verts_hand, verts_object])
    return sdf_loss


def compute_intrinsic_scale_prior(intrinsic_scales, intrinsic_mean):
    return torch.sum(
        (intrinsic_scales - intrinsic_mean)**2) / intrinsic_scales.shape[0]


def compute_contact_loss(verts_hand_b, verts_object_b, faces_object,
                         faces_hand):
    """
    Args:
        verts_hand_b: (B, V1, 3)
        verts_object_b: (B, V2, 3)
        faces_hand: (B, F1, 3)
        faces_object: (B, F2, 3)
    """
    hand_nb = verts_hand_b.shape[0] // verts_object_b.shape[0]
    faces_hand_closed = faces_hand.new(MANO_CLOSED_FACES).unsqueeze(0)
    if hand_nb > 1:
        missed_losses = []
        contact_losses = []
        for hand_idx in range(hand_nb):
            hand_verts = verts_hand_b[hand_idx::hand_nb]
            missed_loss, contact_loss, _, _ = contactloss.compute_contact_loss(
                hand_verts, faces_hand_closed, verts_object_b, faces_object)
            missed_losses.append(missed_loss)
            contact_losses.append(contact_loss)
        missed_loss = torch.stack(missed_losses).mean()
        contact_loss = torch.stack(contact_losses).mean()
    else:
        missed_loss, contact_loss, _, _ = contactloss.compute_contact_loss(
            verts_hand_b, faces_hand_closed, verts_object_b, faces_object)
    return {"contact": missed_loss + contact_loss}


def compute_chamfer_distance(verts_a, verts_b, batch_reduction='sum') -> torch.Tensor:
    """
    Args:
        verts_a: (B, Va, 3)
        verts_b: (B, Vb, 3)

    Returns:
        dist: scalar
    """
    dist, _ = chamfer_distance(verts_a, verts_b, batch_reduction=batch_reduction)
    return dist


def compute_nearest_dist(p1: torch.Tensor, 
                         p2: torch.Tensor,
                         k1=1,
                         k2=1,
                         ret_index=False) -> torch.Tensor:
    """ Compute squared minimum distance from `p_from` to `p_to`.
        loss = \sum_i d_iJ, i in I;
        d_iJ = \sum_j d_iJ, j in J,
        d_ij = | vi - vj |^2
        where I = {i : d_iJ are top k_from smallest  in d_iJ }
              J = {j : d_ij are top k_to smallest in dij }

    Args:
        p1 : (B, Va, 3)
        p2 : (B, Vb, 3)
        k1: int
        k2: int
        ret_index: bool

    Returns:
        dist: (B,)
        - if ret_index:
            from_index: (B, k_from)
            to_index: (B, k_from, k_to)
    """
    knn_ret = knn_points(p1, p2, K=k2)
    d_iJ = knn_ret.dists.sum(dim=-1)  # (B, Va)
    topk_ret = torch.topk(d_iJ, k=k1, dim=-1, largest=False, sorted=True)
    d_IJ = topk_ret.values.sum(dim=-1) / (k1 * k2)

    if ret_index:
        from_index = topk_ret.indices
        to_index = knn_ret.idx.gather(dim=1, index=from_index.unsqueeze(-1).tile(1, 1, k2))
        return d_IJ, from_index, to_index

    return d_IJ


nearest_return_type = namedtuple("return_type", 
                                 "p1_vecs p2_vecs p_ind1 p_ind2 pn1_vecs pn2_vecs")
def find_nearest_vecs(p1: torch.Tensor,
                      p2: torch.Tensor,
                      k1=1,
                      k2=1,
                      pn1: torch.Tensor = None,
                      pn2: torch.Tensor = None):
    """ Find k1 vectors in p1, (k1 x k2) vectors in p2,
     such that they satisfy the best squared minimum distance from `p_from` to `p_to`.
     See compute_nearest_dist()

    Args:
        p1 : (B, Va, 3)
        p2 : (B, Vb, 3)
        k1: int
        k2: int
        pn1: (B, Va, ...) auxillary data to collect using nearest index, e.g. Normals
        pn2: (B, Vb, ...)

    Returns: return_type
        p1_vecs: (B, k1, 3)
        p2_vecs: (B, k1, k2, 3)
        p_ind1:   (B, k1)
        p_ind2:   (B, k1, k2)
        pn1_vecs: (B, k1, 3)
        pn2_vecs: (B, k1, k2, 3)
    """
    knn_ret = knn_points(p1, p2, K=k2)
    d_iJ = knn_ret.dists.sum(dim=-1)  # (B, Va)
    topk_ret = torch.topk(d_iJ, k=k1, dim=-1, largest=False, sorted=True)
    p_ind1 = topk_ret.indices
    p1_vecs = gather_ext(p1, p_ind1, dim=1)
    
    p2_shape = (p1.size(0), p1.size(1), p2.size(1), 3)
    p2_rep = p2.unsqueeze(1).expand(p2_shape)
    p2_vecs = gather_ext(p2_rep, knn_ret.idx, dim=2)  # For all p1, get k2
    p2_vecs = gather_ext(p2_vecs, p_ind1, dim=1)  # Then get k1

    p_ind2 = gather_ext(knn_ret.idx, p_ind1, dim=1)

    pn1_vecs, pn2_vecs = None, None
    if pn1 is not None:
        pn1_vecs = gather_ext(pn1, p_ind1, dim=1)
    if pn2 is not None:
        pn2_rep = pn2.unsqueeze(1).expand(p2_shape)
        pn2_vecs = gather_ext(pn2_rep, knn_ret.idx, dim=2)
        pn2_vecs = gather_ext(pn2_vecs, p_ind1, dim=1)

    ret = nearest_return_type(
        p1_vecs, p2_vecs, p_ind1, p_ind2, pn1_vecs, pn2_vecs)
    return ret


def compute_ordinal_depth_loss(masks:torch.Tensor, 
                               silhouettes: List[torch.Tensor], 
                               depths: List[torch.Tensor],
                               ):
    """
    Args:
        masks: (B, obj_nb, height, width), ground truth mask (spongy)
        silhouettes: [(B, height, width), ...] of len obj_nb, Rendered silhouettes
        depths: [(B, height, width), ...] of len obj_nb, Rendered depths
    """
    loss = torch.as_tensor(0.0).float().cuda()
    num_pairs = 1e-7
    # Create square mask to match square renders
    height = masks.shape[2]
    width = masks.shape[3]
    silhouettes = [silh[:, :height, :width] for silh in silhouettes]
    depths = [depth[:, :height, :width] for depth in depths]
    for i in range(len(silhouettes)):
        for j in range(len(silhouettes)):
            if i == j:
                continue
            has_pred = silhouettes[i] & silhouettes[j]
            pairs = (has_pred.sum([1, 2]) > 0).sum().item()
            if pairs == 0:
                continue
            num_pairs += pairs
            # front_i_gt: pixels which should be i (i closer), 
            #   also exclude those being rendered as background as input mask 
            #   may not align with rendered silhoueets perfectly
            front_i_gt = masks[:, i] & (~masks[:, j])
            front_j_pred = depths[j] < depths[i]  # but get j closer
            mask = front_i_gt & front_j_pred & has_pred
            if mask.sum() == 0:
                continue
            dists = torch.clamp(depths[i] - depths[j], min=0.0, max=2.0)
            loss += torch.sum(torch.log(1 + torch.exp(dists))[mask])
    loss /= num_pairs
    return {"depth": loss}


def iou_loss(pred: torch.Tensor, 
             gt: torch.Tensor, 
             post='rev',
             clamp_max=100.0) -> torch.Tensor:
    """
    Args:
        pred: (..., H, W) values in [0, 1]
        gt: (..., H, W) values in [0, 1]
    
    Returns:
        loss: (...,) in [0, inf)
    """
    union = pred + gt
    inter = pred * gt
    iou = inter.sum(dim=(-2, -1)) / ((union - inter).sum(dim=(-2, -1)) + 1e-7)
    if post == 'log':
        loss = - iou.log_()
        loss = loss.clamp_max_(clamp_max)
    elif post == 'rev':
        loss = 1.0 - iou
    return loss


def rotation_loss_v1(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """ 
    If Ra = [Ra_x; Ra_y; Ra_z],
    compute:
        cos_x = <Ra_x, Rb_x>
        loss_x = - ln((1+cos_x)/2)
        loss = loss_x + loss_y + loss_z
        
    Args:
        Ra, Rb: (B, 3, 3)
    Returns:
        loss: (B,) in [0, inf)
    """
    prod = torch.matmul(Ra.permute(0, 2, 1), Rb)
    cos_vec = prod[..., [0, 1, 2], [0, 1, 2]]
    loss = -1.0 * ((cos_vec + 1.0) / 2).log_()
    return loss.sum(1)