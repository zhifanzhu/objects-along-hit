"""
Hand scale optimisation using E_pull and E_push,
w.r.t the Fixed object vertices in the world.
"""
from functools import reduce
import tqdm
import torch
from torch.optim import Adam
from pytorch3d.ops import knn_gather, knn_points
from homan.contact_prior import get_contact_regions
from torch_scatter import scatter_min, scatter_mean
from nnutils.mesh_utils_extra import compute_vert_normals


def loss_closeness(v_hand, 
                    v_obj, 
                    faces_object,
                    contact_regions,
                    squared_dist=False,):
    """ from loss_closeness
    Args:
        v_hand: (1, V, 3)
        v_obj: (1, V, 3)
    """
    n, t = 1, 1,
    vn_obj = compute_vert_normals(v_obj, faces=faces_object)

    ph_idx = reduce(lambda a, b: a + b, contact_regions.verts, [])
    ph = v_hand[:, ph_idx, :]  # (N*T, CONTACT, 3)
    k1 = 1
    _, idx, nn = knn_points(ph, v_obj, K=k1, return_nn=True)
    # idx: (N*T, CONTACT, k1),  nn: (N*T, CONTACT, k1, 3)
    vn_obj_nn = knn_gather(vn_obj, idx)  # (N*T, CONTACT, k1, 3)

    ph = ph.view(n*t, -1, 1, 3).expand(-1, -1, k1, -1)
    # (N*T, CONTACT, k1, 3) => (N*T, CONTACT, k1)
    prod = torch.sum((ph - nn) * vn_obj_nn, dim=-1)  # length along normal

    prod = prod**2 if squared_dist else prod.abs_()
    index = torch.cat(
        [prod.new_zeros(len(v), dtype=torch.long) + i
            for i, v in enumerate(contact_regions.verts)])

    # Use mean for k1 nearest points
    prod = prod.mean(-1)    # (N*T, CONTACT, k1) => (N*T, CONTACT)
    regions_min, _ = scatter_min(src=prod, index=index, dim=1)  # (N*T, 8)
    num_priors = 5
    regions_min = regions_min[..., :num_priors]

    reduce_type = 'avg'
    if reduce_type == 'min':
        loss = regions_min.min(dim=-1).values
    elif reduce_type == 'avg':
        loss = regions_min.mean(dim=-1)

    return loss

def loss_insideness(v_hand, v_obj,
                    faces_hand, 
                    contact_regions,
                    squared_dist=False,
                    num_nearest_points=3,
                    verbose=False,
                    debug_viz=False):
    """
    For all p in object, find nearest K points in hand prior regions,
        compute distance (inner product w/ normal) as loss at this p.
        negative indicate Wrong position.

        Loss = \Avg -1.0 * max(loss_p, 0)

    Args:
        v_obj_select: List of vertices to compute loss
        num_nearest_points: number of nearest K points in hand

    Returns:
        loss: (N*B)
    """
    k2 = num_nearest_points
    n, t = 1, 1
    v_obj_size = v_obj.size(-2)

    vn_hand = compute_vert_normals(v_hand, faces=faces_hand)
    p_obj = v_obj  # (n*t, V, 3)

    p2_idx = reduce(lambda a, b: a + b, contact_regions.verts, [])
    p2 = v_hand[:, p2_idx, :]
    vn_hand_part = vn_hand[:, p2_idx, :]  # (N*T, CONTACT, 3)

    _, idx, nn = knn_points(p_obj, p2, K=k2, return_nn=True)
    # idx: (N*T, V, k2), nn: (N*T, V, k2, 3)
    nn_normals = knn_gather(vn_hand_part, idx)  # (N*T, V, k2, 3)

    """ Reshaping """
    p1 = p_obj.view(n*t, v_obj_size, 1, 3).expand(-1, -1, k2, -1)
    nn_normals = nn_normals.view(n*t, v_obj_size, k2, 3)

    vec = (p1 - nn)  # (N*T, V, k2, 3)
    prod = (vec * nn_normals).sum(-1)  # (N*T, V, k2)
    if squared_dist:
        prod = prod**2
    score = prod.mean(-1)  # (N*T, V)
    loss = (- score.clamp_max_(0)).mean(-1)  # (N*T)
    if verbose:
        print(f"insideness loss: {loss.mean().item():.3f}")
    return loss

def optim_hand_scale(vo_cam, fo, vh_hand, fh, t_h2c,
                     scale_h=None,
                     num_iters=500, lr=1e-3):
    """
    Args:
        vo_cam: (1, V, 3)
        fo: (F, 3)
        vh_hand: (1, V, 3)
        fh: (F, 3)
        t_h2c: (1, 1, 3)
    
    Returns:
        scale_h: (1, ) cpu tensor
        final_loss: float
    """
    device = vo_cam.device
    vo_cam = vo_cam.to(device)
    fo = fo.to(device)
    vh_hand = vh_hand.to(device)
    fh = fh.to(device)
    t_h2c = t_h2c.to(device)
    if scale_h is None:
        scale_h = torch.ones([1, ], device=device, requires_grad=True)
    else:
        scale_h = scale_h.to(device).clone().detach().requires_grad_(True)
    assert vo_cam.ndim == 3 and fo.ndim == 2 and vh_hand.ndim == 3 and fh.ndim == 2 and t_h2c.ndim == 3
    # print(vo_cam.shape, fo.shape, vh_hand.shape, fh.shape, t_h2c.shape)
    # print(vo_cam.device, fo.device, vh_hand.device, fh.device, t_h2c.device)
    # print(vo_cam.dtype, fo.dtype, vh_hand.dtype, fh.dtype, t_h2c.dtype)

    contact_regions = get_contact_regions()

    optim = Adam(
        [scale_h],
        lr=lr,
        weight_decay=1e-4
    )

    for _ in (pbar := tqdm.tqdm(range(num_iters))):
        # Simple translate hand vertices from hand to cam
        vh_hand_cam = scale_h * (vh_hand + t_h2c)

        e_centroid = ((vh_hand_cam.mean(dim=1) - vo_cam.mean(dim=1))**2).sum()
        e_centroid = 0.1 * e_centroid

        e_push = loss_insideness(
            v_hand=vh_hand_cam, v_obj=vo_cam,
            faces_hand=fh, contact_regions=contact_regions,)
        e_pull = loss_closeness(
            v_hand=vh_hand_cam, v_obj=vo_cam,
            faces_object=fo, contact_regions=contact_regions,)

        loss = e_pull + e_push + e_centroid
        optim.zero_grad()
        loss.backward()
        optim.step()
        pbar.set_description(f"Loss: {loss.item():.3f}")
    scale_h = scale_h.detach().cpu()
    final_loss = loss.detach().item()
    
    return scale_h, final_loss