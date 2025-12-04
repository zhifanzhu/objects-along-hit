import numpy as np
import torch
from einops import rearrange, repeat
from pytorch3d.ops import knn_points
from pytorch3d.transforms import so3_rotation_angle
import open3d as o3d
from code_arctic.mesh_interaction import mesh_distance_approx
from collections import namedtuple


def pose_apply(pose, pts):
    """
    Args:
        pose: (..., 4, 4)
        pts: (V, 3)
    Returns:
        posed_pts: (..., V, 3)
    """
    while pts.ndim < pose.ndim:
        pts = pts.unsqueeze(0)
    pts = torch.transpose(pts, -1, -2)
    posed_pts = pose[..., :3, :3] @ pts + pose[..., :3, [3]]  # (..., 3, V)
    posed_pts = torch.transpose(posed_pts, -1, -2)  # (..., V, 3)
    return posed_pts

def compute_indices_ious(inds_list):
    """
        inds_list: list of N sets
        Returns:
            iou: (4, )
    """
    N = len(inds_list)
    ious = np.empty([N, N], dtype=np.float32)
    for i in range(N):
        if len(inds_list[i]) == 0:
            ious[i, :] = 0
            ious[:, i] = 0
            continue
        for j in range(i, N):
            ind_i = inds_list[i]
            ind_j = inds_list[j]
            inter = ind_i.intersection(ind_j)
            union = ind_i.union(ind_j)
            ious[i, j] = len(inter) / (len(union) + 1e-5)
            ious[j, i] = ious[i, j]
    return ious

def get_sca(vh, fh, vo, fo, ub=0.01, use_old_sca=False):
    """ Stable Contact Area
    Args:
        vh: (T, vh, 3)
        fh: (fh, 3)
        vo: (T, vo, 3)
        fo: (fo, 3)
        [REMOVED] lb: float, lower bound of contact
        ub: float, upper bound of contact 0.01 to keep the same as mvho
        use_approx: this is to keep up with the old SCA calculation

    Returns:
        avg_sca: scalar
        min_sca: scalar
        vh_inds: a set. intersection of all T in-contact hand vertex indices
        vo_inds: a set, intersection of all T in-contact object vertex indices
    """
    retVal = namedtuple(
        'retVal', ['avg_sca', 'min_sca'])
    
    if use_old_sca:
        lb=-0.01
        ub=0.00
        dh, d2 = mesh_distance_approx(vh, vo, fh, fo)  # d2: (T, vo)
        indices_list = []
        for f in range(d2.shape[0]):
            nz = torch.nonzero((d2[f] > lb) & (d2[f] < ub)).view(-1)
            indices_list.append(set(nz.tolist()))

    else:
        # Slow Trimesh version
        # indices_list = []
        # for f in range(len(vo)):
        #     mo = SimpleMesh(vo[f], fo)
        #     dh = - trimesh.proximity.signed_distance(mo, vh[f].cpu())
        #     nz = ((dh <= ub).nonzero()[0]).tolist()
        #     indices_list.append(set(nz))

        indices_list = []
        vo_cpu = vo.cpu().numpy()
        fo_cpu = fo.cpu().numpy()
        vh_cpu = vh.cpu().numpy()
        for f in range(len(vo)):
            scene = o3d.t.geometry.RaycastingScene()
            _vo = o3d.core.Tensor(vo_cpu[f])  # will CUDA version be faster?
            _fo = o3d.core.Tensor(fo_cpu)
            _mo = o3d.t.geometry.TriangleMesh(_vo, _fo)
            scene.add_triangles(_mo)

            _vh = o3d.core.Tensor(vh_cpu[f])
            d = scene.compute_signed_distance(_vh)
            nz = set((d <= ub).nonzero()[0].numpy())
            indices_list.append(set(nz))

    sca = compute_indices_ious(indices_list)
    avg_sca = sca.mean()
    min_sca = sca.min()

    # if compute_extra_inds:
    #     vo_inds = indices_list[0]
    #     for inds in indices_list[1:]:
    #         vo_inds = vo_inds.intersection(inds)

    #     vh_ind_list = []
    #     for f in range(dh.shape[0]):
    #         nz = torch.nonzero((dh[f] > lb) & (dh[f] < ub)).view(-1)
    #         vh_ind_list.append(set(nz.tolist()))
    #     vh_inds = vh_ind_list[0]
    #     for inds in vh_ind_list[1:]:
    #         vh_inds = vh_inds.intersection(inds)
    # else:
    #     vh_inds = None
    #     vo_inds = None

    ret = retVal(avg_sca, min_sca)  # , vh_inds, vo_inds)
    return ret

def calc_symADD(pred_o2w, gt_o2w, vo, diameter, sym_transforms,
                add_thrs=(0.01, 0.02, 0.05, 0.1)):
    """ Symmetric-aware ADD.
    This borrows mssd()'s symmetric processing

    Args:
        pred_o2w: (N, T, 4, 4)
        gt_o2w: (T, 4, 4)

    Returns:
        metrics: dict
            keys:
                - symADD_0.01
                - symADD_0.05
                - symADD_0.1
                - symR_err (degree)
                - symT_err (cm)
            values: (N,)
                if N==1, return a scalar
    """
    N = pred_o2w.shape[0]
    T = pred_o2w.shape[1]
    verts_pred = pose_apply(pred_o2w, vo)  # (N, T, V, 3)
    gt_o2w = repeat(gt_o2w, 't d e -> n t d e', n=N)  # (N, T, 4, 4)

    metrics = dict()
    metrics['v2v_dist_normed'] = torch.tensor(np.inf)
    for sym in sym_transforms:
        T_sym = torch.eye(4)
        T_sym[:3, :3] = torch.from_numpy(sym["R"])
        T_sym[:3, [3]] = torch.from_numpy(sym["t"]).float()
        T_sym = repeat(T_sym, 'd e -> n t d e', n=N, t=T)
        gt_o2w_sym = gt_o2w @ T_sym
        verts_gt = pose_apply(gt_o2w_sym, vo)  # (N, T, V, 3)
        v2v_dist = torch.norm(verts_gt - verts_pred, dim=-1)  # (N, T, V)
        for add_thr in add_thrs:
            key = f'symADD_{add_thr}'
            # We compute a scalar avg_dist for the whole sequence, and compare to the threshold
            # hence the add_cls_value is binary. (hence it's called classification)
            add_cls_val = (v2v_dist.mean((-2, -1)) < add_thr * diameter / 100.).float()
            if key not in metrics:
                metrics[key] = add_cls_val
            metrics[key] = torch.maximum(metrics[key], add_cls_val)
        metrics['v2v_dist_normed'] = torch.minimum(
            metrics['v2v_dist_normed'], v2v_dist.mean((-2, -1)) / (diameter/100.))

        diff_transl = pred_o2w[..., :3, 3] - gt_o2w_sym[..., :3, 3]  # (N, T, 3)
        T_err_perframe = (diff_transl**2).sum(dim=-1).sqrt() * 100.  # (N, T)
        T_err = T_err_perframe.mean(dim=-1)  # (N,)
        key = 'symT_err'
        if key not in metrics:
            metrics[key] = T_err
        metrics[key] = torch.minimum(metrics[key], T_err)

        pred_rot = pred_o2w[..., :3, :3]  # (N, T, 3, 3)
        gt_rot = gt_o2w_sym[..., :3, :3]  # (N, T, 3, 3)
        diff_rot = torch.transpose(pred_rot, -2, -1) @ gt_rot
        err_angle = so3_rotation_angle(diff_rot.view(N*T, 3, 3)).view(N, T) * 180. / np.pi  # (N, T)
        R_err = err_angle.mean(dim=-1)  # (N,)
        key = 'symR_err'
        if key not in metrics:
            metrics[key] = R_err
        metrics[key] = torch.minimum(metrics[key], R_err)

    if N == 1:
        for k, v in metrics.items():
            metrics[k] = v.item()

    return metrics

def compute_metrics(pred_o2w, gt_o2w, vo, diameter):
    """
    Args:
        pred_o2w: (N, T, 4, 4)
        gt_o2w: (T, 4, 4)
        vo: (V, 3)
        diameter: float in cm. Will convert to mm internally.

    Returns:
        metrics: dict
            keys:
                - ADD_0.01
                - ADD_0.05
                - ADD_0.1
                - R_err (degree)
                - T_err (cm)
            values: (N,)
                if N==1, return a scalar
    """
    assert pred_o2w.ndim == 4
    N = pred_o2w.shape[0]
    T = pred_o2w.shape[1]
    gt_o2w = repeat(gt_o2w, 't d e -> n t d e', n=N)  # (N, T, 4, 4)
    verts_gt = pose_apply(gt_o2w, vo)  # (N, T, V, 3)
    verts_pred = pose_apply(pred_o2w, vo)  # (N, T, V, 3)
    v2v_dist = torch.norm(verts_gt - verts_pred, dim=-1)  # (N, T, V)
    metrics = dict()
    for add_thr in (0.01, 0.05, 0.10):
        key = f'ADD_{add_thr}'
        # We compute a scalar avg_dist for the whole sequence, and compare to the threshold
        # hence the add_cls_value is binary. (hence it's called classification)
        add_cls_val = (v2v_dist.mean((-2, -1)) < add_thr * diameter / 100.).float()
        metrics[key] = add_cls_val
    metrics['v2v_dist_normed'] = v2v_dist.mean((-2, -1)) / (diameter/100.)

    # ADD-S
    v2nn_dist = knn_points(
        rearrange(verts_pred, 'n t v d -> (n t) v d'),
        rearrange(verts_gt, 'n t v d -> (n t) v d'),
        K=1).dists
    v2nn_dist = rearrange(v2nn_dist, '(n t) v 1 -> n t v', n=N)
    for adds_thr in (0.01, 0.02, 0.05, 0.10):
        key = f"ADD-S_{adds_thr}"
        add_s_val = (v2nn_dist.mean(dim=(-2, -1)) < adds_thr * diameter / 100.).float()
        metrics[key] = add_s_val

    diff_transl = pred_o2w[..., :3, 3] - gt_o2w[..., :3, 3]  # (N, T, 3)
    T_err_perframe = (diff_transl**2).sum(dim=-1).sqrt() * 100.  # (N, T)
    T_err = T_err_perframe.mean(dim=-1)  # (N,)
    metrics['T_err'] = T_err

    pred_rot = pred_o2w[..., :3, :3]  # (N, T, 3, 3)
    gt_rot = gt_o2w[..., :3, :3]  # (N, T, 3, 3)
    diff_rot = torch.transpose(pred_rot, -2, -1) @ gt_rot
    err_angle = so3_rotation_angle(diff_rot.view(N*T, 3, 3)).view(N, T) * 180. / np.pi  # (N, T)
    R_err = err_angle.mean(dim=-1)  # (N,)
    metrics['R_err'] = R_err

    if N == 1:
        for k, v in metrics.items():
            metrics[k] = v.item()

    return metrics


def compute_metrics_wrapper(seg, potim, cfg, D, CUR, cur_segi):
    """ 
    Note:
        in single_timeline_vds, 
        the argument `CUR` is 0, while `cur_segi` is the actual segi,
            so don't remove them.
    """
    with torch.no_grad():
        abs_inds = torch.arange(seg.st, seg.ed+1)
        metrics = potim.train_loss(abs_inds, cfg.optim_mv, debug_check_nan=False).metrics_dict
        all_o2w = potim.get_obj_transform_world(abs_inds).to_matrix()  # (Np, T, 4, 4)
        """ evaluate metrics """
        if D.has_3d_gt:
            metrics3d = compute_metrics(
                all_o2w.cpu(), D.gt_o2ws[CUR], D.obj.verts, diameter=D.obj.diameter)
            sym_metrics3d = calc_symADD(
                all_o2w.cpu(), D.gt_o2ws[CUR], D.obj.verts, D.obj.diameter, D.obj.sym_transforms)
            metrics.update(metrics3d)
            metrics.update(sym_metrics3d)

        if seg.ref == 'inhand':  # evaluate SCA
            mod = potim.M[CUR]
            vh = mod.v_hand  # (N, T, V, 3)
            fh = rearrange(mod.faces_hand, '1 f d -> f d')
            mod_inds = potim.get_local_inds(abs_inds, CUR)
            vo = mod.get_verts_object(mod_inds)
            fo = mod.faces_object
            sca_metrics = {'avg_sca': [], 'min_sca': []}
            for vh_i, vo_i in zip(vh, vo):
                r = get_sca(vh_i, fh, vo_i, fo, use_old_sca=cfg.use_old_sca)
                sca_metrics['avg_sca'].append(r.avg_sca)
                sca_metrics['min_sca'].append(r.min_sca)
            metrics.update(sca_metrics)

        eval_entry = dict(
            timeline_name=D.timeline_name, segi=cur_segi, ref=D.segments[CUR].ref, **metrics)

    return eval_entry, all_o2w
