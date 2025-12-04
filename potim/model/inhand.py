from functools import reduce
from collections import namedtuple
from typing import List, NamedTuple, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import roma
import torch
import torch.nn as nn
import trimesh
from einops import rearrange, repeat
from libzhifan.geometry import (BatchCameraManager, CameraManager, SimpleMesh,
                                projection)
from libzhifan.geometry import visualize as geo_vis
from libzhifan.geometry import visualize_mesh
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_gather, knn_points
from pytorch3d.transforms import (
    matrix_to_rotation_6d, rotation_6d_to_matrix, so3_rotation_angle)
from torch_scatter import scatter_min, scatter_mean

from homan.contact_prior import get_contact_regions
from homan.interactions import scenesdf
from homan.lossutils import find_nearest_vecs
from homan.lite_hand_module import LiteHandModule
from nnutils.mesh_utils_extra import compute_vert_normals

from potim.defs.sim3 import Sim3
from potim.model.component_base import ComponentBase


NDOF = {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'N6': 6}


class InHand(ComponentBase):

    def __init__(self,
                 ndof: str,
                 scale_obj_getter: nn.Module,
                 rend_size: 256,
                 **kwargs):
        """
        Args:
        """
        super().__init__(**kwargs)
        self.rend_size = rend_size
        self.contact_regions = get_contact_regions()
        self.ndof = NDOF[ndof]
        self.has_3d_gt = False  # will be set by set_obj_gt_3d

        self.scale_obj_getter = scale_obj_getter
        self.allocate_obj_transform()
        self.allocate_hand()

    def register_camera(self, camintr: torch.Tensor):
        """
        camintr: (1, T, 3, 3)
        """
        self.register_buffer(
            'camintr', camintr.detach().clone().requires_grad_(False))

    def register_w2c(self, w2c: torch.Tensor):
        """
        w2c: (1, T, 4, 4)
        """
        assert len(w2c) == self.num_samples
        w2c = w2c.view(1, self.num_samples, 4, 4)
        self.register_buffer('w2c', w2c)

    def allocate_hand(self, requires_grad=False):
        """
        rotations_hand: (1, T, 6)
        translations_hand: (1, T, 3)
        v_hand (buf): (1, T, 778, 3)
        """
        N = self.num_inits
        T = self.num_samples
        I_6d = torch.tensor([[1., 0, 0, 0, 1, 0]]).view(1, 1, 6)
        self.rotations_hand = nn.Parameter(
            I_6d.tile(N, T, 1), requires_grad=requires_grad)
        self.translations_hand = nn.Parameter(
            torch.zeros([N, T, 3]), requires_grad=requires_grad)
        self.register_buffer(
            'v_hand', torch.empty([N, T, 778, 3], dtype=torch.float32)
        )
        self.register_buffer(
            'faces_hand', torch.empty([1538, 3], dtype=torch.long)
        )
        # self.ref_mask_hand = hand_data.ref_mask_hand

    def init_hand(self, v_hand_ego, T_h2c, faces_hand):
        """
        Args:
            v_hand_ego: (N, T, 778, 3)
            T_h2c: (N, T, 4, 4)

        Results:
        rotations_hand: (N, T, 6)
        translations_hand: (N, T, 3)
        v_hand: (N, T, 778, 3)
        """
        N = self.num_inits
        T = self.num_samples
        R_h2c = T_h2c[..., :3, :3]
        rot6d = matrix_to_rotation_6d(R_h2c).view(N, T, 6)
        transl = T_h2c[..., :3, -1].view(N, T, 3)
        self.v_hand.data = v_hand_ego
        self.faces_hand = faces_hand
        self.rotations_hand.data = rot6d
        self.translations_hand.data = transl

    @property
    def rot_mat_hand(self) -> torch.Tensor:
        """ (N, T, 3, 3) matrix, apply to col-vector """
        return rotation_6d_to_matrix(self.rotations_hand)

    """ Object functions """

    def allocate_obj_transform(self):
        """
        self.B_6d: (N, 1, 6)
        self.tb: (N, 1, 3)
        """
        N = self.num_inits
        T = self.num_samples
        ndof = self.ndof

        if ndof == 6:
            self.R_o2h_6d = nn.Parameter(
                torch.zeros([N, T, 6]), requires_grad=True)
            self.t_o2h = nn.Parameter(
                torch.zeros([N, T, 3]), requires_grad=True)
        else:
            I_6d = torch.tensor([[1., 0, 0, 0, 1, 0]]).view(1, 1, 6)
            self.B_6d = nn.Parameter(
                I_6d.tile(N, 1, 1), requires_grad=True)
            self.tb = nn.Parameter(
                torch.zeros([N, 1, 3]), requires_grad=True)
            if ndof > 0:
                self.F = nn.ParameterList(
                    [nn.Parameter(I_6d.clone().tile(N, 1, 1), requires_grad=True)
                    for _ in range(ndof)])
                self.tf = nn.ParameterList(
                    [nn.Parameter(torch.zeros(N, 1, 3), requires_grad=True)
                    for _ in range(ndof)])
                self.ths = nn.ParameterList(
                    [nn.Parameter(torch.zeros([N, T]), requires_grad=True)
                    for _ in range(ndof)])

    def init_obj_transform(self, inits: Sim3, in_coord: str):
        """ Set to static(0-DOF) pose throughout.
        Args:
            inits: rot6d (N, 6) transl (N, 3) scale (N, 1)
            in_coor: one of {'o2h', 'o2c'}
        """
        if in_coord == 'o2h':

            if self.ndof == 6:
                n = len(inits.rot)
                R_o2h_6d = matrix_to_rotation_6d(inits.rot)  # (1, 1, 6)
                R_o2h_6d = repeat(
                    R_o2h_6d, 'n t e -> n (repeat t) e',
                    repeat=self.num_samples)
                t_o2h = repeat(
                    inits.t.view(n, 1, 3), 'n t d -> n (repeat t) d',
                    repeat=self.num_samples)
                self.R_o2h_6d.data = R_o2h_6d.clone()
                self.t_o2h.data = t_o2h.clone()

            else:
                self.B_6d.data = matrix_to_rotation_6d(
                    inits.rot).view(self.B_6d.shape).clone()
                self.tb.data = inits.t.view(self.tb.shape).clone()
                I_6d = torch.tensor([[1., 0, 0, 0, 1, 0]]).view(1, 1, 6)
                N = self.num_inits
                for i in range(self.ndof):
                    self.F[i].data = I_6d.clone().tile(N, 1, 1)
                    self.tf[i].data = torch.zeros(N, 1, 3)
                    self.ths[i].data = torch.zeros([N, self.num_samples])

        elif in_coord == 'o2c':
            raise NotImplementedError

    def register_obj_buffer(self,
                            verts_object_og,
                            faces_object,
                            ):
        """
        Args:
            verts_object_og: (V, 3)
            faces_object: (F, 3)
        """
        N, T = self.num_inits, self.num_samples
        verts_object_og = verts_object_og.view(1, 1, -1, 3)
        self.register_buffer("verts_object_og", verts_object_og)
        """ Do not attempt to copy tensor too early, which will get OOM. """
        self.register_buffer(
            "faces_object", faces_object.long())
        self.register_buffer(
            "textures_object",
            torch.ones(faces_object.shape[0], 1, 1, 1, 3))

    def register_obj_target(self, target_masks_object: torch.Tensor):
        """
        Args:
            target_masks_object: (N*T, W, W)
        """
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())

    def set_obj_gt_3d(self, gt_obj2hand_rot, gt_obj2hand_transl, obj_diameter: float):
        """
        Args:
            gt_obj2hand_rot: (N*T, 6), note even if self.dynamic is False, this is still (N*T, 6)
            gt_obj2hand_transl: (N*T, 1, 3)
            obj_diameter: In millimeters. Used for computing ADD Success Rate
        """
        self.has_3d_gt = True
        self.register_buffer('gt_obj2hand_rot', gt_obj2hand_rot)
        self.register_buffer('gt_obj2hand_transl', gt_obj2hand_transl)
        self.obj_diameter = obj_diameter

    def _expand_obj_faces(self, n, t) -> torch.Tensor:
        """
        e.g. self._expand_obj_faces(n, t) -> shape (n*t, f, 3)
        Args:

        Returns:
            obj_faces: (n * t, F_o, 3)
        """
        num_obj_faces = self.faces_object.size(0)
        return self.faces_object.expand(n*t, num_obj_faces, 3)

    def get_ndof_inhand_pose(self, inds) -> Sim3:
        """
        1-dof formula:
            Base_o2h @ ( inv(o2phi) @ Rxy @ o2phi )

        Args:
            inds: (T,)
        Returns:
            rot_o2h: (N, T, 3, 3)
            transl_o2h: (N, T, 3)
        """
        N = self.num_inits
        T = len(inds)
        B = repeat(
            rotation_6d_to_matrix(self.B_6d),
            'n t d e -> n (repeat t) d e', repeat=T)
        tb = repeat(
            self.tb, 'n t d -> n (repeat t) d e',
            repeat=T, e=1)

        R = repeat(
            torch.eye(3, device=self.B_6d.device),
            'd e -> n t d e', n=N, t=T)
        trans = torch.zeros([N, T, 3, 1], device=self.B_6d.device)
        for i in range(self.ndof):
            F = repeat(
                rotation_6d_to_matrix(self.F[i]),  # (N, 1, 3, 3)
                'n t d e -> n (repeat t) d e', repeat=T)  # (N, T, 3, 3)
            tf = repeat(
                self.tf[i],
                'n t d -> n (repeat t) d e', repeat=T, e=1)
            ths = self.ths[i][:, inds]
            # ths = torch.tanh(ths)

            Rxy_f = self.dRxy_t(ths)
            Ft = F.permute(0, 1, 3, 2)

            R = Ft @ Rxy_f @ F @ R
            trans = Ft @ (Rxy_f @ (F @ trans.view(N, T, 3, 1) + tf) - tf)

        R_o2h = B @ R
        trans = B @ trans + tb
        trans = trans.view(N, T, 3)
        return Sim3(R_o2h, trans, None)

    def get_obj_transform_ego(self, inds, hand_space=False) -> Sim3:
        """ compose N obj pose to N*T hand poses
        Args:
            inds: (T,)

        Returns:
            rots: (N, T, 3, 3) apply to col-vec
            transl: (N, T, 3)
        """
        N = self.num_inits
        if self.ndof == 6:
            R_o2h = rotation_6d_to_matrix(self.R_o2h_6d)[:, inds, ...]
            t_o2h = self.t_o2h[:, inds, :]
        else:
            R_o2h, t_o2h, _ = self.get_ndof_inhand_pose(inds=inds)
        """ Compound T_o2c (T_obj w.r.t camera) = T_h2c x To2h_ """
        R_h2c = self.rot_mat_hand
        R_h2c = R_h2c.view(N, self.num_samples, 3, 3)[:, inds, ...]
        t_h2c = self.translations_hand
        t_h2c = t_h2c.view(N, self.num_samples, 1, 3)[:, inds, ...]
        if hand_space:
            return Sim3(R_o2h, t_o2h, None)
        R_o2c = R_h2c @ R_o2h
        t_o2c = R_h2c @ t_o2h.view(N, len(inds), 3, 1) + t_h2c.view(N, len(inds), 3, 1)
        t_o2c = t_o2c.view(N, len(inds), 3)
        return Sim3(R_o2c, t_o2c, None)

    def get_obj_transform_world(self, inds) -> Sim3:
        """
        Returns:
            R_o2w: (N, T, 3, 3)
            t_o2w: (N, T, 3)
        """
        T = len(inds)
        o2c = self.get_obj_transform_ego(inds, hand_space=False)
        o2c = o2c.to_matrix()
        w2c = self.w2c[:, inds, ...].view(1, T, 4, 4)
        c2w = w2c.inverse()
        o2w = c2w @ o2c
        return Sim3(o2w[..., :3, :3], o2w[..., :3, -1], None)
    
    def get_w2cs(self, inds) -> torch.Tensor:
        """
        Args:
            inds: (T,)

        Returns:
            w2cs: (1, T, 4, 4)
        """
        return self.w2c[:, inds, ...].view(1, len(inds), 4, 4)
    
    def get_verts_object(self, inds, hand_space=False) -> torch.Tensor:
        """
        Args:
            hand_space: bool, if False, return in camera space

        Returns:
            verts_object: (N, T, V, 3)
        """
        rots, transl, _ = self.get_obj_transform_ego(
            inds=inds, hand_space=hand_space)

        verts_obj = self.verts_object_og
        nt = self.num_inits * len(inds)
        verts_obj = verts_obj.expand(self.num_inits, len(inds), -1, -1)
        scale = self.scale_obj_getter()
        scale = scale.view(self.num_inits, 1, 1, 1).expand(
            -1, len(inds), -1, -1)
        rots_row = rots.permute(0, 1, 3, 2)
        transl = transl.view(self.num_inits, len(inds), 1, 3)
        return torch.matmul(verts_obj * scale, rots_row) + transl


class InHandImpl(InHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    """ Object functions """
    """ 3D Object """
    def get_vo_pred_gt(self):
        with torch.no_grad():
            verts_obj = self.verts_object_og.view(1, -1, 3) # (N*T, V, 3)
            N, T, V = self.num_inits, self.train_size, verts_obj.size(1)
            gt_o2h_rot_mat = rotation_6d_to_matrix(
                self.gt_obj2hand_rot)  # (N*T, 3, 3)
            gt_o2h_transl = self.gt_obj2hand_transl
            pred_o2h_rot_mat, pred_o2h_transl = self.get_full_inhand_pose(rot6d=False)
            v_obj_pred = verts_obj @ pred_o2h_rot_mat.permute(
                0, 2, 1) + pred_o2h_transl # (N*T, V, 3)
            v_obj_gt = verts_obj @ gt_o2h_rot_mat.permute(0, 2, 1) + gt_o2h_transl
        return v_obj_pred.cpu(), v_obj_gt.cpu(), pred_o2h_rot_mat.cpu(), pred_o2h_transl.cpu()

    def object_metrics_3d(self):
        """ As we are measuring the object distance against GT object,
        we can simply perform computation in object space.

        Returns:
            chamfer_dist: (N*T) in mm (note CD is x->y + y->x)
            err_angle: (N*T) in degree
            err_transl: (N*T) in mm
            add_success: (N*T) in %
            add_cls: first evaluate binary add_cls on each frame,
                then the add_cls for the sequence is the average
        """
        verts_obj = self.verts_object_og.view(1, -1, 3) # (N*T, V, 3)
        N, T, V = self.num_inits, self.train_size, verts_obj.size(1)
        gt_o2h_rot_mat = rotation_6d_to_matrix(
            self.gt_obj2hand_rot)  # (N*T, 3, 3)
        gt_o2h_transl = self.gt_obj2hand_transl
        pred_o2h_rot_mat, pred_o2h_transl = self.get_full_inhand_pose(rot6d=False)
        diff_transl = gt_o2h_transl - pred_o2h_transl
        diff_rot = pred_o2h_rot_mat.permute(0, 2, 1) @ gt_o2h_rot_mat
        err_transl = (diff_transl**2).sum(dim=(1, 2)).sqrt() * 1000.  # (N*T)
        err_angle = so3_rotation_angle(diff_rot) * 180. / np.pi  # (N*T)
        err_transl = err_transl.view(N, T)
        err_angle = err_angle.view(N, T)

        """ Average Rotation Against Self-Mean"""
        w_i = pred_o2h_rot_mat.new_ones(T) / T # Weight of each matrix, between 0 and 1
        M = torch.sum(w_i[:,None, None] * pred_o2h_rot_mat, dim=0) # 3x3 matrix
        R = roma.special_procrustes(M) # weig
        rot_dev = so3_rotation_angle(torch.matmul(pred_o2h_rot_mat, R.T.view(1, 3, 3))) * 180. / np.pi
        mean_rot_dev = rot_dev.mean()
        max_rot_dev = rot_dev.max()
        # rot_angle = so3_rotation_angle(pred_o2h_rot_mat).view(N, T)

        v_obj_pred = verts_obj @ pred_o2h_rot_mat.permute(
            0, 2, 1) + pred_o2h_transl # (N*T, V, 3)
        v_obj_gt = verts_obj @ gt_o2h_rot_mat.permute(0, 2, 1) + gt_o2h_transl

        diameter = self.obj_diameter
        chamfer_dist = chamfer_distance(
            v_obj_pred, v_obj_gt, batch_reduction=None)[0] * 1000.0
        chamfer_dist = chamfer_dist.view(N, T)
        v2v_dist = ((v_obj_gt - v_obj_pred)**2).sum(dim=2).sqrt()  # (N*T, V)
        add_005 = (v2v_dist < 0.05 * diameter).float().mean(dim=1).view(N, T)
        add_010 = (v2v_dist < 0.10 * diameter).float().mean(dim=1).view(N, T)
        add_seq_010 = (v2v_dist.view(N, T, V).mean(2) < 0.10 * diameter).float().view(N, T)
        add_cls_010 = (v2v_dist.view(N, T, V).mean((1,2)) < 0.10 * diameter).float()
        # ADD-S & ADD-S-CLS
        v2nn_dist = knn_points(v_obj_pred, v_obj_gt, K=1).dists

        add_s_002 = (v2nn_dist < 0.02 * diameter).float().mean(dim=1).view(N, T)
        add_s_cls_002 = (v2nn_dist.view(N, T, V).mean((1,2)) < 0.02 * diameter).float()

        add_s_001 = (v2nn_dist < 0.01 * diameter).float().mean(dim=1).view(N, T)
        add_s_seq_001 = (v2nn_dist.view(N, T, V).mean(2) < 0.01 * diameter).float().view(N, T)
        add_s_cls_001 = (v2nn_dist.view(N, T, V).mean((1,2)) < 0.01 * diameter).float()

        # put metric I care in the front
        out_dict = {
            'add_success_0.10': add_010,
            'add_cls_0.10': add_cls_010,
            'add_s_success_0.01': add_s_001,
            'add_s_cls_0.01': add_s_cls_001,

            'err_angle': err_angle,
            'err_transl': err_transl,
            'chamfer_dist': chamfer_dist,
            'add_success_0.05': add_005,
            'add_s_success_0.02': add_s_002,
            'add_s_cls_0.02': add_s_cls_002,

            'add_seq_0.10': add_seq_010,
            'add_s_seq_0.01': add_s_seq_001,
            # 'rot_angle': rot_angle,
            'mean_rot_dev': mean_rot_dev,
            'max_rot_dev': max_rot_dev,
        }
        return out_dict

    """ Hand-Object interaction """

    def get_sca(self):
        """ Stable Contact Area as a metric
        Returns:
            avg_sca: scalar
            mean_sca: scalar
        """
        raise NotImplementedError("use potim/utils/eval_functions.py:get_sca")

    def physical_factor(self, inds) -> torch.Tensor:
        """ We should relate 3D distance to render image size
        so they have similar magnitude.

        d_pixel = d_3d * factor
        loss = d_pixel**2

        Args:
            inds: (T,)

        Returns:
            factor : (1, T,)
        """
        fx = self.camintr[:, inds, 0, 0]
        return fx * self.rend_size

    def compute_hand_iou(self, lite_hand: LiteHandModule):
        """ Borrowing keep_mask_hand from post!
        Args:
            homan: HOForwarderV2 which stores the same images as this class
        """
        _, ious = lite_hand.loss_sil_hand(compute_iou=True, func='l2')
        return ious

    def penetration_depth(self, h2o_only=True) -> float:
        """
        Max penetration depth over all frames,
        NOTE: report in mm
        Returns:
            - hand into object (N, T)
            - object into hand (N, T)
        """
        N, T = self.num_inits, self.train_size
        f_hand = self.faces_hand[0]
        f_obj = self.faces_object
        v_hand = self.v_hand
        v_obj = self.get_verts_object()

        sdfl = scenesdf.SDFSceneLoss([f_hand, f_obj])
        sdf_loss, sdf_meta = sdfl([v_hand, v_obj])
        # max_depths = sdf_meta['dist_values'][(1, 0)].max(1)[0]
        h_to_o = sdf_meta['dist_values'][(1, 0)].max(1)[0]  # .max().item()
        o_to_h = sdf_meta['dist_values'][(0, 1)].max(1)[0]  # .max().item()
        h_to_o = (h_to_o * 1000).view(N, T)
        o_to_h = (o_to_h * 1000).view(N, T)
        if h2o_only:
            return h_to_o
        else:
            return h_to_o, o_to_h

    """ Contact regions """

    def loss_closeness(self,
                       inds,
                       v_hand=None, v_obj=None, v_obj_select=None,
                       squared_dist=False,
                       num_priors=8,
                       reduce_type='avg',
                       num_nearest_points=1,
                       contact_consistency_weight=0,
                       contact_consistency_method='fullmat_over_pairs',
                       contact_consistency_region_reduce_type='none',
                       ablation_no_closeness=False,
                       verbose=False):
        """ To bring the object closer to the hand.
        L = distance from finger tips to their nearest vertices
        average over 8(=5+3) regions.
        Options:
            (5 regions vs 8 regions) x (min vs avg)

        Args:
            v_hand: (N*T, V, 3)
            v_obj: (N*T, V, 3)
            squared_dist: whether to calc loss as squared distance
            num_priors: 5 or 8
            reduce: 'min' or 'avg'

            w_smoothing: ablation, added Sep-17 2024,
                we ask the nearest-dist vec to be smooth between frames.
            w_smoothing_allpairs: ablation, added Sep-19 2024,
                we ask the nearest-dist vec to be smooth between all pairs of frames.

        Returns:
            loss: (N, T)
        """
        k1 = num_nearest_points
        v_hand = self.v_hand if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj

        n = self.num_inits
        t = len(v_obj) // n
        vn_obj = compute_vert_normals(v_obj, faces=self.faces_object)
        if v_obj_select is not None:
            v_obj = v_obj[:, v_obj_select, :]
            vn_obj = vn_obj[:, v_obj_select, :]

        ph_idx = reduce(lambda a, b: a + b, self.contact_regions.verts, [])
        ph = v_hand[:, ph_idx, :]  # (N*T, CONTACT, 3)
        _, idx, nn = knn_points(ph, v_obj, K=k1, return_nn=True)
        # idx: (N*T, CONTACT, k1),  nn: (N*T, CONTACT, k1, 3)
        vn_obj_nn = knn_gather(vn_obj, idx)  # (N*T, CONTACT, k1, 3)

        ph = ph.view(n*t, -1, 1, 3).expand(-1, -1, k1, -1)
        # (N*T, CONTACT, k1, 3) => (N*T, CONTACT, k1)
        prod = torch.sum((ph - nn) * vn_obj_nn, dim=-1)

        if contact_consistency_weight > 0:
            _sqdist = ((ph - nn)**2).sum(dim=-1)  # (N*T, CONTACT, k1), absolute distance squared

            if contact_consistency_method == 'fullmat_over_smooth':
                _vobj = rearrange(v_obj, '(n t) vo d -> n t 1 vo d', n=n)
                _ph = rearrange(v_hand[:, ph_idx, :], '(n t) CONTACT d -> n t CONTACT 1 d', n=n)
                fullmat = ((_vobj - _ph)**2).sum(dim=-1)  # (N, T, CONTACT, VO)
                fullmat_diff = fullmat[:, 1:] - fullmat[:, :-1]  # (N, T-1, CONTACT, VO)
                l_consist = fullmat_diff.abs().mean(dim=(-2, -1))  # (N, T-1)
                l_consist = torch.cat([l_consist, l_consist.new_zeros(n, 1)], dim=1)

            elif contact_consistency_method == 'fullmat_over_pairs':
                _vobj = rearrange(v_obj, '(n t) vo d -> n t 1 vo d', n=n)
                _ph = rearrange(v_hand[:, ph_idx, :], '(n t) CONTACT d -> n t CONTACT 1 d', n=n)
                fullmat = ((_vobj - _ph)**2).sum(dim=-1)  # (N, T, CONTACT, VO)
                _fullmat_1 = rearrange(fullmat, 'n t CONTACT vo -> n t 1 CONTACT vo', n=n)
                _fullmat_2 = rearrange(fullmat, 'n t CONTACT vo -> n 1 t CONTACT vo', n=n)
                fullmat_diff_pairs = _fullmat_1 - _fullmat_2  # (N, T, T, CONTACT, k1)
                l_consist = fullmat_diff_pairs.abs().mean(dim=(-3, -2, -1))  # (N, T)

            elif contact_consistency_method == 'min_over_smooth':
                _sqdist = rearrange(_sqdist, '(n t) CONTACT k1 -> n t CONTACT k1', n=n)
                _sqdist_diff = _sqdist[:, 1:] - _sqdist[:, :-1]  # (N, T-1, CONTACT, k1)

                if contact_consistency_region_reduce_type != 'none':
                    index = torch.cat(
                        [_sqdist.new_zeros(len(v), dtype=torch.long) + i
                        for i, v in enumerate(self.contact_regions.verts)])
                    if contact_consistency_region_reduce_type == 'min':
                        _sqdist_diff, _ = scatter_min(src=_sqdist_diff, index=index, dim=-2)
                        _sqdist_diff = _sqdist_diff[..., :num_priors, :]
                    elif contact_consistency_region_reduce_type == 'avg':
                        _sqdist_diff = scatter_mean(src=_sqdist_diff, index=index, dim=-2)
                        _sqdist_diff = _sqdist_diff[..., :num_priors, :]
                # _sqdist_diff: (N, T-1, CONTACT, k1) or (N, T-1, num_priors, k1)
                l_consist = _sqdist_diff.abs().mean(dim=(2, 3))  # (N, T-1)
                l_consist = torch.cat([l_consist, l_consist.new_zeros(n, 1)], dim=1)

            elif contact_consistency_method == 'min_over_pairs':
                _sqdist = rearrange(_sqdist, '(n t) CONTACT k1 -> n t CONTACT k1', n=n)
                _sqdist_1 = rearrange(_sqdist, 'n t CONTACT k1 -> n t 1 CONTACT k1', n=n)
                _sqdist_2 = rearrange(_sqdist, 'n t CONTACT k1 -> n 1 t CONTACT k1', n=n)
                _sqdist_diff = _sqdist_1 - _sqdist_2  # (N, T, T, CONTACT, k1)

                if contact_consistency_region_reduce_type != 'none':
                    index = torch.cat(
                        [_sqdist.new_zeros(len(v), dtype=torch.long) + i
                        for i, v in enumerate(self.contact_regions.verts)])
                    if contact_consistency_region_reduce_type == 'min':
                        _sqdist_diff, _ = scatter_min(src=_sqdist_diff, index=index, dim=-2)
                        _sqdist_diff = _sqdist_diff[..., :num_priors, :]
                    elif contact_consistency_region_reduce_type == 'avg':
                        _sqdist_diff = scatter_mean(src=_sqdist_diff, index=index, dim=-2)
                        _sqdist_diff = _sqdist_diff[..., :num_priors, :]
                # _sqdist_diff: (N, T, T, CONTACT, k1) or (N, T, T, num_priors, k1)
                l_consist = _sqdist_diff.abs().mean(dim=(-3, -2, -1))  # (N, T)

        prod = prod**2 if squared_dist else prod.abs_()
        index = torch.cat(
            [prod.new_zeros(len(v), dtype=torch.long) + i
             for i, v in enumerate(self.contact_regions.verts)])

        # Use mean for k1 nearest points
        prod = prod.mean(-1)    # (N*T, CONTACT, k1) => (N*T, CONTACT)
        regions_min, _ = scatter_min(src=prod, index=index, dim=1)  # (N*T, 8)
        regions_min = regions_min[..., :num_priors]

        if reduce_type == 'min':
            loss = regions_min.min(dim=-1).values
        elif reduce_type == 'avg':
            loss = regions_min.mean(dim=-1)

        loss = loss.view(n, t)
        phy_factor = self.physical_factor(inds)
        l_close = loss * phy_factor

        if contact_consistency_weight > 0:
            # l_consist = l_consist.view(n, t) * self.physical_factor(inds) * contact_consistency_weight
            # loss = loss + l_consist
            # if ablation_no_closeness:
            #     loss = l_consist
            l_consist = l_consist.view(n, t) * self.physical_factor(inds)
        else:
            l_consist = None

        return l_close, l_consist

    def loss_insideness(self,
                        inds,
                        v_hand=None, v_obj=None, v_obj_select=None,
                        squared_dist=False,
                        num_nearest_points=3,
                        debug_viz=False):
        """ To avoind inter-penetration.
        For all p in object, find nearest K points in hand prior regions,
            compute distance (inner product w/ normal) as loss at this p.
            negative indicate Wrong position.

            Loss = \Avg -1.0 * max(loss_p, 0)

        Args:
            v_obj_select: List of vertices to compute loss
            num_nearest_points: number of nearest K points in hand

        Returns:
            loss: (N, T)
        """
        k2 = num_nearest_points

        v_hand = self.v_hand if v_hand is None else v_hand
        vn_hand = compute_vert_normals(v_hand, faces=self.faces_hand[0])
        v_obj = self.get_verts_object() if v_obj is None else v_obj
        if v_obj_select is not None:
            v_obj = v_obj[:, v_obj_select, :]

        n, v_obj_size = self.num_inits, v_obj.size(-2)
        t = len(v_obj) // n
        p_obj = v_obj  # (n*t, V, 3)
        # p_obj = v_obj.view(bsize, num_obj * v_obj_size, 3)

        p2_idx = reduce(lambda a, b: a + b, self.contact_regions.verts, [])
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
        phy_factor = self.physical_factor(inds)
        loss = loss.view(n, t)
        loss = loss * phy_factor

        # from potim.debug_utils import check_nan
        # check_nan(loss)

        if debug_viz:
            pose_idx = 0
            scene = 0
            vals = score[scene].detach().cpu().numpy()
            mhand, mobj = self.get_meshes(pose_idx, scene)
            colors = np.where(
                vals[..., None] >= 0,
                mobj.visual.vertex_colors,
                trimesh.visual.interpolate(vals, color_map='jet'))
            mobj.visual.vertex_colors = colors
            return trimesh.Scene([mhand, mobj])

        return loss

    def loss_centroid_smoothness(self, inds):
        """ Ablation: smoothness on the center of the object over time.
        returns:
            loss: (N, T)
        """
        n = self.num_inits
        v_obj = self.get_verts_object(inds)
        vo_center = v_obj.mean(dim=-2)  # (N, T, 3)
        vo_diff = vo_center[:, 1:] - vo_center[:, :-1]  # (N, T-1, 3)
        loss = vo_diff.norm(dim=-1) # (N, T-1)
        loss = torch.cat([loss, loss.new_zeros(n, 1)], dim=1)
        return loss * self.physical_factor(inds)

    def loss_verts_smoothness(self, inds):
        """ vertex-to-vertex temporal smoothness loss
        returns:
            loss: (N, T)
        """
        n = self.num_inits
        v_obj = self.get_verts_object(inds)
        v_diff = v_obj[:, 1:] - v_obj[:, :-1]  # (N, T-1, V, 3)
        loss = v_diff.norm(dim=-1).mean(dim=-1) # (N, T-1)
        loss = torch.cat([loss, loss.new_zeros(n, 1)], dim=1)
        return loss * self.physical_factor(inds)

    """ system objectives """

    def train_loss(self, inds, optim_cfg, debug_check_nan=True) -> dict:
        """
        Args:
            inds: (T,)
            cfg: `optim` section of the config
        Returns:
            loss_dict:
                - keys: 'inside', 'close'
                - values: (N, T)
        """
        retVal = namedtuple('InhandLoss', 'loss_dict loss')
        nt = self.num_inits * len(inds)
        v_hand = self.v_hand[:, inds, ...]  # (1, T, V, 3)
        v_hand = v_hand.expand(self.num_inits, -1, -1, -1)
        v_obj = self.get_verts_object(inds)
        v_hand_sqz = v_hand.reshape(nt, -1, 3)
        v_obj_sqz = v_obj.reshape(nt, -1, 3)

        loss_dict = {}
        l_inside_full = self.loss_insideness(
            inds=inds,
            v_hand=v_hand_sqz, v_obj=v_obj_sqz, v_obj_select=None,
            num_nearest_points=optim_cfg.loss.inside.num_nearest_points)
        l_inside = optim_cfg.loss.inside.weight * l_inside_full # .sum(-1)
        loss_dict['inside'] = l_inside

        l_close_full, l_consist = self.loss_closeness(
            inds=inds,
            v_hand=v_hand_sqz, v_obj=v_obj_sqz, v_obj_select=None,
            num_priors=optim_cfg.loss.close.num_priors,
            reduce_type=optim_cfg.loss.close.reduce,
            num_nearest_points=optim_cfg.loss.close.num_nearest_points,
            contact_consistency_weight=optim_cfg.loss.close.contact_consistency.weight,
            contact_consistency_method=optim_cfg.loss.close.contact_consistency.method,
            contact_consistency_region_reduce_type=optim_cfg.loss.close.contact_consistency.regional_reduce,
            ablation_no_closeness=optim_cfg.loss.close.get('ablation_no_closeness', False),
            )
        l_close = optim_cfg.loss.close.weight * l_close_full # .sum(-1)
        loss_dict['close'] = l_close
        l_consist = optim_cfg.loss.close.contact_consistency.weight * l_consist
        loss_dict['contact_consistency'] = l_consist

        if debug_check_nan:
            from potim.debug_utils import check_nan
            check_nan(l_inside)
            check_nan(l_close)
        else:
            if torch.isnan(l_inside).any():
                print(f"WARNING: cleaning l_inside nan")
                l_inside[torch.isnan(l_inside)] = 0
            if torch.isnan(l_close).any():
                print(f"WARNING: cleaning l_close nan")
                l_close[torch.isnan(l_close)] = 0

        return retVal(loss_dict=loss_dict, loss=None)

    def eval_metrics(self, lite_hand=None, unsafe=True, avg=True, to_scalar=False):
        """ Evaluate metric on ALL frames

        Args:
            avg: bool, if True will reduce over temporal dimension

        Returns: dict
            -h_iou: (N,T)
            -o_iou: (N,T)
            -min_dist: (N,T)
            -chamfer: (N,T) in mm
            -angle: (N,T)
            -transl: (N,T) in mm against GT
        """
        N, T = self.num_inits, self.train_size
        with torch.no_grad():
            v_hand = self.v_hand  # self.get_verts_hand()
            v_obj = self.get_verts_object()
            if self.has_3d_gt:
                obj_metrics_3d = self.object_metrics_3d()
            if v_hand.isnan().any() or v_obj.isnan().any():
                hious = torch.zeros([len(v_hand)]).view(N, T)
                oious = torch.zeros([len(v_hand)]).view(N, T)
                pd_h2o = 10.0 * torch.ones([len(v_hand)]).view(N, T)
                pd_o2h = 10.0 * torch.ones([len(v_hand)]).view(N, T)
                max_min_dist = 10.0 * torch.ones(self.num_inits)
                avg_sca = 0 * torch.zeros(self.num_inits)
                min_sca = 0 * torch.zeros(1)
                max_iv = 10.0 * torch.ones(1)
            else:
                if lite_hand is None:
                    hious = torch.ones([N, T]) * -1
                else:
                    hious = self.compute_hand_iou(lite_hand)
                _, oious, _ = self.forward_obj_pose_render(
                    v_obj=v_obj, loss_only=False)
                max_min_dist = self.max_min_dist(
                    v_hand=v_hand, v_obj=v_obj)
                if unsafe:
                    pd_h2o, pd_o2h = self.penetration_depth(h2o_only=False)
                    max_iv = self.get_iv()
                avg_sca, min_sca, naive_sca = self.get_sca()

        metrics = {
            'hious': hious.view(N, T),
            'oious': oious.view(N, T),
            'max_min_dist': max_min_dist,
            'avg_sca': avg_sca,
            'min_sca': min_sca,
            'naive_sca': naive_sca,
        }
        if self.has_3d_gt:
            metrics.update(obj_metrics_3d)
        if unsafe:
            metrics.update({
                'pd_h2o': pd_h2o.max(),
                'pd_o2h': pd_o2h.max(),
                'max_iv': max_iv,
            })
        if avg:
            for k, v in metrics.items():
                if v.ndim == 1:
                    metrics[k] = v.cpu()
                elif v.ndim == 2:
                    metrics[k] = v.mean(dim=1).cpu()

        if to_scalar:
            metrics = {k: v.item() for k, v in metrics.items()}
        return metrics


matplotlib.use('svg')  # seems svg renders plt faster


class InHandVis(InHandImpl):
    def __init__(self,
                 vis_rend_size=256,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.vis_rend_size = vis_rend_size

    def set_ihoi_img_patch(self, ihoi_img_patch):
        self.ihoi_img_patch = ihoi_img_patch

    def get_meshes(self, pose_idx, scene_idx, **mesh_kwargs) -> Tuple[SimpleMesh]:
        """
        Args:
            pose_idx:
            scene_idx: index of scene (timestep)
                'pose_idx * T + scene_idx' index into N*T

        Returns:
            mhand: SimpleMesh
            mobj: SimpleMesh, or None if obj_idx < 0
        """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        nt_ind = pose_idx * self.train_size + scene_idx
        with torch.no_grad():
            verts_hand = self.v_hand[nt_ind]
            mhand = SimpleMesh(
                verts_hand, self.faces_hand[nt_ind], tex_color=hand_color)
            # if pose_idx < 0:
            #     mobj = None
            # else:
            verts_obj = self.get_verts_object(**mesh_kwargs)[nt_ind]
            mobj = SimpleMesh(
                verts_obj, self.faces_object, tex_color=obj_color)
        return mhand, mobj

    def finger_with_normals(self, nt_ind,
                            regions=(0, 1, 2, 3, 4, 5, 6, 7)) -> trimesh.Scene:
        """
        Returns: a Scene with single hand, finger regions marked with normals.
        """
        hand_color = 'light_blue'
        with torch.no_grad():
            verts_hand = self.v_hand[nt_ind]
            vn = compute_vert_normals(verts_hand, self.faces_hand[nt_ind])
            mhand = SimpleMesh(
                verts_hand, self.faces_hand[nt_ind], tex_color=hand_color)

        paths = []
        for i, v_inds in enumerate(self.contact_regions.verts):
            if i not in regions:
                continue
            geo_vis.color_verts(mhand, v_inds, (255, 0, 0))

            v_parts = verts_hand[v_inds].detach().cpu().numpy()
            vn_parts = vn[v_inds].cpu().numpy()
            vec = np.column_stack(
                (v_parts, v_parts + (vn_parts * mhand.scale * .05)))
            path = trimesh.load_path(vec.reshape(-1, 2, 3))
            paths.append(path)

        return trimesh.Scene([mhand] + paths)

    def visualize_nearest_normals(self,
                                  nt_ind,
                                  display=('hand', 'obj', 'normals'),
                                  regions=5,
                                  ) -> trimesh.Scene:
        """
        Visualize each region's nearest point to the object
        Args:
            nt_ind: index into N*T
        """
        # Find nearest point indices
        k1, k2 = 1, 1
        h_paths = []
        o_paths = []
        vh_inds = []
        vo_inds = []
        with torch.no_grad():
            v_hand = self.v_hand[[nt_ind]]
            v_obj = self.get_verts_object()[[nt_ind]]
            vn_hand = compute_vert_normals(v_hand, faces=self.faces_hand[0])
            vn_obj = compute_vert_normals(v_obj, faces=self.faces_object)
            mhand = SimpleMesh(v_hand[0], self.faces_hand[nt_ind])
            mobj = SimpleMesh(v_obj[0], self.faces_object)
            p2 = v_obj

            for part in self.contact_regions.verts[:regions]:
                p1 = v_hand[:, part, :]
                pn1 = vn_hand[:, part, :]
                v1, v2, vh_ind, vo_ind, vn1, vn2 = find_nearest_vecs(
                    p1, p2, k1=k1, k2=k2, pn1=pn1, pn2=vn_obj)
                """
                To get index in the hand,
                first get index
                """
                vh_ind = vh_ind[nt_ind].squeeze().item()
                vh_ind = part[vh_ind]
                vh_inds.append(vh_ind)
                vo_inds.append(vo_ind[nt_ind].squeeze().item())

                v1 = v1[nt_ind].cpu().numpy()
                vn1 = vn1[nt_ind].cpu().numpy()
                v2 = v2.squeeze_(1)[nt_ind].cpu().numpy()  # (1, 3)
                vn2 = vn2.squeeze_(1)[nt_ind].cpu().numpy()  # (1, 3)
                vec1 = np.column_stack(
                    (v1, v1 + (vn1 * mhand.scale * 0.05)))
                vec2 = np.column_stack(
                    (v2, v2 + (vn2 * mobj.scale * 0.05)))
                vec1 = trimesh.load_path(vec1.reshape(-1, 2, 3))
                vec2 = trimesh.load_path(vec2.reshape(-1, 2, 3))
                h_paths.append(vec1)
                o_paths.append(vec2)

            geo_vis.color_verts(mhand, vh_inds, (255, 0, 0))
            geo_vis.color_verts(mobj, vo_inds, (0, 0, 255))

        scene_geoms = []
        if 'hand' in display:
            scene_geoms.append(mhand)
        if 'obj' in display:
            scene_geoms.append(mobj)
        if 'normals' in display:
            if 'hand' in display:
                scene_geoms += h_paths
            if 'obj' in display:
                scene_geoms += o_paths
        return trimesh.Scene(scene_geoms)

    def to_scene(self, pose_idx, scene_idx=-1,
                 show_axis=False, viewpoint='nr',
                 scene_indices=None, disp=0.15,
                 **mesh_kwargs):
        """ Returns a trimesh.Scene """
        if scene_idx >= 0:
            mhand, mobj = self.get_meshes(
                pose_idx=pose_idx, scene_idx=scene_idx, **mesh_kwargs)
            return visualize_mesh([mhand, mobj],
                                  show_axis=show_axis,
                                  viewpoint=viewpoint)

        """ Render all """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')
        with torch.no_grad():
            verts_hand = self.v_hand.view(
                self.num_inits, self.train_size, -1, 3)
            verts_obj = self.get_verts_object(**mesh_kwargs).view(
                self.num_inits, self.train_size, -1, 3)
            verts_hand = verts_hand[pose_idx]
            verts_obj = verts_obj[pose_idx]

        meshes = []
        scene_indices = scene_indices or range(self.train_size)
        for i, t in enumerate(scene_indices):
            mhand = SimpleMesh(
                verts_hand[t], self.faces_hand[t], tex_color=hand_color)
            mhand.apply_translation_([i * disp, 0, 0])
            mobj = SimpleMesh(
                verts_obj[t], self.faces_object, tex_color=obj_color)
            mobj.apply_translation_([i * disp, 0, 0])
            meshes.append(mhand)
            meshes.append(mobj)
        return visualize_mesh(meshes, show_axis=show_axis, viewpoint=viewpoint)

    def render_summary(self, pose_idx, scene_idx) -> np.ndarray:
        nt_ind = pose_idx * self.train_size + scene_idx
        a1 = np.uint8(self.ihoi_img_patch[nt_ind])
        mask_hand = self.ref_mask_hand[nt_ind].cpu().numpy().squeeze()
        mask_obj = self.ref_mask_object[nt_ind].cpu().numpy()
        all_mask = np.zeros_like(a1, dtype=np.float32)
        all_mask = np.where(
            mask_hand[..., None], (0, 0, 0.8), all_mask)
        all_mask = np.where(
            mask_obj[..., None], (0.6, 0, 0), all_mask)
        all_mask = np.uint8(255*all_mask)
        a2 = cv2.addWeighted(a1, 0.9, all_mask, 0.5, 1.0)
        a3 = np.uint8(self.render_scene(
            pose_idx=pose_idx, scene_idx=scene_idx)*255)
        b = np.uint8(
            255*self.render_triview(pose_idx=pose_idx, scene_idx=scene_idx))
        a = np.hstack([a3, a2, a1])
        return np.vstack([a,
                          b])

    def render_scene(self, pose_idx, scene_idx,
                     with_hand=True, overlay_gt=False,
                     bg_ihoi_patch=True,
                     **mesh_kwargs) -> np.ndarray:
        """ returns: (H, W, 3) """
        # nt_ind = pose_idx * self.train_size + scene_idx
        nt_ind = scene_idx
        if not with_hand:
            img = self.ihoi_img_patch[nt_ind] / 255
        else:
            mhand, mobj = self.get_meshes(pose_idx=pose_idx,
                                          scene_idx=scene_idx, **mesh_kwargs)
            if pose_idx < 0:
                mesh_list = [mhand]
            else:
                mesh_list = [mhand, mobj]
            if self.version == LOWDIM and mesh_kwargs.pop('with_phi', True):
                mphi = self.get_phi_tf(scene_idx=scene_idx, as_mesh=True)
                mesh_list.append(mphi)
            bg_image = self.ihoi_img_patch[nt_ind] if bg_ihoi_patch else None
            img = projection.perspective_projection_by_camera(
                mesh_list,
                CameraManager.from_nr(
                    self.camintr.detach().cpu().numpy()[nt_ind], self.vis_rend_size),
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                ),
                image=bg_image,
            )

        if overlay_gt:
            all_mask = np.zeros_like(img, dtype=np.float32)
            mask_hand = self.ref_mask_hand[nt_ind].cpu().numpy().squeeze()
            all_mask = np.where(
                mask_hand[..., None], (0, 0, 0.8), all_mask)
            # if obj_idx >= 0:
            mask_obj = self.ref_mask_object[nt_ind].cpu().numpy()
            all_mask = np.where(
                mask_obj[..., None], (0.6, 0, 0), all_mask)
            all_mask = np.uint8(255*all_mask)
            img = cv2.addWeighted(np.uint8(img*255), 0.9, all_mask, 0.5, 1.0)
        return img

    def render_triview(self, pose_idx, scene_idx,
                       views=['front', 'left', 'back'],
                       hstack=True,
                       rend_size=256,
                       **mesh_kwargs) -> np.ndarray:
        """
        Returns:
            (H, W, 3)
        """
        mhand, mobj = self.get_meshes(
            pose_idx=pose_idx, scene_idx=scene_idx, **mesh_kwargs)
        meshes = [mhand, mobj]
        if self.version == LOWDIM and mesh_kwargs.pop('with_phi', True):
            mphi = self.get_phi_tf(scene_idx=scene_idx, as_mesh=True)
            meshes.append(mphi)

        direction_map = {'front': '+z', 'left': '+x', 'back': '-z', 'right': '-x'}
        rends = []
        for view in views:
            direction = direction_map[view]
            rend = projection.project_standardized(
                meshes,
                direction=direction,
                image_size=rend_size,
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                )
            )
            rends.append(rend)
        rends = np.hstack(rends) if hstack else np.vstack(rends)
        return rends

    def render_grid_np(self, pose_idx=0, with_hand=True,
                       *args, **kwargs) -> np.ndarray:
        """ low resolution but faster """
        num_grids = min(30, len(self.ihoi_img_patch))
        cam_inds = np.linspace(
            0, len(self.ihoi_img_patch), num_grids, endpoint=False, dtype=np.int)
        num_cols = min(num_grids, 5)
        num_rows = (num_grids + num_cols - 1) // num_cols
        imgs = []
        for cam_idx in cam_inds:
            img = self.render_scene(
                pose_idx=pose_idx,
                scene_idx=cam_idx,
                with_hand=with_hand, *args, **kwargs)
            imgs.append(img)
            if cam_idx == num_grids-1:
                break

        h, w, _ = imgs[0].shape
        out = np.empty(shape=(num_rows*h, num_cols*w, 3), dtype=imgs[0].dtype)
        for row in range(num_rows):
            for col in range(num_cols):
                idx = row*num_cols+col
                if idx >= num_grids:
                    break
                out[row*h:(row+1)*h, col*w:(col+1)*w, :] = imgs[idx]

        return out

    def render_grid(self, pose_idx=0, with_hand=True,
                    figsize=(10, 10), low_reso=True,
                    return_list=False, *args, **kwargs):
        """
        Args:
            obj_idx: -1 for no object
            with_hand: whether to render hand
            figsize: (w, h)
            low_reso: call render_grid_np() if True
            return_list: return a list of images if True
        """
        if low_reso:
            out = self.render_grid_np(
                pose_idx=pose_idx, with_hand=with_hand, *args, **kwargs)
            fig, ax = plt.subplots()
            ax.imshow(out)
            plt.axis('off')
            return fig

        if return_list:
            ret = []
            for cam_idx in range(self.bsize):
                img = self.render_scene(pose_idx=pose_idx,
                                        scene_idx=cam_idx, with_hand=with_hand, *args, **kwargs)
                ret.append(img)
            return ret

        l = self.train_size
        num_cols = 5
        num_rows = (l + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols,
            sharex=True, sharey=True, figsize=figsize)
        for cam_idx, ax in enumerate(axes.flat, start=0):
            if cam_idx > l-1:
                ax.set_axis_off()
                continue
            img = self.render_scene(pose_idx=pose_idx,
                                    scene_idx=cam_idx, with_hand=with_hand, *args, **kwargs)
            ax.imshow(img)
            ax.set_axis_off()
        plt.tight_layout()
        return fig

    def render_global(self,
                      global_cam: BatchCameraManager,
                      global_images: np.ndarray,
                      pose_idx: int,
                      scene_idx: int,
                      with_hand=True,
                      overlay_gt=False,
                      **mesh_kwargs,
                      ) -> np.ndarray:
        """ returns: (H, W, 3) """
        nt_ind = pose_idx * self.train_size + scene_idx
        if not with_hand:
            return self.ihoi_img_patch[nt_ind]
        mhand, mobj = self.get_meshes(pose_idx=pose_idx,
                                      scene_idx=scene_idx, **mesh_kwargs)
        img = projection.perspective_projection_by_camera(
            [mhand, mobj],
            global_cam[scene_idx],
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            ),
            image=global_images[scene_idx],
        )

        if overlay_gt:
            raise NotImplementedError
            all_mask = np.zeros_like(img, dtype=np.float32)
            mask_hand = self.ref_mask_hand[nt_ind].cpu().numpy().squeeze()
            print(mask_hand.shape, all_mask.shape)
            all_mask = np.where(
                mask_hand[..., None], (0, 0, 0.8), all_mask)
            print(all_mask.shape)
            if obj_idx >= 0:
                mask_obj = self.ref_mask_object[nt_ind].cpu().numpy()
                all_mask = np.where(
                    mask_obj[..., None], (0.6, 0, 0), all_mask)
            all_mask = np.uint8(255*all_mask)
            img = cv2.addWeighted(np.uint8(img*255), 0.9, all_mask, 0.5, 1.0)
        return img

    def get_phi_tf(self, scene_idx: int, as_mesh=True):
        """ Get the transformation of phi in Hand space!
        Note: in hand space, so we need to apply Q and t2
        Returns 4x4 torch.tensor
        """
        translate = torch.eye(4)
        translate[:3, -1] = torch.tensor([0, 0, 0])
        assert self.version == LOWDIM
        Rw_aux = torch.eye(4)
        Rw_aux[:3, :3] = rotation_6d_to_matrix(self.Rw_6d)

        o2phi = torch.eye(4)
        o2phi[:3, :3] = rotation_6d_to_matrix(self.P_6d)
        o2phi[:3, -1] = self.t1

        o2h = torch.eye(4)
        o2h[:3, :3] = rotation_6d_to_matrix(self.Q_6d)
        o2h[:3, -1] = self.t2
        h2c = torch.eye(4)
        h2c[:3, :3] = self.rot_mat_hand[scene_idx, ...]
        h2c[:3, -1] = self.translations_hand[scene_idx, ...]
        tf = h2c @ o2h @ torch.inverse(o2phi) @ torch.inverse(Rw_aux) @ translate
        tf = tf.detach()

        if as_mesh:
            mphi = trimesh.creation.cylinder(radius=0.005, height=0.25, transform=tf)
            mphi = SimpleMesh(mphi.vertices, mphi.faces, tex_color=(1,0,1.0))
            return mphi

        return tf

    def make_compare_video(self,
                           global_cam: BatchCameraManager,
                           global_images: np.ndarray,
                           pose_idx: int = 0) -> List[np.ndarray]:
        """
        Args:
            pose_idx: usually 0 for drawing eval frames
        """
        frames = []
        scene_indices = range(self.train_size)
        image_h = global_images[0].shape[0]

        for i in scene_indices:
            img_mesh = self.render_global(
                global_cam=global_cam,
                global_images=global_images,
                pose_idx=pose_idx,
                scene_idx=i,
                with_hand=True,
                overlay_gt=False)

            h, w, _ = img_mesh.shape
            sideview = self.render_triview(
                pose_idx, i, views=['left', 'back'], hstack=False,
                rend_size=image_h//2)
            img = np.hstack([global_images[i], img_mesh * 255 , sideview * 255])
            frames.append(img)
        return frames
