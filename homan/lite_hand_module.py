from typing import List, NamedTuple, Tuple
from hydra.utils import to_absolute_path
from homan.homan_ManoModel import HomanManoModel

import torch
import torch.nn as nn
import neural_renderer as nr
import roma
from pytorch3d.transforms import rotation_6d_to_matrix
from homan.ho_utils import compute_transformation_persp
from nnutils.handmocap import get_hand_faces
from homan.lossutils import (
    rotation_loss_v1, iou_loss
)
from homan.metrics import batch_mask_iou
from config.epic_constants import REND_SIZE

from libzhifan.numeric import check_shape


__left_mano_model = HomanManoModel(
    to_absolute_path("externals/mano"), side='left', pca_comps=45)  # Note(zhifan): in HOMAN num_pca = 16
__right_mano_model = HomanManoModel(
    to_absolute_path("externals/mano"), side='right', pca_comps=45)
_mano_model_dict = {
    'left': __left_mano_model, 'right': __right_mano_model}


class LiteHandModule(nn.Module):

    # Input to this module
    class LiteHandParams(NamedTuple):
        """ As ho_forwarder_v2.py:
            mano_trans init to zero,
            mano_rot init to zero,
            scale_hand init to one
        """
        camintr: torch.Tensor  # (L, 3, 3) Ihoi bounding box camera.
        rotations_hand: torch.Tensor  # (L, 6)
        translations_hand: torch.Tensor
        hand_side: str
        mano_pca_pose: torch.Tensor  # (L, 45)
        mano_betas: torch.Tensor  # (L, 10)
        target_masks_hand: torch.Tensor  # (L, W, W), W=256

    # Output of this module
    class HandData(NamedTuple):
        """ ALl these represent different T*N hand data."""
        camintr: torch.Tensor   # (N*T, 3, 3)  Ihoi bounding box camera.
        rotations_hand: torch.Tensor  # (N*T, 6)
        translations_hand: torch.Tensor  # (N*T, 1, 3)
        v_hand_global: torch.Tensor  # (N*T, V, 3)
        v_hand_local: torch.Tensor  # (N*T, V, 3)
        rot_mat_hand: torch.Tensor  # (N*T, 3, 3)
        """ for visualization """
        faces_hand: torch.Tensor  # (N*T, F, 3)
        ref_mask_hand: torch.Tensor  # (N*T, W, W)

        def __getitem__(self, inds):
            return LiteHandModule.HandData(
                camintr=self.camintr[inds], rotations_hand=self.rotations_hand[inds],
                translations_hand=self.translations_hand[inds],
                v_hand_global=self.v_hand_global[inds], v_hand_local=self.v_hand_local[inds],
                rot_mat_hand=self.rot_mat_hand[inds], faces_hand=self.faces_hand[inds],
                ref_mask_hand=self.ref_mask_hand[inds])

    def __init__(self):
        super().__init__()
        self.mask_size = REND_SIZE

        self.renderer = nr.renderer.Renderer(
            image_size=self.mask_size,
            K=None,
            R=torch.eye(3, device='cuda')[None],
            t=torch.zeros([1, 3], device='cuda'),
            orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = torch.as_tensor(0.3)
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]

    @staticmethod
    def gather0d(source, indices: torch.Tensor):
        """
        Args:
            source: (L, F1, F2, ...)
            indices: (N*T)
        Returns:
            (N*T, F1, F2, ...)
        """
        feature_shape = source.shape[1:]
        dummy_shape = (1,) * len(feature_shape)
        N, T = indices.shape
        indices = indices.view(-1, *dummy_shape).expand(-1, *feature_shape)
        out = torch.gather(source, dim=0, index=indices)
        return out

    def __getitem__(self, indices) -> HandData:
        """
        Args:
            indices: (N, T)

        Returns: a HandData object
            where it holds B*N different hand data.
        """
        # This function should only copy the member data DIRECTLY.
        # transformation of hand data should not be done here.
        indices = torch.as_tensor(indices, device=self.rotations_hand.device)
        v_hand_global = self.get_verts_hand()
        v_hand_local = self.get_verts_hand(hand_space=True)
        camintr = self.gather0d(self.camintr, indices)
        rotations_hand = self.gather0d(self.rotations_hand, indices)
        translations_hand = self.gather0d(self.translations_hand, indices)
        v_hand_global = self.gather0d(v_hand_global, indices)
        v_hand_local = self.gather0d(v_hand_local, indices)
        rot_mat_hand = self.gather0d(self.rot_mat_hand, indices)
        faces_hand = self.gather0d(self.faces_hand, indices)
        ref_mask_hand = self.gather0d(self.ref_mask_hand, indices)
        return self.HandData(
            camintr=camintr, rotations_hand=rotations_hand,
            translations_hand=translations_hand,
            v_hand_global=v_hand_global, v_hand_local=v_hand_local,
            rot_mat_hand=rot_mat_hand, faces_hand=faces_hand,
            ref_mask_hand=ref_mask_hand)

    @property
    def rot_mat_hand(self) -> torch.Tensor:
        """ (L, 3, 3) matrix, apply to col-vector """
        return rotation_6d_to_matrix(self.rotations_hand)

    def get_verts_hand(self, detach_scale=False, hand_space=False) -> torch.Tensor:
        """
        Args:
            hand_space: if True, return hand vertices in hand space itself.
        """
        all_hand_verts = []
        for hand_idx, side in enumerate(self.hand_sides):
            mano_pca_pose = self.mano_pca_pose[hand_idx::self.hand_nb]
            mano_rot = self.mano_rot[hand_idx::self.hand_nb]
            mano_res = self.mano_model.forward_pca(
                mano_pca_pose,
                rot=mano_rot,
                betas=self.mano_betas[hand_idx::self.hand_nb],
                side=side)
            vertices = mano_res["verts"]
            all_hand_verts.append(vertices)
        all_hand_verts = torch.stack(all_hand_verts).transpose(
            0, 1).contiguous().view(-1, 778, 3)
        verts_hand_og = all_hand_verts + self.mano_trans.unsqueeze(1)
        if hand_space:
            return verts_hand_og

        scale = self.scale_hand.detach() if detach_scale else self.scale_hand
        rotations_hand = self.rot_mat_hand

        return compute_transformation_persp(
            meshes=verts_hand_og,
            translations=self.translations_hand,
            rotations=rotations_hand,
            intrinsic_scales=scale,
        )

    def set_hand_params(self, params: LiteHandParams):
        """ Inititalize person parameters
        """
        num_source = len(params.camintr)
        camintr = params.camintr
        rotations_hand = params.rotations_hand
        translations_hand = params.translations_hand
        hand_side = 'left' if 'left' in params.hand_side else 'right'
        mano_pca_pose = params.mano_pca_pose
        mano_betas = params.mano_betas
        target_masks_hand = params.target_masks_hand
        mano_trans = torch.zeros([num_source, 3], device=mano_pca_pose.device)
        mano_rot = torch.zeros([num_source, 3], device=mano_pca_pose.device)

        self.register_buffer("camintr", camintr)
        self.hand_sides = [hand_side]
        self.hand_nb = 1

        self.mano_model = _mano_model_dict[hand_side]
        translation_init = translations_hand.detach().clone()
        self.translations_hand = nn.Parameter(translation_init,
                                              requires_grad=True)
        rotations_hand = rotations_hand.detach().clone()
        if rotations_hand.shape[-1] == 3:
            raise ValueError('Invalid Input')
        self.rotations_hand = nn.Parameter(rotations_hand, requires_grad=True)

        self.mano_pca_pose = nn.Parameter(mano_pca_pose, requires_grad=True)
        self.mano_rot = nn.Parameter(mano_rot, requires_grad=True)
        self.mano_trans = nn.Parameter(mano_trans, requires_grad=True)
        # self.mano_betas = nn.Parameter(torch.zeros_like(mano_betas),
        #                                 requires_grad=True)
        self.mano_betas = nn.Parameter(mano_betas, requires_grad=True)
        self.scale_hand = nn.Parameter(
            torch.ones(1).float(),
            requires_grad=True)

        faces_hand = get_hand_faces(hand_side)
        num_faces_hand = faces_hand.size(1)
        self.register_buffer(
            "textures_hand",
            torch.ones(num_source, num_faces_hand, 1, 1, 1, 3))
        self.register_buffer(
            "faces_hand", faces_hand.expand(num_source, num_faces_hand, 3))
        self.cuda()

        self.register_buffer("ref_mask_hand", (target_masks_hand > 0).float())
        self.register_buffer("keep_mask_hand",
                             (target_masks_hand >= 0).float())
        mask_h, mask_w = target_masks_hand.shape[-2:]
        self.register_buffer(
            "masks_human",
            target_masks_hand.view(num_source, 1, mask_h, mask_w).bool())
        self.cuda()
        self._check_shape_hand(num_source)

    def _check_shape_hand(self, bsize):
        check_shape(self.faces_hand, (bsize, -1, 3))
        check_shape(self.camintr, (bsize, 3, 3))
        check_shape(self.ref_mask_hand,
                    (bsize, self.mask_size, self.mask_size))
        mask_shape = self.ref_mask_hand.shape
        check_shape(self.keep_mask_hand, mask_shape)
        check_shape(self.rotations_hand, (bsize, 6))
        check_shape(self.translations_hand, (bsize, 1, 3))
        # ordinal loss
        check_shape(self.masks_human, (bsize, 1,
                    self.mask_size, self.mask_size))

    def forward_hand(self,
                     loss_weights={
                         'sil': 1,
                         'pca': 1,
                         'rot': 10,
                         'transl': 1,
                     }) -> Tuple[torch.Tensor, dict]:
        l_sil = self.loss_sil_hand(compute_iou=False, func='l2').sum()
        l_pca = self.loss_pca_interpolation().sum()
        l_rot = self.loss_hand_rot().sum()
        l_transl = self.loss_hand_transl().sum()
        losses = {
            'sil': l_sil,
            'pca': l_pca,
            'rot': l_rot,
            'transl': l_transl,
        }
        for k, l in losses.items():
            losses[k] = l * loss_weights[k]
        total_loss = sum([v for v in losses.values()])
        return total_loss, losses

    def loss_pca_interpolation(self) -> torch.Tensor:
        """
        Interpolation Prior: pose(t) = (pose(t+1) + pose(t-1)) / 2

        Returns: (L-2,)
        """
        target = (self.mano_pca_pose[2:] + self.mano_pca_pose[:-2]) / 2
        pred = self.mano_pca_pose[1:-1]
        loss = torch.sum((target - pred)**2, dim=(1))
        return loss

    def loss_hand_rot(self) -> torch.Tensor:
        """ Interpolation loss for hand rotation """
        device = self.rotations_hand.device
        rotmat = self.rot_mat_hand
        # with torch.no_grad():
        rot_mid = roma.rotmat_slerp(
            rotmat[2:], rotmat[:-2],
            torch.as_tensor([0.5], device=device))[0]
        loss = rotation_loss_v1(rot_mid, rotmat[1:-1])
        return loss

    def loss_hand_transl(self) -> torch.Tensor:
        """
        Returns: (L-2,)
        """
        interp = (self.translations_hand[2:] + self.translations_hand[:-2]) / 2
        pred = self.translations_hand[1:-1]
        loss = torch.sum((interp - pred)**2, dim=(1, 2))
        return loss

    def loss_sil_hand(self, compute_iou=False, func='l2'):
        """ returns: (B,) """
        rend = self.renderer(
            self.get_verts_hand(),
            self.faces_hand,
            K=self.camintr,
            mode="silhouettes")
        image = self.keep_mask_hand * rend
        if func == 'l2':
            loss_sil = torch.sum(
                (image - self.ref_mask_hand)**2, dim=(1, 2))
            loss_sil = loss_sil / self.keep_mask_hand.sum(dim=(1, 2))
        elif func == 'iou':
            loss_sil = iou_loss(image, self.ref_mask_hand)
        elif func == 'l2_iou':
            loss_sil = torch.sum(
                (image - self.ref_mask_hand)**2, dim=(1, 2))
            loss_sil = loss_sil / self.keep_mask_hand.sum(dim=(1, 2))
            with torch.no_grad():
                iou_factor = iou_loss(image, self.ref_mask_hand, post='rev')
            loss_sil = loss_sil * iou_factor

        # loss_sil = loss_sil / self.bsize
        if compute_iou:
            ious = batch_mask_iou(image, self.ref_mask_hand)
            return loss_sil, ious
        return loss_sil

    def rend_hand(self):
        rend = self.renderer(
            self.get_verts_hand(),
            self.faces_hand,
            K=self.camintr,
            mode="silhouettes")
        image = self.keep_mask_hand * rend
        return image