""" Implement homan's ManoModel using manopth, instead of Mano """

import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer

class HomanManoModel(nn.Module):
    
    def __init__(self, mano_root, side, pca_comps=16, batch_size=1):
        """
        Args:
            side: one of {'left', 'right'}
        """
        super().__init__()
        # assert pca_comps == 16
        self.mano_layer = ManoLayer(
            flat_hand_mean=False, 
            ncomps=pca_comps, 
            side='left' if 'left' in side else 'right',
            mano_root=mano_root,
            use_pca=True)
        self.register_buffer(
            'hand_faces', self.mano_layer.th_faces.unsqueeze(0))

    def forward_pca(self,
                    pca_pose,
                    rot=None,
                    betas=None,
                    side=None):
        """ 
        Args:
            pca_pose: (?, pca_comps+) torch.Tensor
            rot: (1, 3) torch.Tensor
            betas: (1, 1) torch.Tensor

        Returns:
            mano_res: dict
                - verts: (batch_size, 778, 3)
                - joints: (batch_size, 16, 3)
        """
        if rot is None:
            rot = torch.zeros([1, 3], dtype=pca_pose.dtype, device=pca_pose.device)

        th_pose_coeffs = torch.cat([rot, pca_pose], axis=-1)
        v, j = self.mano_layer.forward(th_pose_coeffs, betas, th_trans=None)
        v /= 1000
        j /= 1000
        j_indices = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
        j_ret = j[:, j_indices, :]

        mano_res = dict(
            verts=v,
            joints=j_ret,
        )
        return mano_res