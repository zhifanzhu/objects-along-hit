""" 
Trace mano_layer.forward()'s transform
since the mano_layer's global rotation and translation 
are not straightforwardly implemented
"""


import torch

from manopth import rodrigues_layer, rotproj, rot6d
from manopth.tensutils import (th_posemap_axisang, th_with_zeros, th_pack,
                               subtract_flat_id, make_list)

from manopth.manolayer import ManoLayer

""" ManoLayer Tranform tracer """
class ManoLayerTracer(ManoLayer):
    __constants__ = [
        'use_pca', 'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check',
        'side', 'center_idx', 'joint_rot_mode'
    ]
    def __init__(self, *args, **kwargs):
        super(ManoLayerTracer, self).__init__(*args, **kwargs)

    def forward_transform(self,
                          th_pose_coeffs,
                          th_betas=torch.zeros(1),
                          th_trans=torch.zeros(1),
                          root_palm=torch.Tensor([0]),
                          share_betas=torch.Tensor([0]),
                          ):
        """
        This outputs an additional hand-to-world transform using the hand-to-world params
            specified in (th_pose_coeffs and th_trans)
        This is essentially reading out the root(0) component
        of via `th_results2[:, :, :, 0]`

        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        root_palm: return palm as hand root instead of wrist
        """

        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        if self.use_pca or self.joint_rot_mode == 'axisang':
            # Remove global rot coeffs
            th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                 self.ncomps]
            if self.use_pca:
                # PCA components --> axis angles
                th_full_hand_pose = th_hand_pose_coeffs.mm(self.th_selected_comps)
            else:
                th_full_hand_pose = th_hand_pose_coeffs

            # Concatenate back global rot
            th_full_pose = torch.cat([
                th_pose_coeffs[:, :self.rot],
                self.th_hands_mean + th_full_hand_pose
            ], 1)
            if self.root_rot_mode == 'axisang':
                # compute rotation matrixes from axis-angle while skipping global rotation
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
                root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
                th_rot_map = th_rot_map[:, 9:]
                th_pose_map = th_pose_map[:, 9:]
            else:
                # th_posemap offsets by 3, so add offset or 3 to get to self.rot=6
                th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose[:, 6:])
                if self.robust_rot:
                    root_rot = rot6d.robust_compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
                else:
                    root_rot = rot6d.compute_rotation_matrix_from_ortho6d(th_full_pose[:, :6])
        else:
            assert th_pose_coeffs.dim() == 4, (
                'When not self.use_pca, '
                'th_pose_coeffs should have 4 dims, got {}'.format(
                    th_pose_coeffs.dim()))
            assert th_pose_coeffs.shape[2:4] == (3, 3), (
                'When not self.use_pca, th_pose_coeffs have 3x3 matrix for two'
                'last dims, got {}'.format(th_pose_coeffs.shape[2:4]))
            th_pose_rots = rotproj.batch_rotprojs(th_pose_coeffs)
            th_rot_map = th_pose_rots[:, 1:].view(batch_size, -1)
            th_pose_map = subtract_flat_id(th_rot_map)
            root_rot = th_pose_rots[:, 0]

        # Full axis angle representation with root joint
        if th_betas is None or th_betas.numel() == 1:
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       self.th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                batch_size, 1, 1)

        else:
            if share_betas:
                th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
            th_v_shaped = torch.matmul(self.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
            th_j = torch.matmul(self.th_J_regressor, th_v_shaped)
            # th_pose_map should have shape 20x135

        th_v_posed = th_v_shaped + torch.matmul(
            self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done !

        # Global rigid transformation

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        # all_transforms = [torch.matmul(test_rot, root_trans.unsqueeze(1))]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)  # (1, 5, 3)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results

        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == 'right':
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        if bool(root_palm):
            palm = (th_verts[:, 95] + th_verts[:, 22]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)

        # Reorder joints to match visualization utilities
        th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        pose_hand2world = th_results2[:, :, :, 0]

        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self.center_idx is not None:
                center_joint = th_jtr[:, self.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
                pose_hand2world[:, :3, -1] -= center_joint.squeeze(1)
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)
            pose_hand2world[:, :3, -1] += th_trans

        # Scale to milimeters
        th_verts = th_verts * 1000
        th_jtr = th_jtr * 1000
        return th_verts, th_jtr, th_results2[:, :, :, 0]


def verify_transform_tracing():
    import torch
    rot_h = torch.tensor([[ 1.4665345 , -0.05057937,  0.05102186]])
    trans_h = torch.tensor([[0.7166632 , 0.32865486, 1.263176  ]])
    pose_h = torch.tensor([
        [-0.01875794, -0.13479437,  0.24923648, -0.14190933, -0.06999575,
        1.0578995 ,  0.5623529 ,  0.1522226 ,  0.05068795,  0.5636545 ,
       -0.09637039,  0.49426743,  0.0017043 ,  0.14128442,  0.57456106,
       -0.04712581, -0.2694806 ,  0.50660276,  0.24623552,  0.7137211 ,
       -0.27414265,  0.46243876,  0.08836462,  1.0691323 ,  0.0679919 ,
       -0.01349943,  0.43018502,  0.37965378,  0.27904364,  0.3275354 ,
        0.3427083 , -0.01312044,  0.99723417, -0.08919289,  0.13270791,
        0.3438644 , -0.8335977 , -0.4185787 ,  0.74871415,  0.19252153,
        0.27427268, -0.9973835 , -0.57042277, -0.725015  ,  1.2476287 ]
    ])
    shape_h = torch.tensor([[-0.79835755, -0.86827576, -5.280928  , -0.67895216, -0.36445457,
        1.0380967 ,  1.7879561 , -1.7621936 , -2.238167  , -2.9983373 ]])

    mano_layer = ManoLayer(
        flat_hand_mean=False, ncomps=45, side='left', use_pca=False,
        mano_root='./externals/mano/') 
    th_pose_coeffs = torch.cat([rot_h, pose_h], axis=-1)
    v_implicit, _ = mano_layer.forward(th_pose_coeffs, th_betas=shape_h, th_trans=trans_h)
    v_implicit /= 1000


    # # explicit transform, INCORRECT
    # rot_zeros = torch.zeros_like(rot_h)
    # trans_zeros = torch.zeros_like(trans_h)
    # th_pose_coeffs = torch.cat([rot_zeros, pose_h], axis=-1)
    # v, _ = mano_layer.forward(th_pose_coeffs, th_betas=shape_h, th_trans=trans_zeros)
    # v /= 1000
    # tf = _get_transform(rot_h, trans_h)
    # v = pose_apply(tf, v)
    # v_explicit_incorrect = v

    # print("Implicit", v_implicit)
    # print("Explicit (Incorrect)", v_explicit_incorrect)
    # print(torch.norm(v_implicit - v_explicit_incorrect))

    # explicit transform, Corrrect
    rot_zeros = torch.zeros_like(rot_h)
    trans_zeros = torch.zeros_like(trans_h)
    v, _ = mano_layer.forward(
        torch.cat([rot_zeros, pose_h], axis=-1), 
        th_betas=shape_h, th_trans=trans_zeros)
    v /= 1000  # in meter
    mano_layer_tracer = ManoLayerTracer(
        flat_hand_mean=False, ncomps=45, side='left', use_pca=False,
        mano_root='./externals/mano/') 
    _, _, tf = mano_layer_tracer.forward_transform(
        torch.cat([rot_h, pose_h], axis=-1), 
        shape_h, th_trans=trans_h)
    v = pose_apply(tf, v)
    v_explicit_correct = v

    print("Implicit", v_implicit)
    print("Explicit (Correct)", v_explicit_correct)
    print(v_implicit - v_explicit_correct)
    print(torch.norm(v_implicit - v_explicit_correct))


def pose_apply(pose: torch.Tensor, verts: torch.Tensor):
    """
    pose: (N, 4, 4)
    verts: (N, V, 3)
    => (N, V, 3)
    """
    rot = pose[:, :3, :3]
    transl = pose[:, :3, 3:]
    return (torch.bmm(rot, verts.permute(0, 2, 1)) + transl).permute(0, 2, 1) 

def _get_transform(rot_world, trans_world):
    """ This convert axis-angle rotation vector and translation 
    to pose matrix (4x4)

    Args:
        rot: (N, 3)
        transl: (N, 3)
    
    Returns: (N, 4, 4)) 
    """
    from pytorch3d.transforms import rotation_conversions as rotcvt
    rot_world = rotcvt.axis_angle_to_matrix(rot_world)  # apply-to-col
    T_world = torch.eye(4).repeat(rot_world.shape[0], 1, 1)
    T_world[:, :3, :3] = rot_world
    T_world[:, :3, -1] = trans_world
    return T_world


if __name__ == '__main__':
    verify_transform_tracing()