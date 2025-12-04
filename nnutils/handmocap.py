import os.path as osp
from typing import Tuple

import numpy as np
import pytorch3d.transforms.rotation_conversions as rot_cvt
import torch
from libzhifan.geometry import BatchCameraManager
from libzhifan.odlib import xywh_to_xyxy, xyxy_to_xywh

from config.epic_constants import FRANKMOCAP_INPUT_SIZE, REND_SIZE
from datasets.data_element import DataElement
from nnutils import geom_utils, image_utils
from nnutils.hand_utils import ManopthWrapper


""" ManopthWrapper"""

__hand_wrapper_left = ManopthWrapper(flat_hand_mean=False, side='left').to('cuda')
__hand_wrapper_right = ManopthWrapper(flat_hand_mean=False, side='right').to('cuda')


def get_hand_wrapper(side: str) -> ManopthWrapper:
    if 'left' in side:
        return __hand_wrapper_left
    elif 'right' in side:
        return __hand_wrapper_right
    else:
        raise ValueError(f"Side {side} not understood.")


def recover_pca_pose(pred_hand_pose: torch.Tensor, side: str) -> torch.Tensor:
    """
    if
        v_exp = ManopthWrapper(pca=False, flat=False).(x_0)
        x_pca = self.recover_pca_pose(self.x_0)  # R^45
    then
        v_act = ManoLayer(pca=True, flat=False, ncomps=45).forward(x_pca)
        v_exp == v_act

    note above requires mano_rot == zeros, since the computation of rotation
        is different in ManopthWrapper
    """
    M_pca_inv = torch.inverse(
        get_hand_wrapper(side).mano_layer_side.th_comps)
    mano_pca_pose = pred_hand_pose.mm(M_pca_inv)
    return mano_pca_pose


def get_hand_faces(side: str) -> torch.Tensor:
    return get_hand_wrapper(side).hand_faces


""" HandBboxDetector """

def collate_mocap_hand(mocap_predictions: list,
                       side: str,
                       fields=('pred_hand_pose', 'pred_hand_betas',
                               'pred_camera', 'bbox_processed')
                       ) -> dict:
    """
    mocap shapes:
        pred_vertices_smpl (778, 3)
        pred_joints_smpl (21, 3)
        faces (1538, 3)
        bbox_scale_ratio ()
        bbox_top_left (2,)
        bbox_processed (4,)
        pred_camera (3,)
        img_cropped (224, 224, 3)
        pred_hand_pose (1, 48)
        pred_hand_betas (1, 10)
        pred_vertices_img (778, 3)
        pred_joints_img (21, 3)

    Args:
        mocap_predictions: list of [
            dict('left_hand': dict
                 'right_hand': dict)
            ]

    Returns:
        mocap_hand: dict with key in `fields`.
    """
    one_hand = dict()
    for key in fields:
        content = []
        for mocap_pred in mocap_predictions:
            elem = torch.as_tensor(mocap_pred[side][key])
            if key != 'pred_hand_pose' and key != 'pred_hand_betas':
                elem = elem.unsqueeze(0)
            content.append(elem)
        content = torch.cat(content, dim=0)
        one_hand[key] = content
    return one_hand


# def get_handmocap_detector(view_type='ego_centric'):
#     from handmocap.hand_bbox_detector import HandBboxDetector
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     bbox_detector =  HandBboxDetector(view_type, device)
#     return bbox_detector


""" Used by `obj_pose` """


def compute_hand_transform(rot_axisang,
                           pred_hand_pose,
                           pred_camera,
                           side: str,
                           hand_cam: BatchCameraManager):
    """
    Args:
        rot_axisang: (B, 3)
        pred_hand_pose: (B, 45)
        pred_camera: (B, 3)
            Used for translate hand_mesh to convert hand_mesh
            so that result in a weak perspective camera.
        hand_wrapper: ManoPthWrapper

    Returns:
        rotation6d: (B, 3, 3), will apply to row-vecs
        translation: (B, 1, 3)
    """
    rotation = rot_cvt.axis_angle_to_matrix(rot_axisang)  # (1, 3) - > (1, 3, 3), col-vec
    rot_homo = geom_utils.rt_to_homo(rotation)
    glb_rot = geom_utils.matrix_to_se3(rot_homo)  # (1, 4, 4) -> (1, 12)
    _, joints = get_hand_wrapper(side)(
        glb_rot,
        pred_hand_pose, return_mesh=True)
    s, tx, ty = torch.split(pred_camera, [1, 1, 1], dim=1)
    device = tx.device

    fx, fy, cx, cy, _, _ = hand_cam.unpack()
    fx, fy, cx, cy = map(
        lambda x: torch.as_tensor(x, device=device).view_as(s),
        (fx, fy, cx, cy))
    f = (fx + fy) / 2  # How to enforce fx=fy?

    sw  = s * FRANKMOCAP_INPUT_SIZE /2  # s*w/2
    tx = tx + 1/s - cx / sw
    ty = ty + 1/s - cy / sw
    tz = f / sw
    translation = torch.cat([tx, ty, tz], dim=1)
    translation = translation - joints[:, 5]
    rotation6d = rot_cvt.matrix_to_rotation_6d(rotation)
    return rotation6d, translation[:, None]


def cam_from_bbox(hand_bbox,
                  fx, # =WEAK_CAM_FX,
                  img_height, #=IMG_HEIGHT,
                  img_width, #=IMG_WIDTH
                  ) -> Tuple[BatchCameraManager, BatchCameraManager]:
    """
    Args:
        hand_bbox: (B, 4) in GLOBAL screen space
            This box should be used in mocap_predictor.
            hand bounding box XYWH in original image
            same as one_hand['bbox_processed']

    Returns:
        hand_cam, global_cam: BatchCameraManager
    """
    hand_crop_h = 224
    hand_crop_w = 224
    _, _, box_w, box_h = torch.split(hand_bbox, [1, 1, 1, 1], dim=1)
    box_h = box_h.view(-1)
    box_w = box_w.view(-1)
    fx = torch.ones_like(box_w) * fx
    fy = torch.ones_like(box_w) * fx
    cx = torch.zeros_like(box_w)
    cy = torch.zeros_like(box_w)
    hand_crop_h = torch.ones_like(box_w) * hand_crop_h  # 224
    hand_crop_w = torch.ones_like(box_w) * hand_crop_w
    hand_cam = BatchCameraManager(
        fx=fx, fy=fy, cx=cx, cy=cy, img_h=hand_crop_h, img_w=hand_crop_w,
        in_ndc=True
    )

    _, _, hand_h, hand_w = torch.split(hand_bbox, [1, 1, 1, 1], dim=1)
    hand_h = hand_h.view(-1)
    hand_w = hand_w.view(-1)
    global_cam = hand_cam.resize(hand_h, hand_w).uncrop(
        hand_bbox, img_height, img_width)
    return hand_cam, global_cam


def extract_forwarder_input(data_elem: DataElement,
                            ihoi_box_expand: float,
                            hand_predictor=None,
                            out_of_image_fill=0,
                            device='cuda',
                            run_hand_predictor=True,
                            patch_crop_method='hand_bbox',
                            debug=False):
    """
    1. Run frankmocap predictor
    2. Extract wrist poses and finger poses.
    3. Extract hand mask, w/ squaring and resizing
    4. Extract Object Mask Patch

    Args:
        out_of_image_fill:
            filling value to out-of-view mask. -1 is ignore, 0 will penelise out-of-view predictions
        patch_crop_method: 'hand_bbox', 'obj_bbox', 'hand_obj_max'.
            specify how to crop patches from raw input.
    """
    images = data_elem.images
    hand_bbox_dicts = data_elem.hand_bbox_dicts
    side = data_elem.side
    obj_bboxes = data_elem.obj_bboxes
    hand_masks = data_elem.hand_masks
    object_masks = data_elem.object_masks
    cat = data_elem.cat
    global_cam = data_elem.global_camera

    """ Process all hands """
    if hand_predictor is None and run_hand_predictor:
        raise ValueError(
            "hand_predictor required, e.g. frankmocap predictor")

    hand_rotation_6d, hand_translation, \
        mano_pca_pose, pred_hand_betas = None, None, None, None
    if run_hand_predictor:
        mocap_predictions = []
        for img, hand_dict in zip(images, hand_bbox_dicts):
            mocap_pred = hand_predictor.regress(
                img[..., ::-1], [hand_dict]
            )
            mocap_predictions += mocap_pred
        one_hands = collate_mocap_hand(mocap_predictions, side)

        """ Extract mocap_output """
        pred_hand_full_pose, pred_hand_betas, pred_camera = map(
            lambda x: torch.as_tensor(one_hands[x], device=device),
            ('pred_hand_pose', 'pred_hand_betas', 'pred_camera'))
        hand_bbox_proc = one_hands['bbox_processed']
        rot_axisang = pred_hand_full_pose[:, :3]
        pred_hand_pose = pred_hand_full_pose[:, 3:]
        mano_pca_pose = recover_pca_pose(pred_hand_pose, side)
        if debug:
            print(pred_camera)
            print(hand_bbox_proc)

        hand_sz = torch.ones_like(global_cam.fx) * 224
        hand_cam = global_cam.crop(hand_bbox_proc).resize(new_w=hand_sz, new_h=hand_sz)
        hand_rotation_6d, hand_translation = compute_hand_transform(
            rot_axisang, pred_hand_pose, pred_camera, side,
            hand_cam=hand_cam)

    """ Extract mask input """
    rend_size = REND_SIZE
    if patch_crop_method == 'hand_bbox':
        hand_bboxes = torch.as_tensor(np.stack([v[side] for v in hand_bbox_dicts]))
        bbox_squared = image_utils.square_bbox_xywh(hand_bboxes, ihoi_box_expand).int()
    elif patch_crop_method == 'obj_bbox':
        bbox_squared = image_utils.square_bbox_xywh(obj_bboxes, ihoi_box_expand).int()
    elif patch_crop_method == 'hand_obj_max':
        hand_bboxes = torch.as_tensor(np.stack([v[side] for v in hand_bbox_dicts]))
        max_bboxes = xywh_to_xyxy(hand_bboxes)
        obj_bboxes_xyxy = xywh_to_xyxy(obj_bboxes)
        max_bboxes[:, :2] = torch.minimum(max_bboxes[:, :2], obj_bboxes_xyxy[:, :2])
        max_bboxes[:, 2:] = torch.maximum(max_bboxes[:, 2:], obj_bboxes_xyxy[:, 2:])
        bbox_squared = xyxy_to_xywh(max_bboxes).int()
    else:
        raise ValueError(f"{patch_crop_method} not understood")
    obj_mask_patch = image_utils.batch_crop_resize(
        object_masks, bbox_squared, rend_size, fill=out_of_image_fill)
    hand_mask_patch = image_utils.batch_crop_resize(
        hand_masks, bbox_squared, rend_size, fill=out_of_image_fill)
    image_patch = image_utils.batch_crop_resize(
        images, bbox_squared, rend_size)
    ihoi_h = torch.ones([len(global_cam)]) * rend_size
    ihoi_w = torch.ones([len(global_cam)]) * rend_size
    ihoi_cam = global_cam.crop(bbox_squared).resize(ihoi_h, ihoi_w)

    ihoi_cam_nr_mat = ihoi_cam.to_nr(return_mat=True)
    ihoi_cam_mat = ihoi_cam.get_K()
    return ihoi_cam_nr_mat, ihoi_cam_mat, image_patch, \
        hand_rotation_6d, hand_translation, \
        mano_pca_pose, pred_hand_betas, \
        hand_mask_patch, obj_mask_patch
