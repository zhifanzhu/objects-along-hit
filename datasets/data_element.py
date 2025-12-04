from typing import NamedTuple

import torch
from libzhifan.geometry import BatchCameraManager


class DataElement(NamedTuple):  # EPIC-HOR
    frame_inds: list  # (N,) bookkeeping frames 
    images: list 
    hand_bbox_dicts: list 
    side: str 
    obj_bboxes: torch.Tensor
    hand_masks: torch.Tensor 
    object_masks: torch.Tensor 
    cat: str
    global_camera: BatchCameraManager
    obj_verts: torch.Tensor  # (V, 3)
    obj_faces: torch.Tensor  # (F, 3)


class ArcticDataElement(NamedTuple):
    images: list   # (N,)
    hand_bbox_dicts: list 
    side: str 
    obj_bboxes: torch.Tensor
    hand_masks: torch.Tensor 
    object_masks: torch.Tensor 
    cat: str
    global_camera: BatchCameraManager

    obj_verts: torch.Tensor  # (V, 3)
    obj_faces: torch.Tensor  # (F, 3)
    gt_hand_pca: torch.Tensor  # (N, 45), this will be concatenated with zeros(3)
    gt_hand_betas: torch.Tensor  # (N, 10)
    gt_hand_rot: torch.Tensor  # (N, 3, 3)
    gt_hand_trans: torch.Tensor  # (N, 3)
    gt_obj2hand_rot: torch.Tensor  # (N, 6)
    gt_obj2hand_transl: torch.Tensor  # (N, 1, 3)
    obj_diameter: float