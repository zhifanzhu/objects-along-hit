from typing import List, NamedTuple

import numpy as np
import pandas as pd
import torch
from libzhifan import io
from libzhifan.geometry import BatchCameraManager, CameraManager
from libzhifan.odlib import xywh_to_xyxy, xyxy_to_xywh
from PIL import Image
from pytorch3d.transforms import matrix_to_rotation_6d
from torch.utils.data import Dataset

from code_arctic.data_reader import (ARCTIC_IMAGES_DIR, CROPPED_IMAGE_SIZE,
                                     LEFT, MASK_LEFT_ID, MASK_OBJ_ID,
                                     MASK_RIGHT_ID, RIGHT)
from code_arctic.data_reader_onthefly import SeqReaderOnTheFly
from datasets.data_element import ArcticDataElement
from nnutils.handmocap import recover_pca_pose
from nnutils.image_utils import square_bbox


class ClipInfo(NamedTuple):
    vid: str  # sid_seq_name
    cat: str
    side: str  # 'left' or 'right'
    start: int 
    end: int
    fmt: str


EGO_VIEW = 0
class ArcticStableDatasetV1(Dataset):

    def __init__(self, 
                 image_sets: str,
                 sample_frames: int,
                 ablate_outside=False):
        """
        Args:
            ann_path: path to csv file
            ablate_outside:
                for Ablation study, if include some out-of-grasp frames
                use 30 frames (1 second) [st-30, ed+30]
        """
        self.hand_expansion = 0.4
        self.image_size = CROPPED_IMAGE_SIZE
        self.sample_frames = sample_frames
        self.ablate_outside = ablate_outside
        if image_sets == 'example_data':
            example_data_infos = [
                ClipInfo('s01/ketchup_grab_01', 'ketchup', 'left', 211, 237, "to-be-filled"),
            ]
            self.data_infos = example_data_infos
        else:
            self.data_infos = self.read_data_infos(image_sets)

        ioi_offset = dict()
        misc_info = io.read_json('DATA_STORAGE/arctic_data/meta/misc.json')
        for sid, info in misc_info.items():
            ioi_offset[sid] = info['ioi_offset']
        self.ioi_offset = ioi_offset

        self.seq_reader_cache = dict()  # ARCTIC raw_seqs is at most 248M, safe to store in memory

    def __len__(self):
        return len(self.data_infos)

    def locate_index_from_output(self, full_key) -> int:
        """ e.g. full_key = s01/ketchup_use_02_left_hand_423_462 """
        for i, info in enumerate(self.data_infos):
            fmt = f'{info.vid}_{info.side}_hand_{info.start}_{info.end}'
            if fmt in full_key or full_key in fmt:
                return i
        return None
    
    def read_data_infos(self, image_sets):
        """
        """
        df = pd.read_csv(image_sets)
        data_infos = []
        for i, row in df.iterrows():
            sid_seq_name = row.sid_seq_name
            obj_name = sid_seq_name.split('/')[1].split('_')[0]
            if self.ablate_outside:
                start, end = row.out_st, row.out_ed
            else:
                start, end = row.start, row.end
            data_infos.append(
                ClipInfo(row.sid_seq_name, obj_name, row.side, start, end, row.fmt))
        return data_infos

    def get_seq_reader(self, index):
        info = self.data_infos[index]
        sid_seq_name = info.vid
        if sid_seq_name in self.seq_reader_cache:
            return self.seq_reader_cache[sid_seq_name]
        # side = info.side
        cat = info.cat
        # start, end = info.start, info.end
        sid, seq_name = sid_seq_name.split('/')
        seq_reader = SeqReaderOnTheFly(
            sid=sid, seq_name=seq_name, obj_name=cat, obj_version='reduced')
        self.seq_reader_cache[sid_seq_name] = seq_reader
        return seq_reader
    
    def _get_camera(self, index) -> CameraManager:
        K_ego = self.get_seq_reader(index).K_ego[0]
        low_h = CROPPED_IMAGE_SIZE[1]
        low_w = CROPPED_IMAGE_SIZE[0]
        orig_h = 2000
        orig_w = 2800
        w_ratio = low_w / orig_w
        h_ratio = low_h / orig_h
        fx = K_ego[0, 0] * w_ratio
        fy = K_ego[1, 1] * h_ratio
        cx = K_ego[0, 2] * w_ratio
        cy = K_ego[1, 2] * h_ratio
        cam_manager = CameraManager(fx, fy, cx, cy, img_h=low_h, img_w=low_w)
        return cam_manager

    def _get_hand_box(self, hand_box, expand=True):
        if not expand:
            return hand_box
        hand_box_xyxy = xywh_to_xyxy(hand_box)
        hand_box_squared_xyxy = square_bbox(
            hand_box_xyxy[None], pad=self.hand_expansion)[0]
        w, h = self.image_size
        hand_box_squared_xyxy[:2] = hand_box_squared_xyxy[:2].clip(min=[0, 0])
        hand_box_squared_xyxy[2:] = hand_box_squared_xyxy[2:].clip(max=[w, h])
        hand_box_squared = xyxy_to_xywh(hand_box_squared_xyxy)
        return hand_box_squared
    
    def prepare_hand_3d_data(self, index, side):
        """ returns transformation from hand-to-ego

        Returns:
            gt_hand_pca: (N, 45)
            gt_hand_rot: (N, 6)
            gt_hand_transl: (N, 1, 3)
            gt_hand_betas: (N, 10)
        """
        seq_reader = self.get_seq_reader(index)
        T_h2e = seq_reader.pose_hand2ego(side)  # (N, 4, 4)
        rot_h2e = T_h2e[:, :3, :3]
        gt_rot_h2e = matrix_to_rotation_6d(rot_h2e)
        gt_transl_h2e = T_h2e[:, :3, 3].view(-1, 1, 3)
        # gt_pose_pca = seq_reader.pose_mano(side, is_pca=True)
        gt_pose = seq_reader.pose_mano(side, is_pca=False)
        gt_hand_betas = torch.from_numpy(
            seq_reader.mano_params[side]['shape']).view(1, 10).tile(len(gt_pose), 1)
        gt_pose_pca = recover_pca_pose(gt_pose, side)
        return gt_pose_pca, gt_rot_h2e, gt_transl_h2e, gt_hand_betas

    def get_mask_with_occlusion(self, 
                                seq_reader: SeqReaderOnTheFly, 
                                frame_idx: int,
                                side: str):
        """
        Returns:
            mask_hand, mask_obj: np.ndarray (H, W) int32
                1 fg, -1 ignore, 0 bg
        """
        mask = seq_reader.render_hand_object_mask(frame_idx)  # 0 for bg, 1 left, 2 right, 3 object
        mask_hand = np.zeros_like(mask).astype(np.int32)
        mask_obj = np.zeros_like(mask).astype(np.int32)
        mask_obj[mask == MASK_OBJ_ID] = 1
        mask_obj[mask == MASK_LEFT_ID] = -1
        mask_obj[mask == MASK_RIGHT_ID] = -1

        mask_hand[mask == MASK_OBJ_ID] = -1
        if side == LEFT:
            mask_hand[mask == MASK_LEFT_ID] = 1
            mask_hand[mask == MASK_RIGHT_ID] = -1
        elif side == RIGHT:
            mask_hand[mask == MASK_RIGHT_ID] = 1
            mask_hand[mask == MASK_LEFT_ID] = -1
        else:
            raise ValueError()
        return mask_hand, mask_obj

    def __getitem__(self, index, sample_frames=None) -> ArcticDataElement:
        """
        Args:
            sample_frames: if not None, will follow this number of frames,
                used during evaluation
    
        Returns: N is len(sample_frames)
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
        """
        info = self.data_infos[index]
        sid_seq_name = info.vid
        side = info.side
        cat = info.cat
        start, end = info.start, info.end
        sid, _ = sid_seq_name.split('/')
        seq_reader = self.get_seq_reader(index)
        sample_frames = self.sample_frames if sample_frames is None else sample_frames

        assert side in {LEFT, RIGHT}
        image_format = f'{ARCTIC_IMAGES_DIR}/{sid_seq_name}/{EGO_VIEW}/%05d.jpg'
        frame_indices = [v for v in np.linspace(
            start, end, num=sample_frames, dtype=int, endpoint=True)]
        keep_frame_indices = []
        for v in frame_indices:
            lbox, rbox, obox = seq_reader.get_boxes(v)
            hbox = lbox if side == LEFT else rbox
            if hbox is not None and obox is not None:
                keep_frame_indices.append(v)
        frame_indices = keep_frame_indices
        
        """ Load GT 3D data """
        # Assuming no articulation change, hence we can use any valid frame_idx, we use 0
        gt_hand_pca, gt_hand_rot, gt_hand_transl, gt_hand_betas = \
            self.prepare_hand_3d_data(index, side)
        v_obj, f_obj, T_o2l, T_o2r = seq_reader.neutralized_obj_params()  # (N, V, 3) (F, 3), (N, 4, 4), (N, 4, 4)
        v_obj = v_obj[start]  # (V, 3)
        T_o2h = T_o2l if side == LEFT else T_o2r
        T_o2h = T_o2h[frame_indices]  # (T, 4, 4)
        gt_obj2hand_rot = matrix_to_rotation_6d(T_o2h[:, :3, :3])
        gt_obj2hand_transl = T_o2h[:, :3, 3].view(-1, 1, 3)
        gt_hand_pca = gt_hand_pca[frame_indices]
        gt_hand_rot = gt_hand_rot[frame_indices]
        gt_hand_transl = gt_hand_transl[frame_indices]
        gt_hand_betas = gt_hand_betas[frame_indices]

        images = []
        hand_bbox_dicts = []
        obj_bbox_arrs = []
        hand_masks = []
        object_masks = []
        for frame_idx in frame_indices:
            image = np.asarray(Image.open(image_format % (frame_idx + self.ioi_offset[sid])))
            lbox, rbox, obox = seq_reader.get_boxes(frame_idx)
            bbox_o = obox
            # bbox_o = bboxes['object'][frame_idx]
            if side == LEFT:
                hand_box = self._get_hand_box(lbox)
                hand_bbox_dict = dict(right_hand=None, left_hand=hand_box)
            elif side == RIGHT:
                hand_box = self._get_hand_box(rbox)
                hand_bbox_dict = dict(right_hand=hand_box, left_hand=None)
            images.append(image)
            obj_bbox_arrs.append(bbox_o)
            hand_bbox_dicts.append(hand_bbox_dict)
            hand_mask, obj_mask = self.get_mask_with_occlusion(seq_reader, frame_idx, side)
            hand_masks.append(hand_mask)
            object_masks.append(obj_mask)

        side_return = side + '_hand'
        images = np.asarray(images)
        obj_bbox_arrs = torch.as_tensor(obj_bbox_arrs)
        hand_masks = torch.as_tensor(hand_masks)
        object_masks = torch.as_tensor(object_masks)
        global_cam = self._get_camera(index)
        batch_global_cam = global_cam.repeat(len(images), device='cpu')
        cat = 'arctic_' + cat
        element = ArcticDataElement(
            images=images, hand_bbox_dicts=hand_bbox_dicts, side=side_return,
            obj_bboxes=obj_bbox_arrs, hand_masks=hand_masks,
            object_masks=object_masks, cat=cat, global_camera=batch_global_cam,
            obj_verts=v_obj, obj_faces=f_obj, gt_hand_pca=gt_hand_pca, 
            gt_hand_rot=gt_hand_rot, gt_hand_trans=gt_hand_transl,
            gt_obj2hand_rot=gt_obj2hand_rot, gt_obj2hand_transl=gt_obj2hand_transl,
            gt_hand_betas=gt_hand_betas, obj_diameter=seq_reader.obj_diameter
            )
        return element


if __name__ == '__main__':
    dataset = ArcticStableDatasetV1(image_sets='example_data', sample_frames=30)
    e = dataset[0]
    print("Done")

