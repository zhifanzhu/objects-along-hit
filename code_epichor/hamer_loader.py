""" Load hamer from disk """

from collections import namedtuple
import os

import numpy as np
import torch
import cv2
from libzhifan.geometry import CameraManager, SimpleMesh, projection
from manopth.manolayer import ManoLayer
from pytorch3d.transforms import rotation_conversions as rot_cvt

from homan.manolayer_tracer import ManoLayerTracer
import pandas as pd
from datasets.epic_local_reader import EpicLocalReader
from homan.homan_ManoModel import HomanManoModel
from einops import repeat


EPIC_LOCAL_ROOT = 'DATA_STORAGE/epic/'


_ManoParams = namedtuple('ManoParams', 'pca_pose betas rot6d transl side')


# Src: hamer/datasets/utils.py
def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


def compute_hand_translation(focal, 
                             pred_cam, 
                             box, 
                             out_w, 
                             out_h,
                             ortho2persp=False):
    """
    Args:
        focal: scalar
        pred_cam: [s, tx, ty]
        box: [xmin, ymin, xmax, ymax]
        out_w: scalar
        out_h: scalar
        ortho2persp: orthorgraphic to perspective correction.
    
    Returns:
        global_transl: [tx, ty, tz]
        ortho2persp_rot: 3x3
    """
    rescale_factor = 2.0
    center = (box[2:4] + box[0:2]) / 2.0
    scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
    center_x = center[0]
    center_y = center[1]
    BBOX_SHAPE = [192, 256]
    bbox_size = expand_to_aspect_ratio(
        scale*200, target_aspect_ratio=BBOX_SHAPE).max()

    # Making global translation
    s, tx, ty = pred_cam  # params['pred_cam']
    x0 = center_x - bbox_size/2
    y0 = center_y - bbox_size/2
    # putting 1/s and xmin is equivalent to putting center_x
    tx = tx + 1/s + (2*x0-out_w)/(s*bbox_size+1e-9)
    ty = ty + 1/s + (2*y0-out_h)/(s*bbox_size+1e-9)
    tz = 2*focal/(s*bbox_size+1e-9)
    global_transl = torch.Tensor([tx, ty, tz])
    extra_rot = torch.eye(3, dtype=torch.float32, device=global_transl.device)

    if ortho2persp:
        s, tx, ty = pred_cam
        local_transl = torch.Tensor([tx, ty, 0])
        x0 = center_x - bbox_size/2
        y0 = center_y - bbox_size/2
        tx_g = 1/s + (2*x0-out_w)/(s*bbox_size+1e-9)  # putting 1/s and xmin is equivalent to putting center_x
        ty_g = 1/s + (2*y0-out_h)/(s*bbox_size+1e-9)
        tz_g = 2*focal/(s*bbox_size+1e-9)
        local2global_transl = torch.Tensor([tx_g, ty_g, tz_g])
        ortho2persp_rot = compute_sphere_geodesic_transform(
            np.array([0, 0, 1]), local2global_transl)
        ortho2persp_rot = torch.from_numpy(ortho2persp_rot).float().to(global_transl.device)
        global_transl = local2global_transl + local_transl
        extra_rot = ortho2persp_rot
    return global_transl, extra_rot


class HamerLoader:
    """ Load hamer from disk """

    def __init__(self,
                 ho_version: str,
                 mano_root='externals/mano',
                 load_only=True,
                 image_sets='code_epichor/image_sets/epicgrasps_2431.csv',
                 device='cuda'):
        """
        Args:
            ho_version: 'v1' or 'hoa_potim'
                        or 'v1-diagnosehands' (original v1 saved in barry)
            load_dir: e.g. <load_dir>/P01_01/frame_0000012345.pt
        """
        self.device = device
        self.ho_version = ho_version
        if ho_version in {'v1', 'v1-diagnosehands'}:
            df = pd.read_csv(image_sets)
            hoa_cache = torch.load('./DATA_STORAGE/epicgrasps_storage/cache/hoa_hbox.pth')  # xywh in 1920x1080
            if ho_version == 'v1':  # DATA_STORAGE release
                self.load_dir = './DATA_STORAGE/epicgrasps_storage/hamer_hov1'
            elif ho_version == 'v1-diagnosehands':  # original results on barry
                self.load_dir = TODO # '/media/barry/DATA/Zhifan/epic_hor_data/hamer_hov1'

            def to_hierarchy(df, hoa_cache):
                all_boxes = dict()
                for i, row in df.iterrows():
                    mp4_name = row['mp4_name']
                    vid = row['vid']
                    side = row['handside'].replace(' hand', '')
                    if vid not in all_boxes:
                        all_boxes[vid] = dict()
                    if side not in all_boxes[vid]:
                        all_boxes[vid][side] = dict()
                    if mp4_name not in hoa_cache:
                        print(f'{mp4_name} need to re-run hoa_cache')
                        continue
                    for frame, box in hoa_cache[mp4_name].items():
                        all_boxes[vid][side][frame] = box
                return all_boxes
            
            self.all_boxes = to_hierarchy(df, hoa_cache)
            self.box_resolution = np.float32([1920, 1080] * 2)

        elif ho_version == 'hoa_potim':
            self.load_dir = f'{EPIC_LOCAL_ROOT}/hamer_hoa_potim'
            self.box_resolution = np.float32([1920, 1080] * 2)

        else:
            raise ValueError(f"Unkown {ho_version=}")

        if not load_only:  # for debugging
            self._setup_for_debug(mano_root)
        
        self.l_mano_tracer = ManoLayerTracer(
            flat_hand_mean=False, ncomps=45, side='left', use_pca=True,
            mano_root='externals/mano/').to(self.device)
        self.r_mano_tracer = ManoLayerTracer(
            flat_hand_mean=False, ncomps=45, side='right', use_pca=True,
            mano_root='externals/mano/').to(self.device)
    
    def _setup_for_debug(self, mano_root='./externals/mano'):
        # self.reader = Reader('./DATA_STORAGE/epicgrasps_storage')
        self.reader = EpicLocalReader()
        self.left_mano = ManoLayer(
            flat_hand_mean=True, ncomps=45, side='left',
            mano_root=mano_root, use_pca=False)
        self.right_mano = ManoLayer(
            flat_hand_mean=True, ncomps=45, side='right',
            mano_root=mano_root, use_pca=False)

    def load_frame_all_params(self, vid: str, frame: int) -> dict:
        """ Load both hand params from *.pt file """
        return torch.load(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')
    
    def avail_side_frames(self, vid) -> list:
        """ return: avail frames for left / right differently """
        avail_frames = dict(left=[], right=[]) 
        fns = os.listdir(f'{self.load_dir}/{vid}')
        for fn in fns:
            frame = int(fn[6:-3])
            fpath = os.path.join(self.load_dir, vid, fn)
            params = torch.load(fpath)
            if 'left' in params:
                avail_frames['left'].append(frame)
            if 'right' in params:
                avail_frames['right'].append(frame)
        avail_frames['left'] = sorted(avail_frames['left'])
        avail_frames['right'] = sorted(avail_frames['right'])
        return avail_frames

    def avail_frames(self, vid: str) -> list:
        """ Return available frames for a video """
        print("Using avail_side_frames instead.")
        return sorted([
            int(f.split('.')[0].split('_')[-1])
            for f in os.listdir(f'{self.load_dir}/{vid}')])

    def has_frame(self, vid: str, frame: int) -> bool:
        """ Check if a frame is available """
        return os.path.exists(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')

    def _load_hamer_box(self, vid, frame, is_left) -> np.ndarray:
        """ return xyxy in 1920x1080 / 256x456 resolution """
        if self.ho_version in {'v1', 'v1-diagnosehands'}:
            x1, y1, w, h = self.all_boxes[vid]['left' if is_left else 'right'][frame]
            box = np.array([x1, y1, x1+w, y1+h])
            return box
        
        elif self.ho_version == 'hoa_potim':
            if not self.has_frame(vid, frame):
                return None
            all_params = torch.load(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')
            hand_side = 'left' if is_left else 'right'
            if hand_side not in all_params:
                return None
            box_xyxy_456x256 = all_params[hand_side]['box_xyxy_456x256'].numpy()
            _box_scale = np.float32([1920, 1080, 1920, 1080]) / np.float32([456, 256, 456, 256])
            box_xyxy_1920x1080 = box_xyxy_456x256 * _box_scale
            return box_xyxy_1920x1080

    def _load_hamer_params(self, vid, frame, is_left) -> dict:
        """ src: <hamer-repo>/infer_epic/infer.py"""
        if not self.has_frame(vid, frame):
            return None
        all_params = torch.load(f'{self.load_dir}/{vid}/frame_{frame:010d}.pt')
        hand_side = 'left' if is_left else 'right'
        if hand_side not in all_params:
            return None
        return all_params[hand_side]

    def get_hamer_parmas(self, global_cam, vid, frame_inds: list, is_left: bool, 
                         ortho2persp=False):
        """  This will return params to be used by
        HomanManoModel(mean = False, pca = True)

        Originally HAMER renders with mean=True and pca=False, we do the convertion

        Returns:
            mano_pca_pose: (N, 45)
            mano_hand_betas: (N, 10)
            hand_rotation_6d: (N, 6) apply-to-col
            hand_translation: (N, 1, 3)
        """
        out_w = global_cam.img_w
        out_h = global_cam.img_h
        box_scale = np.float32([out_w, out_h, out_w, out_h]) / self.box_resolution

        hand_poses = [None for _ in frame_inds]
        mano_betas = [None for _ in frame_inds]
        global_orients = [None for _ in frame_inds]
        global_translations = [None for _ in frame_inds]
        ortho2persp_rots = [None for _ in frame_inds]

        for i, frame in enumerate(frame_inds):
            box = self._load_hamer_box(vid, frame, is_left)
            box = box * box_scale
            params = self._load_hamer_params(vid, frame, is_left)
            hand_poses[i] = rot_cvt.matrix_to_axis_angle(params['hand_pose']).view(1, 45)
            global_orients[i] = rot_cvt.matrix_to_axis_angle(params['global_orient'])
            global_translations[i], ortho2persp_rots[i] = compute_hand_translation(
                global_cam.fx, params['pred_cam'], box, out_w, out_h,
                ortho2persp=ortho2persp)
            mano_betas[i] = params['betas'].view(1, 10)

        global_orients = torch.cat(global_orients, axis=0)  # (N, 3)
        hand_poses = torch.cat(hand_poses, axis=0)  # (N, 45)
        global_translations = torch.cat(global_translations, axis=0)  # (N, 3)
        ortho2persp_rots = torch.stack(ortho2persp_rots, axis=0)  # (N, 3, 3)
        mano_betas = mano_betas[0].repeat(len(frame_inds), 1)  # Shape follows first frame
        if is_left:
            global_orients[:, 1:] *= -1
            hand_poses = hand_poses.view(-1, 15, 3) * torch.Tensor([1, -1, -1])
            hand_poses = hand_poses.view(-1, 45)
        hand_poses = hand_poses.to(self.device)
        mano_betas = mano_betas.to(self.device)
        global_orients = global_orients.to(self.device)
        global_translations = global_translations.to(self.device)
        ortho2persp_rots = ortho2persp_rots.to(self.device)
        thetas = torch.cat([global_orients, hand_poses], axis=1)

        mano_tracer = self.l_mano_tracer if is_left else self.r_mano_tracer
        _, _, T_world = mano_tracer.forward_transform(thetas, root_palm=True)

        T_ortho_correction = torch.eye(4, device=T_world.device).unsqueeze(0).repeat(len(frame_inds), 1, 1)
        T_ortho_correction[:, :3, :3] = ortho2persp_rots
        T_world = T_ortho_correction @ T_world

        T_rot = T_world[:, :3, :3]
        T_transl = T_world[:, :3, -1]
        hand_translation = T_transl.view(-1, 1, 3) + global_translations.view(-1, 1, 3)
        hand_rotation_6d = rot_cvt.matrix_to_rotation_6d(T_rot)

        # mano_pca_pose = recover_pca_pose(hand_poses, side='left' if is_left else 'right')
        M_pca_inv = torch.inverse(mano_tracer.th_comps)
        mano_pca_pose = (hand_poses - mano_tracer.th_hands_mean).mm(M_pca_inv)
        side = 'left' if is_left else 'right'

        return _ManoParams(
            mano_pca_pose, mano_betas, hand_rotation_6d, hand_translation,
            side)
    

    """ Below are for various debugging. """

    def visualise_frame(self,
                        vid: str, frame: int,
                        hand_side: str,
                        out_h=256, out_w=456,
                        epic_focal=5000,
                        ortho2persp=False,
                        ret_mesh=False) -> np.ndarray:
        """
        Assumes each *.pt contains:
        dict(left=params, right=params)
        params = dict(
            pred_cam
                Tensor torch.Size([3])
            global_orient
                Tensor torch.Size([1, 3, 3])
            hand_pose
                Tensor torch.Size([15, 3, 3])
            betas
                Tensor torch.Size([10])
        )

        Args:
            hand_side: 'left' or 'right'
            epic_focal: by Hamer's default, this is 5000
        """
        # DO NOT DELETE THIS FUNCTIN YET! STILL USEFUL FOR DEBUGGING
        # raise NotImplementedError
        box = self._load_hamer_box(vid, frame, 'left' in hand_side)
        params = self._load_hamer_params(vid, frame, 'left' in hand_side)
        img_pil = self.reader.read_image_pil(vid, frame).resize([out_w, out_h])

        glb_orient = rot_cvt.matrix_to_axis_angle(params['global_orient'])
        thetas = rot_cvt.matrix_to_axis_angle(
            params['hand_pose']).view([1, 45])
        thetas = torch.cat([glb_orient, thetas], axis=1)
        betas = params['betas'].view([1, 10])

        if hand_side == 'left':
            mano = self.left_mano
            # The following is to flip the hand (flipping y and z)
            thetas = thetas.view(16, 3)
            thetas[:, 1:] *= -1
            thetas = thetas.view([1, 48])
        elif hand_side == 'right':
            mano = self.right_mano
        box_scale = np.float32([out_w, out_h, out_w, out_h]) / self.box_resolution
        box *= box_scale

        vh, jh = mano.forward(thetas, betas)
        vh /= 1000.
        jh /= 1000.

        global_cam = CameraManager(
            fx=epic_focal, fy=epic_focal, cx=out_w//2, cy=out_h//2,
            img_h=out_h, img_w=out_w)
        global_transl, ortho2persp_rot = compute_hand_translation(
            epic_focal, params['pred_cam'], box, out_w, out_h,
            ortho2persp=ortho2persp)

        if ortho2persp:
            mesh = SimpleMesh(vh @ ortho2persp_rot.T + global_transl, mano.th_faces)
        else:
            mesh = SimpleMesh(vh + global_transl, mano.th_faces)
        if ret_mesh:
            return mesh
        rend = projection.perspective_projection_by_camera(
            mesh, global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        return rend

    def visualise_frame_homan(self,
                              vid: str, frame: int,
                              hand_side: str,
                              out_h=256, out_w=456,
                              epic_focal=5000,
                              ret_mesh=False) -> np.ndarray:
        """ This implement the forward pass with HomanManoModel(mean=False, pca=True)
        note: visualise_frame() uses ManoLayer(mean=True, pca=False)

        Args:
            hand_side: 'left' or 'right'
        """
        img_pil = self.reader.read_image_pil(vid, frame).resize([out_w, out_h])
        global_cam = CameraManager(
            fx=epic_focal, fy=epic_focal, cx=out_w//2, cy=out_h//2,
            img_h=out_h, img_w=out_w)
        
        mano_tracer = self.l_mano_tracer if hand_side == 'left' else self.r_mano_tracer
        mano_pca_pose, mano_betas, hand_rotation_6d, hand_translation, side = \
            self.get_hamer_parmas(global_cam, vid, [frame], hand_side == 'left')
        th_pose_coeffs = torch.cat(
            [mano_pca_pose.new_zeros([1, 3]), mano_pca_pose], -1)
        vh, _ = mano_tracer.forward(th_pose_coeffs, mano_betas)  # (1, 778, 3)
        vh = vh / 1000.

        rot_mat = rot_cvt.rotation_6d_to_matrix(hand_rotation_6d)
        vh = vh @ rot_mat.permute(0, 2, 1) + hand_translation

        mesh = SimpleMesh(vh[0], mano_tracer.th_faces)
        if ret_mesh:
            return mesh
        rend = projection.perspective_projection_by_camera(
            mesh, global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        return rend

    def visualise_frame_homan_real(self,
                                   vid: str, frame: int,
                                   hand_side: str,
                                   out_h=256, out_w=456,
                                   epic_focal=5000,
                                   ortho2persp=False,
                                   ret_mesh=False) -> np.ndarray:
        """ This uses real HomanManoModel(mean=False, pca=True)

        Args:
            hand_side: 'left' or 'right'
        """
        img_pil = self.reader.read_image_pil(vid, frame).resize([out_w, out_h])
        global_cam = CameraManager(
            fx=epic_focal, fy=epic_focal, cx=out_w//2, cy=out_h//2,
            img_h=out_h, img_w=out_w)
        
        mano_pca_pose, mano_betas, hand_rotation_6d, hand_translation, side = \
            self.get_hamer_parmas(global_cam, vid, [frame], hand_side == 'left', ortho2persp=ortho2persp)

        if hand_side == 'left':
            mano_homan = HomanManoModel(
                "externals/mano", side='left', pca_comps=45)  # Note(zhifan): in HOMAN num_pca = 16
        elif hand_side == 'right':
            mano_homan = HomanManoModel(
                "externals/mano", side='left', pca_comps=45)  # Note(zhifan): in HOMAN num_pca = 16
        else:
            raise ValueError(f"Bad {hand_side=}")

        num_samples = 1
        mano_rot = torch.zeros([num_samples, 3], device=mano_pca_pose.device)
        mano_trans = torch.zeros([num_samples, 3], device=mano_pca_pose.device)
        mano_res = mano_homan.forward_pca(
            mano_pca_pose,
            rot=mano_rot,
            betas=mano_betas,
            side=side)
        vertices = mano_res["verts"]
        verts_hand_og = vertices + mano_trans.view(-1, 1, 3)

        T_h2c = repeat(torch.eye(4), 'd e -> t d e', t=num_samples).clone()
        R_h2c = rot_cvt.rotation_6d_to_matrix(hand_rotation_6d)
        t_h2c = hand_translation
        T_h2c[:, :3, :3] = R_h2c
        T_h2c[:, :3, 3] = t_h2c.view(-1, 3)

        verts_hand_ego = verts_hand_og @ R_h2c.permute(0, 2, 1) + t_h2c

        mesh = SimpleMesh(verts_hand_ego[0], mano_homan.hand_faces)
        if ret_mesh:
            return mesh
        rend = projection.perspective_projection_by_camera(
            mesh, global_cam,
            method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False),
            image=np.asarray(img_pil))
        return rend

    
def compute_sphere_geodesic_transform(vec0, vec1):
    """ return 3x3 rotation to transform vec0 to vec1 """
    def normalise(vec):
        return vec / np.linalg.norm(vec)
    vec0 = normalise(vec0)
    vec1 = normalise(vec1)
    axis = normalise(np.cross(vec0, vec1))
    angle = np.arccos(vec0.dot(vec1))
    rotvec = axis * angle
    rot_mat, _ = cv2.Rodrigues(rotvec)
    return rot_mat