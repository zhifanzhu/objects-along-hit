import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
from libzhifan import io, odlib
from libzhifan.geometry import CameraManager, SimpleMesh, projection, visualize_mesh
from manopth.manolayer import ManoLayer
from PIL import Image

from code_arctic.arctic_obj_loader import ArcticOBJLoader
from code_arctic.data_reader import (
    ARCTIC_IMAGES_DIR,
    CROPPED_IMAGE_SIZE,
    LEFT,
    MASK_LEFT_ID,
    MASK_OBJ_ID,
    MASK_RIGHT_ID,
    RIGHT,
    _get_transform,
    bbox_from_mask,
    orig_reso,
    pose_apply,
    rot_transl_apply,
)
from homan.manolayer_tracer import ManoLayerTracer

odlib.setup('xywh', norm=False)


""" Original: code_arctic/data_reader.py"""

RAW_SEQS_DIR = 'DATA_STORAGE/arctic_data/raw_seqs'
egocam_fmt = f'{RAW_SEQS_DIR}/{{sid}}/{{seq_name}}.egocam.dist.npy'
mano_fmt = f'{RAW_SEQS_DIR}/{{sid}}/{{seq_name}}.mano.npy'
object_fmt = f'{RAW_SEQS_DIR}/{{sid}}/{{seq_name}}.object.npy'
smplx_fmt = f'{RAW_SEQS_DIR}/{{sid}}/{{seq_name}}.smplx.npy'
ARCTIC_OUT_DIR = 'DATA_STORAGE/arctic_outputs'
mask_fmt = f'{ARCTIC_OUT_DIR}/masks_low/{{sid}}/{{seq_name}}/{{frame_idx:05d}}.png'
box_fmt = f'{ARCTIC_OUT_DIR}/masks_low/{{sid}}/{{seq_name}}_boxes.csv'


NEUTRAL_TRANSFORMS = {
    'ketchup': np.float32([
        [0, 0, -1, 0],
        [0, 1, 0, -0.02],
        [1, 0, 0, 0.15],
        [0, 0, 0, 1]]),
    'box': np.float32([
        [-0.01939892,  0.01211364, -0.99973845,  0.07696321],
       [ 0.69330317, -0.7203046 , -0.02218065,  0.0230532 ],
       [-0.7203849 , -0.6935521 ,  0.0055747 ,  0.11971147],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    'capsulemachine': np.float32([
        [-0.02659027, -0.02621649, -0.99930257,  0.03653535],
        [ 0.5630728, -0.82638013,  0.0066972,   0.00475189],
        [-0.8259794, -0.562502,    0.03673546,  0.09327681],
        [ 0.,          0.,          0.,          1.        ]]
    ),
    'espressomachine': np.float32(
        [[-0.8121901 , -0.5830121 ,  0.02107439, -0.05360536],
       [ 0.58309764, -0.8123983 , -0.00246361,  0.00557016],
       [ 0.01855711,  0.01028751,  0.9997749 ,  0.07177647],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
    ),
    'laptop': np.float32(
        [[-1.7244322e-02, -9.4035792e-04, -9.9985087e-01,  2.1517918e-02],
       [ 9.8581499e-01,  1.6695632e-01, -1.7159268e-02, -9.1749534e-02],
       [ 1.6694754e-01, -9.8596382e-01, -1.9520295e-03,  3.4809774e-03],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
    ),
    'microwave': np.float32(
        [[-3.8236716e-01, -9.2401040e-01, -3.7836816e-04,  9.4636939e-02],
       [ 9.2398947e-01, -3.8235572e-01, -6.8950341e-03,  7.6281339e-02],
       [ 6.2264120e-03, -2.9860430e-03,  9.9997616e-01,  5.9353221e-02],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
    ),
    'mixer': np.float32(
        [[ 0.01333138, -0.01478844, -0.99980175, -0.00625887],
       [ 0.16188033,  0.986732  , -0.01243661, -0.02879081],
       [ 0.9867203 , -0.16168244,  0.01554846,  0.05007098],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
    ),
    'notebook': np.float32(
        [[ 0.9880046 ,  0.15441005,  0.00210828,  0.07832433],
       [-0.15442444,  0.98791254,  0.01348873,  0.01424514],
       [ 0.        , -0.0136525 ,  0.9999068 ,  0.05121864],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
    ),
    'phone': np.float32(
        [[ 4.0270938e-03,  1.1405205e-02, -9.9992687e-01, -1.4395401e-02],
        [ 1.4034447e-01,  9.9003172e-01,  1.1857563e-02, -7.1525574e-10],
        [ 9.9009454e-01, -1.4038196e-01,  2.3862929e-03,  4.5689765e-02],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
    ),
    'scissors': np.float32(
        [[-7.4740551e-02, -9.9720299e-01, -7.5286735e-08,  2.1798150e-03],
       [ 0.0000000e+00,  7.5497901e-08, -1.0000000e+00,  3.2598341e-03],
       [ 9.9720299e-01, -7.4740551e-02, -5.6427547e-09,  5.7451673e-02],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]
    ),
    'waffleiron': np.float32(
        [[-0.03142943, -0.00654844, -0.99948454,  0.04744862],
       [ 0.15219834, -0.98834854,  0.0016895 , -0.00217322],
       [-0.9878501 , -0.1520668 ,  0.03205989,  0.10105472],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
    )
}


class SeqReaderOnTheFly:
    def __init__(self, 
                 sid: str,
                 seq_name: str,
                 obj_name: str, 
                 obj_version: str,
                 view_id=0,
                 preload_obj_faces=False,
                 storage_dir='./DATA_STORAGE/'):
        """ A reader that output data on the fly rather than using processed/processed_verts

        hand's rot_r is in m space

        Args:
            view_id: camera id, 0 is ego camera
            obj_version: 'original' or 'reduced'
        """
        self.images_dir = osp.join(ARCTIC_IMAGES_DIR, sid, seq_name)
        self.sid = sid
        self.seq_name = seq_name
        self.view_id = view_id

        self.ioi_offset = io.read_json(
            osp.join(storage_dir, 'arctic_data/meta/misc.json')
            )[sid]['ioi_offset']

        box_df = pd.read_csv(box_fmt.format(sid=sid, seq_name=seq_name))  # cached boxes
        box_df = box_df.where(pd.notnull(box_df), None)
        func = lambda x: np.asarray(
             [int(v) for v in x.replace('[','').replace(']','').split()])
        self.lboxes = box_df['lbox'].map(lambda x: func(x) if x is not None else None)
        self.rboxes = box_df['rbox'].map(lambda x: func(x) if x is not None else None)
        self.oboxes = box_df['obox'].map(lambda x: func(x) if x is not None else None)

        """ left / right hand"""
        self.l_mano_layer = ManoLayer(
            flat_hand_mean=False, ncomps=45, side='left', use_pca=False,
            mano_root='./externals/mano/')
        self.r_mano_layer = ManoLayer(
            flat_hand_mean=False, ncomps=45, side='right', use_pca=False,
            mano_root='./externals/mano/')
        self.fl = self.l_mano_layer.th_faces.cpu().numpy()
        self.fr = self.r_mano_layer.th_faces.cpu().numpy()

        self.obj_loader = ArcticOBJLoader(version=obj_version)
        self.obj_name = obj_name
        self.obj_diameter = io.read_json(
            'DATA_STORAGE/arctic_data/meta/object_meta.json')[obj_name]['diameter']
        if preload_obj_faces:
            self.fo = self.obj_loader.load_obj_unarticulated(obj_name, return_mesh=True)[0].faces

        """ Params """
        self.mano_params = np.load(mano_fmt.format(sid=sid, seq_name=seq_name), allow_pickle=True).item()

        object_params = np.load(object_fmt.format(sid=sid, seq_name=seq_name), allow_pickle=True)
        self.obj_arti = torch.from_numpy(object_params[:, 0])
        self.obj_rot = torch.from_numpy(object_params[:, 1:4])
        self.obj_trans = torch.from_numpy(object_params[:, 4:])

        egocam = np.load(egocam_fmt.format(sid=sid, seq_name=seq_name), allow_pickle=True).item()
        self.num_frames = len(egocam['R_k_cam_np'])
        self.K_ego = torch.FloatTensor(egocam['intrinsics']).view(1, 3, 3).tile(self.num_frames, 1, 1)
        R_ego = torch.FloatTensor(egocam["R_k_cam_np"])
        T_ego = torch.FloatTensor(egocam["T_k_cam_np"])
        world2ego = torch.zeros((self.num_frames, 4, 4))
        world2ego[:, :3, :3] = R_ego
        world2ego[:, :3, 3] = T_ego.view(self.num_frames, 3)
        world2ego[:, 3, 3] = 1
        self.world2ego = world2ego

    def render_image(self, frame_idx) -> np.ndarray:
        frame_idx = frame_idx + self.ioi_offset
        img_path = osp.join(self.images_dir, str(self.view_id), f'{frame_idx:05d}.jpg')
        return np.asarray(Image.open(img_path))
    
    def render_image_mesh(self, frame_idx, 
                          with_rend=True,
                          with_triview=False,
                          side=None):
        img = self.render_image(frame_idx)/255
        if with_rend:
            rend = self.render_mesh(frame_idx)
            out = np.hstack([img, rend])
            out = cv2.resize(out, [840, 300])
        else:
            out = cv2.resize(img, [420, 300])
        if with_triview:
            assert side is not None
            triview = self.render_triview(frame_idx, side, coor_sys='pytorch3d')
            out = np.hstack([out, triview])
        return out
    
    def render_image_boxes(self, frame_idx) -> Image.Image:
        lbox, rbox, obox = self.get_boxes(frame_idx)
        img_pil = Image.fromarray(self.render_image(frame_idx))
        img_pil = odlib.draw_bboxes_image_array(img_pil, obox[None], color='red')
        img_pil = odlib.draw_bboxes_image_array(img_pil, lbox[None], color='green')
        img_pil = odlib.draw_bboxes_image_array(img_pil, rbox[None], color='blue')
        return np.asarray(img_pil)

    def render_hand_object_mask(self, frame_idx, use_disk=True):
        """ 
        Returns:
            sil: (H, W), dtype=np.uint8
                0 for bg, 1 left, 2 right, 3 object
        """
        if use_disk:
            mask = np.asarray(
                Image.open(mask_fmt.format(sid=self.sid, seq_name=self.seq_name, frame_idx=frame_idx)))
            return mask

        vl = self.hand_verts(frame_idx, 'left', space='ego', as_mesh=False)
        vr = self.hand_verts(frame_idx, 'right', space='ego', as_mesh=False)
        left = SimpleMesh(verts=vl, faces=self.fl, tex_color=(0, 1.0, 0))
        right = SimpleMesh(verts=vr, faces=self.fr, tex_color=(0, 0, 1.0))
        K_ego = self.K_ego[frame_idx, ...]
        img_w, img_h = CROPPED_IMAGE_SIZE
        w_ratio = img_w / orig_reso[0]
        h_ratio = img_h / orig_reso[1]
        fx = K_ego[0, 0] * h_ratio  # This is in-fact confusing: why not w_ratio?
        fy = K_ego[1, 1] * w_ratio
        cx = K_ego[0, 2] * h_ratio
        cy = K_ego[1, 2] * w_ratio
        cam_manager = CameraManager(fx, fy, cx, cy, img_h=img_h, img_w=img_w)

        proj_method=dict(name='pytorch3d_instance', coor_sys='nr', in_ndc=False,
            blur_radius=1e-7)
        obj_cam = self.obj_verts(frame_idx, space='ego', as_mesh=True)
        # left-1, right-2, obj-3
        rend = projection.perspective_projection_by_camera(
            [left, right, obj_cam],
            camera=cam_manager,
            method=proj_method)  

        rend = rend.astype(np.uint8)
        return rend

    def get_boxes_and_mask(self, frame_idx, use_disk=True):
        """
        Returns:
            lbox, rbox, obox: [x, y, w, h] in pixel
                can be None
            mask: (H, W), dtype=np.uint8
                0 for bg, 1 left, 2 right, 3 object
        """
        mask = self.render_hand_object_mask(frame_idx=frame_idx, use_disk=use_disk)
        lbox, rbox, obox = self.get_boxes(frame_idx, use_disk=use_disk)
        return mask, lbox, rbox, obox

    def get_boxes(self, frame_idx, use_disk=True):
        """
        Returns:
            lbox, rbox, obox: [x0, y0, w, h] in pixel
                can be None
        """
        if use_disk:
            lbox = self.lboxes[frame_idx]
            rbox = self.rboxes[frame_idx]
            obox = self.oboxes[frame_idx]
            return lbox, rbox, obox
        mask = self.render_hand_object_mask(frame_idx=frame_idx)
        obox = bbox_from_mask(mask, MASK_OBJ_ID)
        lbox = bbox_from_mask(mask, MASK_LEFT_ID)
        rbox = bbox_from_mask(mask, MASK_RIGHT_ID)
        return lbox, rbox, obox
    
    # def get_camera(self, frame_idx):
    #     """
    #     Returns:

    #     """
    #     raise NotImplementedError
    
    def transform_space(self, v, space):
        """
        Transform from World-coor into `space`-coor

        Args:
            v: (N, V, 3)

        Returns:
            v: (N, V, 3)
        """
        if space == 'ego':
            T_w2e = self.world2ego
            v = pose_apply(T_w2e, v)
        elif space == 'world':
            # Identity
            pass
        elif space == 'left' or space == 'right':
            T_h_world = self.pose_hand2world(side=space)
            v = pose_apply(torch.inverse(T_h_world), v)
        elif space == 'obj':
            T_o_world = _get_transform(self.obj_rot, 
                                    self.obj_trans/1000)
            v = pose_apply(torch.inverse(T_o_world), v)
        else:
            raise ValueError(f'Unknown space {space}')
        return v

    def hand_tips(self, side, frame_idx=None, space='ego'):
        """
        Returns:
            (N, 21, 3) hand tips in meter space
        """
        def forward_hand_verts_myimpl(side):
            assert side in {LEFT, RIGHT}, f'Unknown side {side}'
            mano_layer = self.l_mano_layer if side == LEFT else self.r_mano_layer
            pose_h = torch.from_numpy(self.mano_params[side]['pose'])
            shape_h = torch.from_numpy(self.mano_params[side]['shape']).view(1, 10).tile(self.num_frames, 1)
            rot_h = torch.from_numpy(self.mano_params[side]['rot'])
            trans_h = torch.from_numpy(self.mano_params[side]['trans'])
            rot_zeros = torch.zeros_like(rot_h)
            trans_zeros = torch.zeros_like(trans_h)
            _, j = mano_layer.forward(
                torch.cat([rot_zeros, pose_h], axis=-1), 
                th_betas=shape_h, th_trans=trans_zeros, root_palm=True)
            j /= 1000  # in meter

            _, _, T_world = ManoLayerTracer(
                flat_hand_mean=False, ncomps=45, side=side, use_pca=False,
                mano_root='./externals/mano/').forward_transform(
                torch.cat([rot_h, pose_h], axis=-1), 
                th_betas=shape_h, th_trans=trans_h, root_palm=True)
            j = pose_apply(T_world, j)
            return j, mano_layer.th_faces

        # world space
        j, _ = forward_hand_verts_myimpl(side)
        j = self.transform_space(j, space)

        if frame_idx is not None:
            j = j[frame_idx]
        return j

    def hand_verts(self, frame_idx, side, space='ego', as_mesh=False,
                   zero_rot_trans=False, tex_color='light_blue'):
        """
        Args:
            space: 'world' or 'ego'
            zero_rot_trans: This is for preparing Dataloader input in MVHO,
                shouldn't impact output results.
        
        Returns:
            (N, 778, 3) if frame_idx is None, otherwise (778, 3)
        """
        def forward_hand_verts(side, zero_rot_trans):
            """
            zero_rot_trans: if True, return in hand-space, else world-space
            """
            mano_layer = self.l_mano_layer if side == LEFT else self.r_mano_layer
            pose_h = torch.from_numpy(self.mano_params[side]['pose'])
            shape_h = torch.from_numpy(self.mano_params[side]['shape']).view(1, 10).tile(self.num_frames, 1)
            rot_h = torch.from_numpy(self.mano_params[side]['rot'])
            trans_h = torch.from_numpy(self.mano_params[side]['trans'])
            if zero_rot_trans:
                rot_h = torch.zeros_like(rot_h)
                trans_h = None
            th_pose_coeffs = torch.cat([
                rot_h,
                pose_h], axis=-1)
            v, _ = mano_layer.forward(
                th_pose_coeffs, th_betas=shape_h, th_trans=trans_h)
            v /= 1000
            return v, mano_layer.th_faces

        # world space
        v, f = forward_hand_verts(side, zero_rot_trans=zero_rot_trans)
        if zero_rot_trans:
            # In this case hand is in hand-space
            T_h2e = self.pose_hand2ego(side=side)
            v = pose_apply(T_h2e, v)
            # v = pose_apply(T_hand2world, v)
            # v = self.transform_space(v, space=space)
        else:
            v = self.transform_space(v, space=space)

        if as_mesh:
            return SimpleMesh(v[frame_idx], f, tex_color=tex_color)
        else:
            return v if frame_idx is None else v[frame_idx] 

    def obj_verts(self, frame_idx, space='ego', as_mesh=False, with_faces=False,
                  tex_color='red'):
        """
        This doesn't handle the articulation

        obj_trans is in mm

        Returns: (N, V, 3) if frame_idx is None else (V, 3)
        """
        N = self.num_frames
        T_o_world = _get_transform(self.obj_rot, 
                                   self.obj_trans/1000)
        vo, fo = self.obj_loader.batch_articulate(
            name=self.obj_name, artis=self.obj_arti)  # (N, V, 3)
        vo = pose_apply(T_o_world, vo)
        vo = self.transform_space(vo, space)

        if frame_idx is None:
            # All frames
            assert as_mesh is False
            if with_faces:
                return vo, fo
            return vo
        else:
            vo = vo[frame_idx]
            if as_mesh:
                return SimpleMesh(verts=vo, faces=fo, tex_color=tex_color)
            else:
                if with_faces:
                    return vo, torch.from_numpy(fo)
                return vo
    
    def neutralized_obj_params(self, debug=False):
        """ Placing the object s.t. z-axis is the symmetric axis, and center is around origin
        return the transformed pose_obj2hand accordingly
        i.e. T_obj2hand_original = T_obj2hand_neutralized @ T_neutral

        Returns:
            vo: (N, V, 3) in neutralized obj space
            fo: (F, 3)
            T_o2l_neutral, T_o2r_neutral: (N, 4, 4)
            T_neutral: (4, 4)
        """
        vo, fo = self.obj_verts(frame_idx=None, space='obj', with_faces=True)
        T_neutral = torch.as_tensor(NEUTRAL_TRANSFORMS[self.obj_name]).view(1, 4, 4)
        inv_T_neutral = torch.inverse(T_neutral)
        T_o2l, T_o2r = self.pose_obj2hand()
        T_o2l_neutral = torch.matmul(T_o2l, inv_T_neutral)
        T_o2r_neutral = torch.matmul(T_o2r, inv_T_neutral)
        vo_neutral = pose_apply(T_neutral.tile(len(vo), 1, 1), vo)
        if debug:
            vr = self.hand_verts(551, 'right', space='right', as_mesh=False)
            right = SimpleMesh(verts=vr, faces=self.fr, tex_color=(0, 0, 1.0))
            vo_right = pose_apply(T_o2r_neutral, vo_neutral)
            obj = SimpleMesh(verts=vo_right[551], faces=fo)
            meshes = [obj, right]
            return visualize_mesh(meshes, show_axis=True, viewpoint='nr')
        return vo_neutral, fo, T_o2l_neutral, T_o2r_neutral
    
    def pose_mano(self, side, is_pca):
        """ returns (N, 45) """
        pose_h = torch.from_numpy(self.mano_params[side]['pose']).cuda()  # joints
        if is_pca:
            from nnutils.handmocap import get_hand_wrapper
            hand_wrapper = get_hand_wrapper('left' if side == LEFT else 'right')
            pose_pca = hand_wrapper.pose_to_pca(pose_h)
            return pose_pca
        else:
            return pose_h
    
    def pose_hand2ego(self, side):
        """ returns (N, 4, 4) """
        T_hand2world = self.pose_hand2world(side=side)
        T_w2e = self.world2ego
        T_h2e = torch.bmm(T_w2e, T_hand2world)
        return T_h2e
    
    def pose_hand2world(self, side):
        """
        Returns: (N, 4, 4)
        """
        pose_h = torch.from_numpy(self.mano_params[side]['pose'])
        shape_h = torch.from_numpy(self.mano_params[side]['shape']).view(1, 10).tile(self.num_frames, 1)
        rot_h= torch.from_numpy(self.mano_params[side]['rot'])
        trans_h= torch.from_numpy(self.mano_params[side]['trans'])
        _, _, T_world = ManoLayerTracer(
            flat_hand_mean=False, ncomps=45, side=side, use_pca=False,
            mano_root='./externals/mano/').forward_transform(
            torch.cat([rot_h, pose_h], axis=-1), 
            th_betas=shape_h, th_trans=trans_h, root_palm=True)
        return T_world

    def pose_obj2world(self):
        T_o_world = _get_transform(self.obj_rot, self.obj_trans/1000)
        return T_o_world
    
    def pose_obj2ego(self):
        T_o_world = self.pose_obj2world()
        T_w2e = self.world2ego
        T_o2e = torch.bmm(T_w2e, T_o_world)
        return T_o2e

    def pose_obj2hand(self):
        """
        Returns:
            T_obj2left, T_obj2right: (num_frames, 4, 4)
        """
        T_o_world = self.pose_obj2world()
        T_l_world = self.pose_hand2world(side='left')
        T_r_world = self.pose_hand2world(side='right')
        T_o2l = torch.bmm(torch.inverse(T_l_world), T_o_world)
        T_o2r = torch.bmm(torch.inverse(T_r_world), T_o_world)
        return T_o2l, T_o2r
    
    def render_mesh(self, frame_idx) -> np.ndarray:
        """ 
        Args:
            vo_orig: object vertices in original space. If None, use self.vo_orig
        """
        vl = self.hand_verts(frame_idx, 'left', space='ego', as_mesh=False)
        vr = self.hand_verts(frame_idx, 'right', space='ego', as_mesh=False)
        left = SimpleMesh(verts=vl, faces=self.fl, tex_color=(0, 1.0, 0))
        right = SimpleMesh(verts=vr, faces=self.fr, tex_color=(0, 0, 1.0))
        K_ego = self.K_ego[frame_idx, ...]
        img_w, img_h = CROPPED_IMAGE_SIZE
        w_ratio = img_w / orig_reso[0]
        h_ratio = img_h / orig_reso[1]
        fx = K_ego[0, 0] * h_ratio  # This is in-fact confusing: why not w_ratio?
        fy = K_ego[1, 1] * w_ratio
        cx = K_ego[0, 2] * h_ratio
        cy = K_ego[1, 2] * w_ratio
        cam_manager = CameraManager(fx, fy, cx, cy, img_h=img_h, img_w=img_w)

        proj_method=dict(name='pytorch3d', coor_sys='nr', in_ndc=False)
        obj_cam = self.obj_verts(frame_idx, space='ego', as_mesh=True)
        rend = projection.perspective_projection_by_camera(
            [left, right, obj_cam],
            camera=cam_manager,
            method=proj_method,
            image=self.render_image(frame_idx))
        return rend
    
    def to_scene(self, frame_idx, space='ego', side=None, 
                 show_axis=True,
                 viewpoint='nr'):
        vl = self.hand_verts(frame_idx, 'left', space=space, as_mesh=False)
        vr = self.hand_verts(frame_idx, 'right', space=space, as_mesh=False)

        left = SimpleMesh(verts=vl, faces=self.fl, tex_color=(0, 1.0, 0))
        right = SimpleMesh(verts=vr, faces=self.fr, tex_color=(0, 0, 1.0))
        obj = self.obj_verts(frame_idx, space=space, as_mesh=True)

        if side is None:
            meshes = [obj, left, right]
        else:
            meshes = [obj, left if side == LEFT else right]
        return visualize_mesh(meshes, show_axis=show_axis, viewpoint=viewpoint)

    def render_triview(self, frame_idx, side: str, coor_sys='nr',
                       rend_size=300) -> np.ndarray:
        """
        Returns:
            (H, W, 3)
        """
        dmax = 0.30
        mobj = self.obj_verts(frame_idx, space=side, as_mesh=True)
        mhand = self.hand_verts(frame_idx, side=side, space=side, as_mesh=True)
        front = projection.project_standardized(
            [mhand, mobj],
            direction='+z',
            image_size=rend_size,
            method=dict(
                name='pytorch3d',
                coor_sys=coor_sys,
                in_ndc=False
            ),
            centering=False,
            manual_dmax=dmax,
            show_axis=True)
        left = projection.project_standardized(
            [mhand, mobj],
            direction='+x',
            image_size=rend_size,
            method=dict(
                name='pytorch3d',
                coor_sys=coor_sys,
                in_ndc=False
            ),
            centering=False,
            manual_dmax=dmax,
            show_axis=True)
        back = projection.project_standardized(
            [mhand, mobj],
            direction='-y',
            image_size=rend_size,
            method=dict(
                name='pytorch3d',
                coor_sys=coor_sys,
                in_ndc=False
            ),
            centering=False,
            manual_dmax=dmax,
            show_axis=True)
        return np.hstack([front, left, back])
