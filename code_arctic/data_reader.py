from hydra.utils import to_absolute_path
import os.path as osp
import numpy as np
import torch
from pytorch3d.transforms import rotation_conversions as rotcvt
from pytorch3d.transforms.rotation_conversions import axis_angle_to_quaternion, quaternion_apply
from PIL import Image
import cv2
import trimesh
# from nnutils.handmocap import get_hand_faces
from manopth.manolayer import ManoLayer
from homan.manolayer_tracer import ManoLayerTracer

from libzhifan.geometry import SimpleMesh
from libzhifan.geometry import CameraManager, projection, visualize_mesh
from libzhifan import io
from libzhifan import odlib
odlib.setup('xywh', norm=False)


""" This file contains SeqReader which is slow,
but don't delete me as many variables will be references."""


orig_reso = (2800, 2000)
CROPPED_IMAGE_SIZE = (840, 600)
ARCTIC_IMAGES_DIR = './DATA_STORAGE/arctic_data/cropped_images'

MASK_LEFT_ID = 1
MASK_RIGHT_ID = 2
MASK_OBJ_ID = 3

LEFT = 'left'
RIGHT = 'right'


class SeqReader:
    def __init__(self, 
                 proc_data: dict, 
                 sid: str,
                 seq_name: str,
                 obj_name: str, 
                 view_id=0):
        """ Stores all num_frames 

        hand's rot_r is in m space

        Args:
            proc_data: path to processed data

            obj_pose: (num_frames, 7)
            view_id: camera id, 0 is ego camera
        """
        self.images_dir = to_absolute_path(osp.join(ARCTIC_IMAGES_DIR, sid, seq_name))
        self.proc_data = proc_data
        self.sid = sid
        self.seq_name = seq_name
        self.view_id = view_id

        self.ioi_offset = io.read_json(
            to_absolute_path('arctic_data/meta/misc.json')
            )[sid]['ioi_offset']

        """ left / right hand"""
        self.l_mano_layer = ManoLayer(
            flat_hand_mean=False, ncomps=45, side='left', use_pca=False,
            mano_root=to_absolute_path('./externals/mano/'))
        self.r_mano_layer = ManoLayer(
            flat_hand_mean=False, ncomps=45, side='right', use_pca=False,
            mano_root=to_absolute_path('./externals/mano/'))
        self.fl = self.l_mano_layer.th_faces.cpu().numpy()
        self.fr = self.r_mano_layer.th_faces.cpu().numpy()

        mesh = trimesh.load(
            to_absolute_path(f'arctic_data/meta/object_vtemplates/{obj_name}/mesh.obj'))
        self.vo_orig = torch.from_numpy(mesh.vertices).float() / 1000
        parts_ids = io.read_json(
            to_absolute_path(f'arctic_data/meta/object_vtemplates/{obj_name}/parts.json'))
        PART_TOP_ID = 0
        self.vo_top_idx = np.asarray(parts_ids) == PART_TOP_ID

        self.fo = mesh.faces

        self.K_ego = proc_data['params']['K_ego']
        self.num_frames = len(self.K_ego)

    def render_image(self, frame_idx) -> np.ndarray:
        frame_idx = frame_idx + self.ioi_offset
        img_path = osp.join(self.images_dir, str(self.view_id), f'{frame_idx:05d}.jpg')
        return np.asarray(Image.open(img_path))
    
    def render_image_mesh(self, frame_idx, 
                          with_rend=True,
                          with_triview=False,
                          side=None,
                          use_preprocessed_verts=False):
        img = self.render_image(frame_idx)/255
        if with_rend:
            rend = self.render_mesh(frame_idx, use_preprocessed_verts=use_preprocessed_verts)
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

    def render_hand_object_mask(self, frame_idx):
        """ 
        Returns:
            sil: (H, W), dtype=np.uint8
                0 for bg, 1 left, 2 right, 3 object
        """
        vl = self.hand_verts(frame_idx, 'left', space='ego', as_mesh=False)
        vr = self.hand_verts(frame_idx, 'right', space='ego', as_mesh=False)
        vo_cam = self.obj_verts(frame_idx, space='ego', as_mesh=False)
        left = SimpleMesh(verts=vl, faces=self.fl, tex_color=(0, 1.0, 0))
        right = SimpleMesh(verts=vr, faces=self.fr, tex_color=(0, 0, 1.0))
        obj_cam = SimpleMesh(verts=vo_cam, faces=self.fo, tex_color=(1.0, 0, 0))
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
        obj_cam = SimpleMesh(verts=vo_cam, faces=self.fo, tex_color=(1.0, 0, 0))
        # left-1, right-2, obj-3
        rend = projection.perspective_projection_by_camera(
            [left, right, obj_cam],
            camera=cam_manager,
            method=proj_method)  

        rend = rend.astype(np.uint8)
        return rend

    def get_boxes_and_mask(self, frame_idx):
        """
        Returns:
            lbox, rbox, obox: [x, y, w, h] in pixel
                can be None
            mask: (H, W), dtype=np.uint8
                0 for bg, 1 left, 2 right, 3 object
        """
        mask = self.render_hand_object_mask(frame_idx=frame_idx)
        obox = bbox_from_mask(mask, MASK_OBJ_ID)
        lbox = bbox_from_mask(mask, MASK_LEFT_ID)
        rbox = bbox_from_mask(mask, MASK_RIGHT_ID)
        return lbox, rbox, obox, mask

    def get_boxes(self, frame_idx):
        """
        Returns:
            lbox, rbox, obox: [x, y, w, h] in pixel
                can be None
        """
        mask = self.render_hand_object_mask(frame_idx=frame_idx)
        obox = bbox_from_mask(mask, MASK_OBJ_ID)
        lbox = bbox_from_mask(mask, MASK_LEFT_ID)
        rbox = bbox_from_mask(mask, MASK_RIGHT_ID)
        return lbox, rbox, obox
    
    def get_camera(self, frame_idx):
        """
        Returns:

        """
        raise NotImplementedError
    
    def hand_tips(self, side, frame_idx=None, space='ego'):
        """
        Returns:
            (N, 21, 3) hand tips in meter space
        """
        def forward_hand_verts_myimpl(side):
            if side == 'left':
                suffix = 'l'
                mano_layer = self.l_mano_layer
            elif side == 'right':
                suffix = 'r'
                mano_layer = self.r_mano_layer
            else:
                raise ValueError(f'Unknown side {side}')

            pose_h = torch.from_numpy(self.proc_data['params']['pose_' + suffix])
            shape_h = torch.from_numpy(self.proc_data['params']['shape_' + suffix])
            rot_h = torch.from_numpy(self.proc_data['params']['rot_' + suffix])
            trans_h = torch.from_numpy(self.proc_data['params']['trans_' + suffix])
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

        if space == 'ego':
            params = self.proc_data['params']
            T_w2e = torch.from_numpy(params['world2ego'])
            j = pose_apply(T_w2e, j)
        elif space == 'world':
            pass
        elif space == 'left' or space == 'right':
            T_h_world = self.pose_hand2world(side=space)
            j = pose_apply(torch.inverse(T_h_world), j)
        elif space == 'obj':
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown space {space}')

        if frame_idx is not None:
            j = j[frame_idx]
        return j

    def hand_verts(self, frame_idx, side, space='ego', as_mesh=False, use_myimpl=False,
                   tex_color='light_blue'):
        """
        Args:
            space: 'world' or 'ego'
        
        Returns:
            (N, 778, 3) if frame_idx is None, otherwise (778, 3)
        """
        def forward_hand_verts(side):
            if side == 'left':
                suffix = 'l'
                mano_layer = self.l_mano_layer
            elif side == 'right':
                suffix = 'r'
                mano_layer = self.r_mano_layer

            pose_h = torch.from_numpy(self.proc_data['params']['pose_' + suffix])
            shape_h = torch.from_numpy(self.proc_data['params']['shape_' + suffix])
            rot_h = torch.from_numpy(self.proc_data['params']['rot_' + suffix])
            trans_h = torch.from_numpy(self.proc_data['params']['trans_' + suffix])  # in meter space
            th_pose_coeffs = torch.cat([
                rot_h,
                pose_h], axis=-1)
            v, _ = mano_layer.forward(
                th_pose_coeffs, th_betas=shape_h, th_trans=trans_h)
            v /= 1000
            return v, mano_layer.th_faces

        def forward_hand_verts_myimpl(side):
            if side == 'left':
                suffix = 'l'
                mano_layer = self.l_mano_layer
            elif side == 'right':
                suffix = 'r'
                mano_layer = self.r_mano_layer

            pose_h = torch.from_numpy(self.proc_data['params']['pose_' + suffix])
            shape_h = torch.from_numpy(self.proc_data['params']['shape_' + suffix])
            rot_h = torch.from_numpy(self.proc_data['params']['rot_' + suffix])
            trans_h = torch.from_numpy(self.proc_data['params']['trans_' + suffix])
            rot_zeros = torch.zeros_like(rot_h)
            trans_zeros = torch.zeros_like(trans_h)
            v, _ = mano_layer.forward(
                torch.cat([rot_zeros, pose_h], axis=-1), 
                th_betas=shape_h, th_trans=trans_zeros, root_palm=True)
            v /= 1000  # in meter

            _, _, T_world = ManoLayerTracer(
                flat_hand_mean=False, ncomps=45, side=side, use_pca=False,
                mano_root='./externals/mano/').forward_transform(
                torch.cat([rot_h, pose_h], axis=-1), 
                th_betas=shape_h, th_trans=trans_h, root_palm=True)
            v = pose_apply(T_world, v)
            return v, mano_layer.th_faces

        # world space
        if not use_myimpl:
            v, f = forward_hand_verts(side)
        else:
            v, f = forward_hand_verts_myimpl(side)

        if space == 'ego':
            params = self.proc_data['params']
            T_w2e = torch.from_numpy(params['world2ego'])
            v = pose_apply(T_w2e, v)
        elif space == 'world':
            pass
        elif space == 'left' or space == 'right':
            T_h_world = self.pose_hand2world(side=space)
            v = pose_apply(torch.inverse(T_h_world), v)
        elif space == 'obj':
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown space {space}')

        if as_mesh:
            return SimpleMesh(v[frame_idx], f, tex_color=tex_color)
        else:
            return v if frame_idx is None else v[frame_idx] 

    def obj_verts(self, frame_idx, space='ego', as_mesh=False,
                  tex_color='red'):
        """
        This doesn't handle the articulation

        obj_trans is in mm

        Returns: (N, V, 3) if frame_idx is None else (V, 3)
        """
        def forward_obj_arti(vo, top_idx, arti: torch.Tensor):
            """
            vo: (N, V, 3)
            top_idx: (V,)
            arti: (N, 1)
            """
            vo = torch.as_tensor(vo).clone()  # (N, V, 3)
            Z_AXIS = torch.tensor([0, 0, -1], dtype=torch.float).view(1, 3)
            angles = arti.view(-1, 1)
            quat_arti = axis_angle_to_quaternion(Z_AXIS * angles)  # [N, 4]
            v_top_articulated = quaternion_apply(quat_arti[:, None, :], vo[:, top_idx, ...])  # (N, VT, 3)
            vo[:, top_idx, ...] = v_top_articulated
            return vo

        N = self.num_frames
        arti = torch.from_numpy(self.proc_data['params']['obj_arti'])
        T_o_world = _get_transform(self.proc_data['params']['obj_rot'], 
                                   self.proc_data['params']['obj_trans']/1000)

        vo = self.vo_orig.view(1, -1, 3).repeat(N, 1, 1)
        vo = forward_obj_arti(vo, self.vo_top_idx, arti)  # (N, V, 3)
        vo = pose_apply(T_o_world, vo)
        
        if space == 'ego':
            T_w2e = torch.from_numpy(self.proc_data['params']['world2ego'])
            vo = pose_apply(T_w2e, vo)
        elif space == 'world':
            pass
        elif space == 'left' or space == 'right':
            T_h_world = self.pose_hand2world(side=space)
            vo = pose_apply(torch.inverse(T_h_world), vo)
        elif space == 'obj':
            vo = pose_apply(torch.inverse(T_o_world), vo)
        else:
            raise ValueError(f'Unknown space {space}')
        
        if frame_idx is None:
            assert as_mesh is False
        else:
            vo = vo[frame_idx]

        if as_mesh:
            return SimpleMesh(verts=vo, faces=self.fo, tex_color=tex_color)
        else:
            return vo

    def obj_verts_in_hand(self, frame_idx, T_o2h, as_mesh=False,
                          tex_color='red'):
        """
        This doesn't handle the articulation

        obj_trans is in mm

        Returns: (N, V, 3) if frame_idx is None else (V, 3)
        """
        def forward_obj_arti(vo, top_idx, arti: torch.Tensor):
            """
            vo: (N, V, 3)
            top_idx: (V,)
            arti: (N, 1)
            """
            vo = torch.as_tensor(vo).clone()  # (N, V, 3)
            Z_AXIS = torch.tensor([0, 0, -1], dtype=torch.float).view(1, 3)
            angles = arti.view(-1, 1)
            quat_arti = axis_angle_to_quaternion(Z_AXIS * angles)  # [N, 4]
            v_top_articulated = quaternion_apply(quat_arti[:, None, :], vo[:, top_idx, ...])  # (N, VT, 3)
            vo[:, top_idx, ...] = v_top_articulated
            return vo

        N = len(T_o2h)
        # arti = torch.from_numpy(self.proc_data['params']['obj_arti'])
        vo = self.vo_orig.view(1, -1, 3).repeat(N, 1, 1)
        # vo = forward_obj_arti(vo, self.vo_top_idx, arti)  # (N, V, 3)
        vo = pose_apply(T_o2h, vo)
        
        if frame_idx is None:
            assert as_mesh is False
        else:
            vo = vo[frame_idx]

        if as_mesh:
            return SimpleMesh(verts=vo, faces=self.fo, tex_color=tex_color)
        else:
            return vo
    
    def pose_hand2world(self, side):
        """
        Returns: (N, 4, 4)
        """
        if side == 'left':
            suffix = 'l'
        elif side == 'right':
            suffix = 'r'

        pose_h = torch.from_numpy(self.proc_data['params']['pose_' + suffix])
        shape_h = torch.from_numpy(self.proc_data['params']['shape_' + suffix])
        rot_h = torch.from_numpy(self.proc_data['params']['rot_' + suffix])
        trans_h = torch.from_numpy(self.proc_data['params']['trans_' + suffix])
        _, _, T_world = ManoLayerTracer(
            flat_hand_mean=False, ncomps=45, side=side, use_pca=False,
            mano_root='./externals/mano/').forward_transform(
            torch.cat([rot_h, pose_h], axis=-1), 
            th_betas=shape_h, th_trans=trans_h, root_palm=True)
        return T_world

    def pose_obj2world(self):
        params = self.proc_data['params']
        T_o_world = _get_transform(params['obj_rot'], params['obj_trans']/1000)
        return T_o_world

    def pose_obj2hand(self):
        """
        Warning: it is impossible to get hand-to-world transform by performing
        global_orientation tranform ourselfves; everything has to be done in 
        the intricate mano_layer

        however, we 
        In arctic,
            hand undergoes: [hand <-> world, implicit in mano_layer] -> ego
            object undergoes: object -> world -> ego,
        hence, to find T_obj2hand, we need to find T_hand2world and T_obj2world
        they can be read from self.proc_data['params']

        Returns:
            T_obj2left, T_obj2right: (num_frames, 4, 4)
        """
        T_o_world = self.pose_obj2world()
        T_l_world = self.pose_hand2world(side='left')
        T_r_world = self.pose_hand2world(side='right')
        T_o2l = torch.bmm(torch.inverse(T_l_world), T_o_world)
        T_o2r = torch.bmm(torch.inverse(T_r_world), T_o_world)
        return T_o2l, T_o2r
    
    def use_preprocessed_verts(self, frame_idx):
        """
        verts as output by running e.g.
            python scripts_data/process_seqs.py \
                --mano_p ./data/arctic_data/data/raw_seqs/s01/ketchup_grab_01.mano.npy \
                --export_verts
        """
        raise ValueError("Need to run arctic code to exporting verts")
        vl = self.proc_data['cam_coord']['verts.left'][frame_idx, self.view_id, ...]  # n_frames, views, 778, 3
        vr = self.proc_data['cam_coord']['verts.right'][frame_idx, self.view_id, ...]
        vo_cam = self.proc_data['cam_coord']['verts.object'][frame_idx, self.view_id, ...]
        return vl, vr, vo_cam
    
    def render_mesh(self, frame_idx, use_preprocessed_verts) -> np.ndarray:
        """ 
        Args:
            vo_orig: object vertices in original space. If None, use self.vo_orig
        """
        if use_preprocessed_verts:
            vl, vr, vo_cam = self.use_preprocessed_verts(frame_idx)
        else:
            vl = self.hand_verts(frame_idx, 'left', space='ego', as_mesh=False)
            vr = self.hand_verts(frame_idx, 'right', space='ego', as_mesh=False)
            vo_cam = self.obj_verts(frame_idx, space='ego', as_mesh=False)
        left = SimpleMesh(verts=vl, faces=self.fl, tex_color=(0, 1.0, 0))
        right = SimpleMesh(verts=vr, faces=self.fr, tex_color=(0, 0, 1.0))
        obj_cam = SimpleMesh(verts=vo_cam, faces=self.fo, tex_color=(1.0, 0, 0))
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
        obj_cam = SimpleMesh(verts=vo_cam, faces=self.fo, tex_color=(1.0, 0, 0))
        rend = projection.perspective_projection_by_camera(
            [left, right, obj_cam],
            camera=cam_manager,
            method=proj_method,
            image=self.render_image(frame_idx))
        return rend
    
    def to_scene(self, frame_idx, space='ego', side=None, 
                 use_preprocessed_verts=False,
                 show_axis=True,
                 viewpoint='nr'):
        if use_preprocessed_verts:
            assert space == 'ego'
            vl, vr, vo = self.use_preprocessed_verts(frame_idx)
        else:
            vl = self.hand_verts(frame_idx, 'left', space=space, as_mesh=False)
            vr = self.hand_verts(frame_idx, 'right', space=space, as_mesh=False)
            vo = self.obj_verts(frame_idx, space=space, as_mesh=False)

        left = SimpleMesh(verts=vl, faces=self.fl, tex_color=(0, 1.0, 0))
        right = SimpleMesh(verts=vr, faces=self.fr, tex_color=(0, 0, 1.0))
        obj = SimpleMesh(verts=vo, faces=self.fo, tex_color=(1.0, 0, 0))

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
        mhand =self.hand_verts(frame_idx, side=side, space=side, as_mesh=True)
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


def rot_transl_apply(rot: torch.Tensor, transl: torch.Tensor, verts: torch.Tensor):
    """
    rot: (N, 3, 3)
    transl: (N, 1, 3)
    verts: (N, V, 3)
    => (N, V, 3)
    """
    return (torch.bmm(rot, verts.permute(0, 2, 1)) + transl).permute(0, 2, 1) 


def pose_apply(pose: torch.Tensor, verts: torch.Tensor):
    """
    pose: (N, 4, 4)
    verts: (N, V, 3)
    => (N, V, 3)
    """
    rot = pose[:, :3, :3]
    transl = pose[:, :3, 3:]
    return (torch.bmm(rot, verts.permute(0, 2, 1)) + transl).permute(0, 2, 1) 


def _get_transform(rot, trans):
    """ This convert axis-angle rotation vector and translation 
    to pose matrix (4x4)

    Args:
        rot: (N, 3)
        transl: (N, 3)
    
    Returns: (N, 4, 4)) 
    """
    def tensorize(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x
    rot_world = tensorize(rot)
    rot_world = rotcvt.axis_angle_to_matrix(rot_world)  # apply-to-col
    trans_world = tensorize(trans)
    T_world = torch.eye(4).repeat(rot_world.shape[0], 1, 1)
    T_world[:, :3, :3] = rot_world
    T_world[:, :3, -1] = trans_world
    return T_world


def bbox_from_mask(mask, mask_id=1):
    """ 2d bbox. ret: [x0, y0, w, h] """
    _is, _js = np.where(mask == mask_id)
    if _is.size == 0:
        return None
    x0 = _js.min()
    w = _js.max() - x0
    y0 = _is.min()
    h = _is.max() - y0
    return np.asarray([x0, y0, w, h])