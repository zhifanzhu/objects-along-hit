import trimesh
import torch
import cv2
import os
import os.path as osp
import pandas as pd
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from pytorch3d.transforms import matrix_to_axis_angle
from homan.manolayer_tracer import ManoLayerTracer, ManoLayer, pose_apply
from libzhifan import io, odlib
from libzhifan.geometry import SimpleMesh, CameraManager, projection, BatchCameraManager
from code_hot3d.data_loaders.AriaDataProvider import AriaDataProvider
from code_hot3d.data_loaders.QuestDataProvider import QuestDataProvider
from code_hot3d.dataset_api import Hot3dDataProvider
from moviepy import editor

from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.stream_id import StreamId  # @manual
odlib.setup('xywh')


PALETTE_IMG = './weights/hot3d/PALETTE.png'
OBJ_DIAMETERS_PATH = './weights/hot3d/diameters.json'
# HOT3DDATA_ROOT = '/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/'
HOT3DDATA_ROOT = './DATA_STORAGE/hot3d_dataset/'
MANO_HAND_MODEL_PATH = './externals/mano/'
REDUCED_MESHES_ROOT = './weights/obj_models/hot3d_export/'
MASK_FMT = './DATA_STORAGE/hot3d_dataset/masks/{seq_name}/{frame_idx:06d}.png'



LEFT = 'left'
RIGHT = 'right'

class HOT3DReader(Hot3dDataProvider, AriaDataProvider, QuestDataProvider):

    def __init__(self,
                 seq_name: str,
                 obj_version: str,
                 data_root=HOT3DDATA_ROOT):
        self.seq_name = seq_name
        self.data_root = data_root
        self.seq_dir = osp.join(self.data_root, self.seq_name)

        sequence_folder = osp.join(self.data_root, self.seq_name)
        object_library = osp.join(self.data_root, 'assets')
        mano_hand_model_path = MANO_HAND_MODEL_PATH
        mano_hand_model = MANOHandModel(mano_hand_model_path)
        Hot3dDataProvider.__init__(self,
            sequence_folder, object_library=object_library,
            mano_hand_model=mano_hand_model, fail_on_missing_data=True)
        self.fl = self._mano_hand_data_provider.mano_layer.mano_layer_left.faces
        self.fr = self._mano_hand_data_provider.mano_layer.mano_layer_right.faces
        self._prepare_mano_layers()

        assert obj_version in {'original', 'reduced'}
        self.obj_version = obj_version
        self._palette = Image.open(PALETTE_IMG).getpalette()
        self._maskid2color = {
            i: self._palette[3*i:3*i+3]
            for i in range(len(self._palette)//3)}
        self.obj_diameters = io.read_json(OBJ_DIAMETERS_PATH)['diameters']

        # TODO: Check if we are able to use either quest or Aria just by initialising them in these conditions
        vrs_filepath = osp.join(self.data_root, self.seq_name, 'recording.vrs')
        if StreamId('214-1') not in self.device_data_provider.get_image_stream_ids():
            self.quest_sequence = True
            QuestDataProvider.__init__(self, vrs_filepath, osp.join(self.data_root, self.seq_name, 'camera_models.json'))
            self.img_w, self.img_h = 1280, 1024
        else:
            self.quest_sequence = False
            mp4_folder_path = osp.join(self.data_root, self.seq_name, 'mps')
            AriaDataProvider.__init__(self, vrs_filepath, mp4_folder_path)
            self.img_w, self.img_h = 1408, 1408

        self._provider_post_init()
        self._prepare_object_models()
        self._prepare_obj_bboxes()
        self._prepare_handbox_data()
        # self._prepare_hand3d_data()
        self._prepare_camera_data()

    @property
    def skip_reduced_cats(self):
        """ These categories are skipped as their mesh can't be reduced in blender. """
        return {'aria_small', 'vase', 'dvd_remote', 'flask', 'spoon_wooden', 'dumbbell_5lb'}
    
    @property
    def mano_beta(self):
        """ (10,) shared for both hands """
        return self._mano_hand_data_provider._mano_shape_params.float()

    def _prepare_mano_layers(self):
        """
        Note that ARIA's left mano has a special fix 
        https://github.com/facebookresearch/hot3d/blob/09b62e6f0eae911f245101d695c7c4cb80bee820/hot3d/data_loaders/mano_layer.py#L99-L109
        """
        """ HOMan's MANOLayer """
        flat_hand_mean = False
        self.r_mano = ManoLayer(
            use_pca=True, ncomps=15, mano_root='./externals/mano/', 
            side=RIGHT, flat_hand_mean=flat_hand_mean)
        self.r_mano_tracer = ManoLayerTracer(
            use_pca=True, ncomps=15, mano_root='./externals/mano/', 
            side=RIGHT, flat_hand_mean=flat_hand_mean)
        self.l_mano = ManoLayer(
            use_pca=True, ncomps=15, mano_root='./externals/mano/', 
            side=LEFT, flat_hand_mean=flat_hand_mean)
        self.l_mano_tracer = ManoLayerTracer(
            use_pca=True, ncomps=15, mano_root='./externals/mano/', 
            side=LEFT, flat_hand_mean=flat_hand_mean)
        
        # Follow ARIA's fix for left mano
        #         https://github.com/facebookresearch/hot3d/blob/09b62e6f0eae911f245101d695c7c4cb80bee820/hot3d/data_loaders/mano_layer.py#L99-L109
        if (
            torch.sum(
                torch.abs(
                    self.l_mano.th_shapedirs[:, 0, :]
                    - self.r_mano.th_shapedirs[:, 0, :]
                )
            )
            < 1
        ):
            self.l_mano.th_shapedirs[:, 0, :] *= -1
            self.l_mano_tracer.th_shapedirs[:, 0, :] *= -1

    def _provider_post_init(self):
        if self.quest_sequence:
            self._stream_id = StreamId("1201-1")
            ts_list = self.device_data_provider.get_sequence_timestamps()
        else:
            self._stream_id = StreamId("214-1")
            ts_list = self._vrs_data_provider.get_timestamps_ns(self._stream_id, TimeDomain.TIME_CODE)
        self._ts2frame = None  # 0 index
        self._frame2ts = None
        self._frame2ts = ts_list
        self._ts2frame = {ts: frame for frame, ts in enumerate(ts_list)}
        self.num_frames = len(ts_list)

    def _prepare_object_models(self):
        instance_mapping = io.read_json(osp.join(self.data_root, 'assets/instance.json'))
        self.box2d_objects_df = pd.read_csv(osp.join(self.seq_dir, 'box2d_objects.csv'))
        uids = set(self.box2d_objects_df['object_uid'].unique())
        self.cat_to_uid = {}
        self.obj_mesh_dict = {}  # key: cat,  value: mesh
        for obj_uid in uids:
            cat = instance_mapping[str(obj_uid)]['instance_name']
            if self.obj_version == 'original':
                mesh = trimesh.load(osp.join(self.data_root, 'assets', f'{obj_uid}.glb'), force='mesh')
            elif self.obj_version == 'reduced':
                if cat in self.skip_reduced_cats:
                    continue
                mesh = trimesh.load(osp.join(REDUCED_MESHES_ROOT, f'{cat}.obj'))
            self.obj_mesh_dict[cat] = mesh
            self.cat_to_uid[cat] = str(obj_uid)
        self._T_o2w_dict = self.precompute_T_o2w_dict()

    def _prepare_handbox_data(self):
        lboxes, rboxes = [], []

        LEFTID = 0
        RIGHTID = 1
        for frame, ts in enumerate(self._frame2ts):
            box_data = self.hand_box2d_data_provider.get_bbox_at_timestamp(
                self._stream_id, ts,
                TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE)
            box2ds = box_data.box2d_collection.box2ds
            # for i in [LEFTID, RIGHTID]:
            for i in box2ds.keys():
                box2d = box2ds[i].box2d
                # the visibility is for occlusion, not for out-of-view
                # visibility_ratio = box2ds[i].visibility_ratio
                if box2d is None:
                    box = [-1, -1, -1, -1] # , visibility_ratio]
                else:
                    box = [box2d.left, box2d.top, box2d.width, box2d.height] # , visibility_ratio]
                if i == LEFTID:
                    lboxes.append(box)
                elif i == RIGHTID:
                    rboxes.append(box)
                else:
                    raise ValueError(f'Unknown hand: {i}')
        self._lboxes_orig = np.asarray(lboxes)
        self._rboxes_orig = np.asarray(rboxes)

    # def _prepare_hand3d_data(self):
    #     # T_l2ws, T_r2ws = [], []
    #     poses_l, poses_r = [], []

    #     provider = self._mano_hand_data_provider
    #     for frame, ts in enumerate(self._frame2ts):
    #         hand_poses_with_dt = provider.get_pose_at_timestamp(
    #                 timestamp_ns=ts,
    #                 time_query_options=TimeQueryOptions.CLOSEST,
    #                 time_domain=TimeDomain.TIME_CODE)
    #         hand_pose_collection = hand_poses_with_dt.pose3d_collection
    #         for hand_pose_data in hand_pose_collection.poses.values():
    #             if hand_pose_data.is_left_hand():
    #                 poses_l.append(np.asarray(hand_pose_data.joint_angles))
    #                 # T_l2w = hand_pose_data.wrist_pose.to_matrix()
    #                 # T_w2l = np.linalg.inv(T_l2w)
    #                 # T_l2ws.append(T_l2w)
    #             if hand_pose_data.is_right_hand():
    #                 poses_r.append(np.asarray(hand_pose_data.joint_angles))
    #                 # T_r2w = hand_pose_data.wrist_pose.to_matrix()
    #                 # T_w2r = np.linalg.inv(T_r2w)
    #                 # T_r2ws.append(T_r2w)

    #     # self.T_l2w = rearrange(T_l2ws, 'f m n -> f m n')
    #     # self.T_r2w = rearrange(T_r2ws, 'f m n -> f m n')
    #     self.poses_l = rearrange(poses_l, 'f n -> f n')
    #     self.poses_r = rearrange(poses_r, 'f n -> f n')

    def _prepare_camera_data(self):
        [raw_T_device_camera, raw_intrinsics] = \
            self.device_data_provider.get_camera_calibration(self._stream_id)
        pinhole_cw90_intrinsics = calibration.rotate_camera_calib_cw90deg(raw_intrinsics)
        T_device_camera = pinhole_cw90_intrinsics.get_transform_device_camera()

        T_c2w = []
        fx, fy, cx, cy = [], [], [], []
        for f, ts in enumerate(self._frame2ts):
            T_world_device = self.device_pose_data_provider.get_pose_at_timestamp(
                ts, time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE).pose3d.T_world_device # .to_matrix()
            c2w = (T_world_device @ T_device_camera).to_matrix()
            _fx, _fy = pinhole_cw90_intrinsics.get_focal_lengths()
            _cx, _cy = pinhole_cw90_intrinsics.get_principal_point()
            T_c2w.append(c2w)
            fx.append(_fx)
            fy.append(_fy)
            cx.append(_cx)
            cy.append(_cy)

        self.T_c2w = torch.from_numpy(
            rearrange(T_c2w, 'f m n -> f m n')).float()  # (N, 4, 4)
        self.T_w2c = self.T_c2w.inverse()
        fx, fy, cx, cy = map(lambda x : torch.from_numpy(np.asarray(x)), (fx, fy, cx, cy))
        img_h, img_w = pinhole_cw90_intrinsics.get_image_size()
        img_h = repeat(torch.tensor(img_h), '-> b', b=len(fx))
        img_w = repeat(torch.tensor(img_w), '-> b', b=len(fx))
        self._cam_params = (fx, fy, cx, cy, img_h, img_w)
        self.batch_camera_manager = BatchCameraManager(
            fx, fy, cx, cy, img_h=img_w, img_w=img_w,
            in_ndc=False, device='cpu')
        self.K_ego = self.batch_camera_manager.get_K()

    def set_current_object(self, cat: str):
        self.cat = cat
        self.obj_uid = self.cat_to_uid[cat]
        self.obj_mesh_selected = self.obj_mesh_dict[cat]
        self.vo_orig = torch.from_numpy(self.obj_mesh_selected.vertices).float()
        self.fo = torch.from_numpy(self.obj_mesh_selected.faces).int()
        # Convert to frame based ndarray
        T_o2w = []
        for frame, ts in enumerate(self._frame2ts):
            _T_o2w_dict = self._T_o2w_dict[str(self.obj_uid)]
            if ts not in _T_o2w_dict:
                T_o2w.append(np.ones((4, 4)) * -1)
            else:
                T_o2w.append(self._T_o2w_dict[str(self.obj_uid)][ts])
        self.T_o2w = torch.from_numpy(rearrange(T_o2w, 'f m n -> f m n')).float()  # (N, 4, 4)
        """ T_o2c: (N, 4, 4) """
        T_o2w = self.T_o2w
        T_c2w = self.T_c2w
        self.T_o2c = T_c2w.inverse() @ T_o2w

    def _prepare_obj_bboxes(self):
        # extract boxes, (N, 4) of (x0, y0, w, h)
        # also extract visibility (occlusion)
        self._obj_boxes_orig = dict()
        self._obj_visible = dict()  # UNUSED
        for cat, obj_uid in self.cat_to_uid.items():
            boxes = []
            visibility = []
            for frame, ts in enumerate(self._frame2ts):
                box_data = self.object_box2d_data_provider.get_box2d_at_timestamp(
                    self._stream_id, ts,
                    TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE)
                box2ds = box_data.box2d_collection.box2ds
                if obj_uid not in box2ds:
                    box = [-1, -1, -1, -1]
                    vis = 0
                else:
                    box2d = box2ds[obj_uid].box2d
                    vis = box2ds[obj_uid].visibility_ratio
                    # The visibility is for occlusion, not for out-of-view
                    # visibility_ratio = box2ds[self.obj_uid].visibility_ratio
                    if box2d is None:
                        box = [-1, -1, -1, -1]
                    else:
                        box = [box2d.left, box2d.top, box2d.width, box2d.height]
                boxes.append(box)
                visibility.append(vis)

            self._obj_boxes_orig[cat] = np.asarray(boxes)
            self._obj_visible[cat] = np.asarray(visibility)

    def precompute_T_o2w_dict(self):
        time_query_options=TimeQueryOptions.CLOSEST
        time_domain=TimeDomain.TIME_CODE
        T_o2w_dict = {}

        for frame, ts_ns in enumerate(self._frame2ts):
            pose_info = self.object_pose_data_provider.get_pose_at_timestamp(
                ts_ns, time_query_options, time_domain)
            if pose_info is None:
                continue
            poses = pose_info.pose3d_collection.poses
            for obj_uid, pose in poses.items():
                if obj_uid not in T_o2w_dict:
                    T_o2w_dict[obj_uid] = {}
                T_o2w = pose.T_world_object.to_matrix()
                T_o2w_dict[obj_uid][ts_ns] = T_o2w

        return T_o2w_dict

    def get_object_mesh(self, cat: str):
        return self.obj_mesh_dict[cat]

    def frame_to_timestamp(self, frame: int):
        """ 0 index """
        return self._frame2ts[frame]

    def timestamp_to_frame(self, ts: int):
        return self._ts2frame[ts]
    
    """ 3D functions """
    def hand_pca_pose(self, frame_idx, side: str):
        """ ret: (1, 15) """
        ts = self.frame_to_timestamp(frame_idx)
        hand_poses_with_dt = self._mano_hand_data_provider.get_pose_at_timestamp(
                timestamp_ns=ts,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE)
        hand_pose_collection = hand_poses_with_dt.pose3d_collection

        # world space
        hpose = None
        for hpose_data in hand_pose_collection.poses.values():
            if hpose_data.is_left_hand() and side == LEFT:
                hpose = hpose_data
                break
            if hpose_data.is_right_hand() and side == RIGHT:
                hpose = hpose_data
                break
        if hpose is None:
            jangles = [-1.0 for _ in range(15)]
        else:
            jangles = hpose.joint_angles
        return torch.tensor(jangles).view(1, 15)
    
    def hand_joint_angles(self, frame_idx, side: str):
        """ ret: (1, 45), for flat_hand_mean==False
        """
        pca_pose = self.hand_pca_pose(frame_idx, side)
        mano = self.l_mano if side == LEFT else self.r_mano
        M_selected_componetns = mano.th_selected_comps
        hands_mean = mano.th_hands_mean
        jangles = pca_pose @ M_selected_componetns
        jangles = jangles + hands_mean
        return jangles

    def hand_verts_from_api(self, frame_idx, side: str, space='ego', as_mesh=False):
        ts = self.frame_to_timestamp(frame_idx)
        provider = self._mano_hand_data_provider
        hand_poses_with_dt = self._mano_hand_data_provider.get_pose_at_timestamp(
                timestamp_ns=ts,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE)
        hand_pose_collection = hand_poses_with_dt.pose3d_collection

        # world space
        vl, vr = None, None
        for hand_pose_data in hand_pose_collection.poses.values():
            if hand_pose_data.is_left_hand():
                vl = provider.get_hand_mesh_vertices(hand_pose_data)
            if hand_pose_data.is_right_hand():
                vr = provider.get_hand_mesh_vertices(hand_pose_data)

        if side == LEFT:
            verts = vl
            faces = self.fl
        elif side == RIGHT:
            verts = vr
            faces = self.fr
        else:
            raise ValueError(f'Unknown hand: {side}')

        if verts is None:
            return None

        if space == 'world':
            pass  # Already in world space
        elif space == 'ego':
            T_w2c = self.T_w2c[frame_idx]
            verts = rearrange(verts, 'v d -> d v')
            verts = T_w2c[:3, :3] @ verts + T_w2c[:3, [3]]
            verts = rearrange(verts, 'd v -> v d')
        else:
            raise ValueError(f'Unknown space: {space}')

        if as_mesh:
            return SimpleMesh(verts=verts, faces=faces, tex_color='green')
        return verts
    
    def hand_tips(self, frame_idx, side: str, space='ego'):
        """ Useful for analysing finger pose variation. See SG-Study in the paper. """
        raise NotImplementedError
    
    def hand_verts_homan(self, frame_idx, side: str, space='ego', as_mesh=False):
        """ Compute hand vertices from mano_tracer,
        This is to ensure the hand is consistent with HOMan's implementation,
        this put the hand wrist at the coordinate's origin.
        """
        ts = self.frame_to_timestamp(frame_idx)
        hand_poses_with_dt = self._mano_hand_data_provider.get_pose_at_timestamp(
                timestamp_ns=ts,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE)
        hand_pose_collection = hand_poses_with_dt.pose3d_collection

        def forward_hand_verts(side, hpose, shape_h):
            """ return:
            v_h: in hand space (1, 778, 3)
            tf: transform matrix (1, 4, 4)
            """
            assert side in {LEFT, RIGHT}, f'Unknown side: {side}'
            mano_layer = self.r_mano \
                if side == RIGHT else self.l_mano
            mano_tracer = self.r_mano_tracer \
                if side == RIGHT else self.l_mano_tracer
            T_h2w = torch.from_numpy(
                hpose.wrist_pose.to_matrix()).view(-1, 4, 4).float()
            rot_h = matrix_to_axis_angle(T_h2w[..., :3, :3])
            trans_h = T_h2w[..., :3, -1]
            jangles = torch.tensor(hpose.joint_angles).view(-1, 15)

            rot_zeros = torch.zeros([1, 3])
            trans_zeros = torch.zeros([1, 3])
            shape_h = self._mano_hand_data_provider._mano_shape_params.clone()
            shape_h = shape_h.view([1, 10]).float()
            v_h, _ = mano_layer.forward(
                torch.cat([rot_zeros, jangles], axis=-1),
                th_betas=shape_h, th_trans=trans_zeros)  # verts in hand space
            v_h /= 1000.
            _, _, tf = mano_tracer.forward_transform(
                torch.cat([rot_h, jangles], axis=-1),
                th_betas=shape_h, th_trans=trans_h)
            return v_h, tf

        # world space
        hpose = None
        for hpose_data in hand_pose_collection.poses.values():
            if hpose_data.is_left_hand() and side == LEFT:
                hpose = hpose_data
                break
            if hpose_data.is_right_hand() and side == RIGHT:
                hpose = hpose_data
                break
        shape_h = self._mano_hand_data_provider._mano_shape_params.clone()
        if hpose is None:
            return None
        v_h, h2w = forward_hand_verts(side, hpose, shape_h=shape_h)

        verts = None
        if space == 'world':
            v_w = pose_apply(h2w, v_h)
            verts = v_w
        elif space == 'ego':
            T_w2e = self.T_w2c[[frame_idx], ...]
            T_h2e = T_w2e @ h2w
            v_e = pose_apply(T_h2e, v_h)
            verts = v_e
        elif space == side:
            verts = v_h
        else:
            raise ValueError(f'Unknown space: {space}')
        
        verts = verts.squeeze(0)  # (V, 3)
        if as_mesh:
            faces = self.fl if side == LEFT else self.fr
            return SimpleMesh(verts=verts, faces=faces, tex_color='light_blue')
        
        return verts

    def hand_verts(self, frame_idx, hand: str, space='ego', as_mesh=False, use_api=False):
        if use_api:
            return self.hand_verts_from_api(frame_idx, hand, space, as_mesh)
        else:
            return self.hand_verts_homan(frame_idx, hand, space, as_mesh)

    def obj_verts(self, frame_idx, space='ego', as_mesh=False):
        vo_orig = rearrange(self.vo_orig, 'v d -> d v')
        if space == 'ego':
            T_o2c = self.T_o2c[frame_idx]
            vo = T_o2c[:3, :3] @ vo_orig + T_o2c[:3, [3]]
            vo = rearrange(vo, 'd v -> v d')
        elif space == 'world':
            T_o2w = self.T_o2w[frame_idx]
            vo = T_o2w[:3, :3] @ vo_orig + T_o2w[:3, [3]]
            vo = rearrange(vo, 'd v -> v d')
        else:
            raise ValueError(f'Unknown space: {space}')

        if as_mesh:
            return SimpleMesh(verts=vo, faces=self.fo, tex_color='yellow')
        return vo

    def joint_angles(self, frames, side: str):
        """ first 15 components of mano pca pose.
        ret: (T, 15)
        """
        joint_angles = []
        for frame_idx in frames:
            ts = self.frame_to_timestamp(frame_idx)
            hand_poses_with_dt = self._mano_hand_data_provider.get_pose_at_timestamp(
                    timestamp_ns=ts,
                    time_query_options=TimeQueryOptions.CLOSEST,
                    time_domain=TimeDomain.TIME_CODE)
            hand_pose_collection = hand_poses_with_dt.pose3d_collection

            # world space
            hpose = None
            for hpose_data in hand_pose_collection.poses.values():
                if hpose_data.is_left_hand() and side == LEFT:
                    hpose = hpose_data
                    break
                if hpose_data.is_right_hand() and side == RIGHT:
                    hpose = hpose_data
                    break
            if hpose is None:
                jangles = [-1.0 for _ in range(15)]
            else:
                jangles = hpose.joint_angles
            joint_angles.append(jangles)
        joint_angles = torch.tensor(joint_angles)
        return joint_angles
    
    def poses_hand2ego_batch(self, frames, side: str):
        """
        Args:
            frames: (T, )
        Returns:
            (T, 4, 4)
        """
        T_h2w = self.poses_hand2world_batch(frames, side)
        T_w2e = self.T_w2c[frames, ...]
        T_h2e = T_w2e @ T_h2w
        return T_h2e
    
    def poses_hand2world_batch(self, frames, side: str):
        """ returns: (T, 4, 4) """
        batch_jangles = []
        batch_rot_h = []
        batch_trans_h = []
        for frame_idx in frames:
            ts = self.frame_to_timestamp(frame_idx)
            provider = self._mano_hand_data_provider
            hand_poses_with_dt = provider.get_pose_at_timestamp(
                timestamp_ns=ts,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE)
            hand_pose_collection = hand_poses_with_dt.pose3d_collection

            hpose = None
            for hand_pose_data in hand_pose_collection.poses.values():
                if hand_pose_data.is_left_hand() and side == LEFT:
                    hpose = hand_pose_data
                    break
                if hand_pose_data.is_right_hand() and side == RIGHT:
                    hpose = hand_pose_data
                    break
            
            mano_tracer = self.r_mano_tracer \
                if side == 'right' else self.l_mano_tracer
            T_h2w = torch.from_numpy(
                hpose.wrist_pose.to_matrix()).float()
            rot_h = matrix_to_axis_angle(T_h2w[..., :3, :3])
            trans_h = T_h2w[..., :3, -1]
            jangles = torch.tensor(hpose.joint_angles)
            batch_jangles.append(jangles)
            batch_rot_h.append(rot_h)
            batch_trans_h.append(trans_h)
        
        jangles = rearrange(batch_jangles, 't d -> t d')
        rot_h = rearrange(batch_rot_h, 't d -> t d')  # in axis-angle, hence one-dim
        trans_h = rearrange(batch_trans_h, 't d -> t d')
        shape_h = self._mano_hand_data_provider._mano_shape_params.clone()
        shape_h = repeat(shape_h, 'd -> t d', t=len(frames)).float()
        _, _, tf = mano_tracer.forward_transform(
            torch.cat([rot_h, jangles], axis=-1),
            th_betas=shape_h, th_trans=trans_h)
        return tf

    def pose_hand2ego_single(self, frame_idx, side):
        """ ret: (4, 4) """
        h2e = self.poses_hand2ego_batch([frame_idx], side)
        h2e = h2e.view(4, 4)
        return h2e

    def pose_hand2world_single(self, frame_idx, side):
        """ ret: (4, 4) """
        h2w = self.poses_hand2world_batch([frame_idx], side)
        h2w = h2w.view(4, 4)
        return h2w

    def pose_obj2world(self, cat, frames):
        """
        Args:
            frames: (T, )
        Returns:
            o2w: (T, 4, 4)
        """
        obj_uid = self.cat_to_uid[cat]
        _T_o2w_dict = self._T_o2w_dict[obj_uid]
        T_o2w = []
        for frame in frames:
            ts = self.frame_to_timestamp(frame)
            if ts not in _T_o2w_dict:
                T_o2w.append(np.ones((4, 4)) * -1)
            else:
                T_o2w.append(self._T_o2w_dict[obj_uid][ts])
        T_o2w = torch.from_numpy(rearrange(T_o2w, 'f m n -> f m n')).float()  # (T, 4, 4)
        return T_o2w

    def poses_obj2hand_batch(self, frames, cat, side):
        """ ret: (T, 4, 4) """
        T_o2w = self.pose_obj2world(cat, frames)
        T_h2w = self.poses_hand2world_batch(frames, side)
        T_o2h = T_h2w.inverse() @ T_o2w
        return T_o2h

    """ 2D functions """

    def get_boxes(self, frame_idx):
        """ (x0, y0, w, h, visiblity) in the rotated image
        TODO: undistort the boxes?
        """
        cat = self.cat
        def rotate_box(box):
            # l, t, w, h, vis = box
            l, t, w, h = box
            return np.asarray([self.img_h - t - h, l, h, w]) # , vis])
        return map(rotate_box,
            (self._lboxes_orig[frame_idx],
            self._rboxes_orig[frame_idx],
            self._obj_boxes_orig[cat][frame_idx]))

    def read_image(self, frame: int):
        """ 0 index, return undistorted image (pinhole) """
        ts_ns = self.frame_to_timestamp(frame)
        img = self.get_undistorted_image(ts_ns, self._stream_id)
        if img is None:
            return np.zeros((self.img_w, self.img_h, 3), dtype=np.uint8)
        # Adding fake channels so we can process them downline during visualisation. Else there are errors in libzhifan
        img = np.stack((img,) * 3, axis=-1) if self.quest_sequence else img
        img = np.rot90(img, -1)
        return img

    def render_image_boxes(self, frame_idx) -> Image.Image:
        lbox, rbox, obox = self.get_boxes(frame_idx)
        lbox = lbox[:4]
        rbox = rbox[:4]
        obox = obox[:4]
        img_pil = Image.fromarray(self.read_image(frame_idx))
        img_pil = odlib.draw_bboxes_image_array(img_pil, obox[None], color='red')
        img_pil = odlib.draw_bboxes_image_array(img_pil, lbox[None], color='green')
        img_pil = odlib.draw_bboxes_image_array(img_pil, rbox[None], color='blue')
        return np.asarray(img_pil)

    def render_hand_object_mask(self, frame_idx, use_disk: bool, blend=False):
        """
        Returns:
            sil: (H, W), dtype=np.uint8
                0 for bg, 1 left, 2 right, 3 object
        """
        if use_disk:
            mask_path = MASK_FMT.format(seq_name=self.seq_name, frame_idx=frame_idx)
            mask = np.asarray(Image.open(mask_path))
            if blend:
                image = self.read_image(frame_idx)
                masked = image.copy()
                for i in np.unique(rend):
                    if i == 0:
                        continue
                    mask = rend == i
                    masked[mask] = self._maskid2color[i]
                blend = cv2.addWeighted(image, 0.5, masked, 0.5, 1.0)
                return blend
            return mask

        use_api = False
        left = self.hand_verts(frame_idx, 'left', space='ego', as_mesh=True, use_api=use_api)
        right = self.hand_verts(frame_idx, 'right', space='ego', as_mesh=True, use_api=use_api)
        cam_manager = self.batch_camera_manager[frame_idx]

        proj_method=dict(name='pytorch3d_instance', coor_sys='nr', in_ndc=False,
            blur_radius=1e-7, max_faces_per_bin=100000)
        obj_cam = self.obj_verts(frame_idx, space='ego', as_mesh=True)

        meshes = []
        mesh_to_id = []  # left-1, right-2, obj-3
        if left is not None:
            meshes.append(left)
            mesh_to_id.append(1)
        if right is not None:
            meshes.append(right)
            mesh_to_id.append(2)
        if obj_cam is not None:
            meshes.append(obj_cam)
            mesh_to_id.append(3)
        proj_method.update({
            'mesh_to_id': mesh_to_id
        })

        rend = projection.perspective_projection_by_camera(
            meshes,
            camera=cam_manager,
            method=proj_method)
        rend = rend.astype(np.uint8)

        if blend:
            image = self.read_image(frame_idx)
            id_to_color = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0)}
            masked = image.copy()
            for i in np.unique(rend):
                if i == 0:
                    continue
                mask = rend == i
                masked[mask] = id_to_color[i]
            blend = cv2.addWeighted(image, 0.5, masked, 0.5, 1.0)
            return blend

        return rend

    def get_mask_mapping(self):
        # Mask mapping will not be affected by skipped reduced meshes
        instance_mapping = io.read_json(
            osp.join(self.data_root, 'assets/instance.json'))
        uids = set(self.box2d_objects_df['object_uid'].unique())
        all_cat_to_uid = {}
        for obj_uid in uids:
            cat = instance_mapping[str(obj_uid)]['instance_name']
            all_cat_to_uid[cat] = str(obj_uid)

        mask_mapping = {'left': 1, 'right': 2}
        for cat in sorted(all_cat_to_uid.keys()):
            mask_mapping[cat] = len(mask_mapping) + 1
        return mask_mapping

    def load_fg_masks(self, cat: str, frames: list,
                      ignore_other_objects=False):
        """
        Load foreground mask of a perticular category.
        set hands to -1,
        if `ignore_other_objects`, set other objects to -1(ignore),
            otherwise 0(bg)
        """
        masks = []
        c = self.get_mask_mapping()[cat]
        for f in frames:
            # sil: (H, W), dtype=np.uint8
            #     0 for bg, 1 left, 2 right, 3 object
            sil = self.render_hand_object_mask(f, use_disk=True)
            obj_mask = torch.zeros([*sil.shape[:2]])
            obj_mask[sil == c] = 1
            obj_mask[sil == 1] = -1  # left hand
            obj_mask[sil == 2] = -1  # right hand
            if ignore_other_objects:
                for k, v in self.get_mask_mapping().items():
                    if k == cat:
                        continue
                    obj_mask[sil == v] = -1
            masks.append(obj_mask)
        masks = rearrange(masks, 'b h w -> b h w')
        return masks

    def render_all_hand_object_mask(self,
                                    frame_idx,
                                    use_disk=False,
                                    blend=False):
        if use_disk:
            mask_path = MASK_FMT.format(seq_name=self.seq_name, frame_idx=frame_idx)
            rend = np.asarray(Image.open(mask_path))

            if blend:
                image = self.read_image(frame_idx)
                masked = image.copy()
                for i in np.unique(rend):
                    if i == 0:
                        continue
                    mask = rend == i
                    masked[mask] = self._maskid2color[i]
                blend = cv2.addWeighted(image, 0.5, masked, 0.5, 1.0)
                return blend
            return rend

        ts = self._frame2ts[frame_idx]
        use_api = False
        left = self.hand_verts(frame_idx, 'left', space='ego', as_mesh=True, use_api=use_api)
        right = self.hand_verts(frame_idx, 'right', space='ego', as_mesh=True, use_api=use_api)
        mask_mapping = self.get_mask_mapping()
        meshes, mesh_to_id = [], []
        if left is not None:
            meshes.append(left)
            mesh_to_id.append(mask_mapping['left'])
        if right is not None:
            meshes.append(right)
            mesh_to_id.append(mask_mapping['right'])
        T_c2w = self.T_c2w[frame_idx]
        for cat in sorted(self.cat_to_uid.keys()):
            obj_uid = self.cat_to_uid[cat]
            obj_mesh_orig = self.obj_mesh_dict[cat]
            vo_orig = torch.from_numpy(obj_mesh_orig.vertices).float()

            str_obj_uid = str(obj_uid)
            if str_obj_uid not in self._T_o2w_dict \
                or ts not in self._T_o2w_dict[str_obj_uid]:
                continue
            T_o2w = self._T_o2w_dict[str_obj_uid][ts]
            T_o2w = torch.from_numpy(T_o2w).float()
            """ T_o2c: (4, 4) """
            T_o2c = T_c2w.inverse() @ T_o2w

            vo_orig = rearrange(vo_orig, 'v d -> d v')
            vo = T_o2c[:3, :3] @ vo_orig + T_o2c[:3, [3]]
            vo = rearrange(vo, 'd v -> v d')
            obj_cam = SimpleMesh(verts=vo, faces=obj_mesh_orig.faces)
            # print(cat, obj_cam)
            meshes.append(obj_cam)
            mesh_to_id.append(mask_mapping[cat])

        proj_method=dict(
            name='pytorch3d_instance', coor_sys='nr', in_ndc=False,
            blur_radius=1e-7, # max_faces_per_bin=100000,
            # bin_size=0,
            mesh_to_id=mesh_to_id)
        if self.obj_version == 'original':
            proj_method['bin_size'] = 0

        cam_manager = self.batch_camera_manager[frame_idx]
        rend = projection.perspective_projection_by_camera(
            meshes,
            camera=cam_manager,
            method=proj_method)
        rend = rend.astype(np.uint8)

        if blend:
            image = self.read_image(frame_idx)
            masked = image.copy()
            for i in np.unique(rend):
                if i == 0:
                    continue
                mask = rend == i
                masked[mask] = self._maskid2color[i]
            blend = cv2.addWeighted(image, 0.5, masked, 0.5, 1.0)
            return blend

        return rend

    def generate_video(self, st: int, ed: int, out_dir, fps=15):
        os.makedirs(out_dir, exist_ok=True)
        images = []
        import tqdm
        for f in tqdm.trange(st, ed+1):
            img = self.render_all_hand_object_mask(
                f, use_disk=True, blend=True)
            # img = self.read_image(f)
            font = cv2.FONT_HERSHEY_DUPLEX
            img = cv2.putText(img, f'{f=}', (10, 60), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            images.append(img)

        clip = editor.ImageSequenceClip(images, fps=fps)
        clip.write_videofile(osp.join(
            out_dir, f'{self.seq_name}-{st}-{ed}.mp4'), logger=None, verbose=False)
