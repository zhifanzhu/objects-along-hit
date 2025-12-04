from typing import List
import os
import os.path as osp
import numpy as np
import torch
import pandas as pd
from dotmap import DotMap
from torch.utils.data import Dataset
from einops import rearrange, repeat
from code_epichor.epic_tlreader import EPICTimelineReader
from nnutils.image_utils import square_bbox_xywh, batch_crop_resize

from libzhifan import io
from potim.defs.types import (
    PotimSegment,
    INHAND, SCENE_STATIC, SCENE_DYNAMIC)
from potim.defs.types import PotimSegment, get_sample_indices, fixed_segment_sampling
from potim.utils.scene_static import (
    setup_inits_o2w_extra_disk_results,
)
from potim.utils.rand_inhand_pose_init import random_inhand_upright_epic
from potim.utils.scene_static_epic import setup_inits_o2w_multi_upright
from potim.utils.cmd_logger import getLogger

logger = getLogger(__name__)


SAM2_QUALITY_CSV = 'code_epichor/timelines/sam2_quality.csv'
EPICFIELD_METRIC_TRANSFORM_DIR = 'weights/epicfields_metric_transform'


class _EPICSingleVideo(Dataset):

    def __init__(self,
                 reader: EPICTimelineReader,
                 timeline,
                 max_samples_per_seg=30,
                 roi_box_expand=0.4,
                 valid_frame_pixel_thr=25,
                 occlude_level='all',
                 inhand_pose_init='learnt',
                 skip_dynamic=False,
                 force_max_samples=True,
                 static_init_method='multi_upright',
                 display_raw_hands=False,
                 **kwargs,
                 ):
        """
        Args:
            display_raw_hands: in case we need to debug raw hand in o3d views, set this to True
        """
        self.reader = reader

        self.max_samples_per_seg = max_samples_per_seg
        self.roi_box_expand = roi_box_expand
        self.valid_frame_pixel_thr = valid_frame_pixel_thr
        if occlude_level == 'all':
            self.ignore_other_objects = True
        elif occlude_level == 'ho':
            self.ignore_other_objects = False
        else:
            raise ValueError(f'Unknown {occlude_level=}')
        self.inhand_pose_init = inhand_pose_init
        self.force_max_samples = force_max_samples
        self.display_raw_hands = display_raw_hands
        self.set_static_init_method(static_init_method, None)

        if skip_dynamic:
            timeline['segments'] = [
                seg for seg in timeline['segments'] if seg['ref'] != SCENE_DYNAMIC]
        else:
            pass

        # replicate timeline for each segment
        self.single_segment_timelines = []
        for seg in timeline['segments']:
            single_seg_tl = timeline.copy()
            single_seg_tl['segments'] = [seg]
            self.single_segment_timelines.append(single_seg_tl)
        self.num_segments = len(self.single_segment_timelines)

        """ Raw timeline info """
        self.tl = timeline
        self.segments = self.tl['segments']

    def __len__(self):
        return self.num_segments

    def get_ref(self, idx):
        return self.single_segment_timelines[idx]['segments'][0]['ref']
    
    def set_static_init_method(self, static_o2w_init: str, inhand_load_from: str):
        """
        Args:
            static_o2w_init: 'upright', 'upright_and_disk'
        """
        if static_o2w_init == 'upright':
            raise ValueError("Deprecated. Use 'multi_upright' instead")
            def init_func(D, i):
                """ ret: Sim(3), num_inits """
                sta_seg_init = setup_inits_o2w_multi_upright(
                    D, index_of_static=i, estimate_scale=False,
                    num_rots_asym=1)
                num_inits = 1
                return sta_seg_init, num_inits
        elif static_o2w_init == 'multi_upright':
            def init_func(D, i):
                """ ret: Sim(3), num_inits """
                num_rots_asym = 10
                sta_seg_init = setup_inits_o2w_multi_upright(
                    D, index_of_static=i, estimate_scale=False,
                    num_rots_asym=num_rots_asym)
                num_inits = len(sta_seg_init.rot)
                return sta_seg_init, num_inits
        elif static_o2w_init == 'upright_and_disk':
            raise ValueError("Deprecated. Use 'multi_upright' instead")
            # This should be obsolete since v6
            def init_func(D, i):
                sta_seg_init = setup_inits_o2w_extra_disk_results(
                    D, D.meta_samples.frames_per_seg[i], inhand_load_from, 
                    use_transl=True, use_scale=False)
                num_inits = len(sta_seg_init.rot)
                return sta_seg_init, num_inits
        else:
            raise ValueError(f"Unkown {static_o2w_init=}")
        self.static_init_func = init_func
    
    def scale_init_func(self, tl, num_inits):
        """
        Returns:
            scale_hand, scale_obj: (N,)
        """
        scale_hand = torch.ones([num_inits]).float()
        scale_obj = torch.ones([num_inits]).float()
        return scale_hand, scale_obj

    def __getitem__(self, idx, tl=None):
        """
        Args:
            tl: By default is None.
                Sometime we want to forge non-single-segment-timelines to pass in,
                e.g. joint_static
        """
        if tl is None:
            tl = self.single_segment_timelines[idx]
        reader = self.reader

        D = DotMap()
        D.dataset_name = 'epic'
        D.has_3d_gt = False
        D.timeline = tl
        D.timeline_name = tl['timeline_name']
        D.cat = tl['cat']
        all_potim_segments = [
            PotimSegment(
                st=seg['st'], ed=seg['ed'], ref=seg['ref'], side=seg['side'])
            for seg in tl['segments']]
        valid_frames = self.prepare_valid_frames(tl['segments'])
        D.meta_samples = fixed_segment_sampling(
            all_potim_segments, valid_frames,
            max_samples=self.max_samples_per_seg, force_max_samples=self.force_max_samples)
        D.segments = D.meta_samples.segments
        D.num_samples = D.meta_samples.num_samples
        if D.num_samples == 0:
            return None

        # Read w2cs for all segments
        D.w2cs = []
        for i, seg in enumerate(D.segments):
            frames_seg = D.meta_samples.frames_per_seg[i]
            w2c = reader.read_w2c(frames_seg)
            D.w2cs.append(w2c)

        # Optimisation targets (mask, verts, faces, hands=None for now)
        v_obj, f_obj = reader.obj_verts, reader.obj_faces
        D.obj = DotMap()
        D.obj.verts = torch.from_numpy(v_obj).float()
        D.obj.faces = torch.from_numpy(f_obj).long()

        D.obj.mask = reader.load_fg_masks(
            D.meta_samples.nonunique_frames,
            ignore_other_objects=self.ignore_other_objects)
        D.images = []
        for f in D.meta_samples.nonunique_frames:
            D.images.append(reader.read_image(f))
        D.images = rearrange(D.images, 'b h w c -> b h w c')
        # Obj bboxes (from masks)
        D.roi_box_expand = self.roi_box_expand
        D.obj.bboxes = []
        for i, m in enumerate(D.obj.mask):
            _y, _x = np.nonzero(m==1).split(dim=1, split_size=1)
            bbox = torch.tensor([_x.min(), _y.min(), _x.max() - _x.min(), _y.max() - _y.min()]).float()
            D.obj.bboxes.append(bbox)
        D.obj.bboxes = torch.stack(D.obj.bboxes)

        # ROI Crop image & masks
        D.rend_size = 256
        D.rois = square_bbox_xywh(D.obj.bboxes, D.roi_box_expand).int()
        D.obj.mask_patch = batch_crop_resize(D.obj.mask, D.rois, D.rend_size, fill=-1)
        D.image_patches = batch_crop_resize(D.images, D.rois, D.rend_size, fill=0)

        # Camera
        frames = D.meta_samples.nonunique_frames
        global_cam = reader.global_cam.repeat(len(frames))
        D.global_cam = global_cam
        D.roi_h = torch.ones([len(global_cam)]) * D.rend_size
        D.roi_w = torch.ones([len(global_cam)]) * D.rend_size
        D.roi_cam = D.global_cam.crop(D.rois).resize(D.roi_h, D.roi_w)
        D.roi_camintr = D.roi_cam.to_nr(return_mat=True)

        D.num_inits = 0
        # Setup inits o2w, used by static
        D.static_inits_o2w_list = []
        for i, seg in enumerate(D.segments):
            sta_seg_init = None
            if seg.ref == SCENE_STATIC:
                sta_seg_init, num_inits = self.static_init_func(D, i)
                D.static_inits_o2w_list.append(sta_seg_init)
                D.num_inits = num_inits
            else:
                D.static_inits_o2w_list.append(None)

        # Init o2h for testing NO-INIT-PASSING
        if self.inhand_pose_init == 'learnt':
            D.init_o2h_list = []
            for i, seg in enumerate(D.segments):
                # each o2h is (N, 4, 4) as we only init to static
                if seg.ref == SCENE_STATIC:
                    D.init_o2h_list.append(None)
                    continue
                elif seg.ref == SCENE_DYNAMIC:
                    # Use a phony identity
                    T_o2h = torch.eye(4).view(1, 4, 4)
                    D.num_inits = 1
                    D.init_o2h_list.append(T_o2h)
                elif seg.ref == INHAND:
                    T_o2h = reader.o2h_init_poses(seg.side, D.cat)
                    D.num_inits = T_o2h.shape[0]
                    D.init_o2h_list.append(T_o2h)
        elif self.inhand_pose_init == 'random':
            D.init_o2h_list = []
            for i, seg in enumerate(D.segments):
                # each o2h is (N, 4, 4) as we only init to static
                if seg.ref != INHAND:
                    raise NotImplementedError("need to implement the scene_dynamic")
                    D.init_o2h_list.append(None)
                    continue
                T_o2h = random_inhand_upright_epic(D.cat)
                D.num_inits = T_o2h.shape[0]
                D.init_o2h_list.append(T_o2h)
        elif self.inhand_pose_init is None:
            D.init_o2h_list = []
            for i, seg in enumerate(D.segments):
                if seg.ref != INHAND:
                    D.init_o2h_list.append(None)
                    continue
                T_o2h = torch.empty(0, 4, 4)
                D.num_inits = 0
                D.init_o2h_list.append(T_o2h)
        else:
            raise ValueError(f'Unknown {self.inhand_pose_init=}')

        # Load scale 
        D.scale_hand, D.scale_obj = self.scale_init_func(tl, D.num_inits)

        # Setup T_h2c for inhand, also setup v_hand for inhand
        D.T_h2cs = []
        D.vhand_ego_list = []
        D.f_hand_list = []
        D.vhand_untransl_list = []  # eventually we can try to remove them to simplify the code
        D.t_h2c_untransl_list = []  # eventually we can try to remove them to simplify the code
        for i, seg in enumerate(D.segments):
            frames_seg = D.meta_samples.frames_per_seg[i]
            faces_hand = None
            T_h2c = None
            vh_egos = None
            vh_untransl = None
            t_h2c_untransl = None
            if seg.side in {'left', 'right'}:
                faces_hand = reader.fl if seg.side == 'left' else reader.fr  # already in (1, F, 3) and Tensor
                T_h2c, vh_egos, vh_untransl, t_h2c_untransl = reader.read_hamer_poly_scales(
                    frames_seg, seg.side, scale_hand=D.scale_hand
                ) # T_h2c : (N, T, 4, 4), vh_egos: (N, T, V, 3)

            D.T_h2cs.append(T_h2c)
            D.vhand_ego_list.append(vh_egos)
            D.f_hand_list.append(faces_hand)
            D.vhand_untransl_list.append(vh_untransl)
            D.t_h2c_untransl_list.append(t_h2c_untransl)
        
        # For Debugging fit_hand_scale: Raw hand scale for any available hand in any available frames
        #   Access by D.hands.vl_list[abs_ind]
        if self.display_raw_hands:
            logger.debug("Reading raw hands")
            D.hands = DotMap(fl=reader.fl, fr=reader.fr)  # (1, F, 3)
            D.hands.vl_list = []
            D.hands.vr_list = []
            hamer_avail_frames = self.reader.get_hamer_avail_frames()
            for i, seg in enumerate(D.segments):
                for f in D.meta_samples.frames_per_seg[i]:
                    vl, vr = None, None
                    for side in ('left', 'right'):
                        if f in hamer_avail_frames[side]:
                            _, vh, _, _ = reader.read_hamer([f], side)  # TODO; how do I reflect updated scale_hand into visualisation?
                            vh = vh.view(-1, 3)
                            if side == 'left':
                                vl = vh
                            else:
                                vr = vh
                    D.hands.vl_list.append(vl)
                    D.hands.vr_list.append(vr)
            logger.debug("DONE Reading raw hands")
        else:
            D.hands = None

        D.o3d = self.prepare_o3d_hover_data(D)
        return D

    def prepare_valid_frames(self, segments) -> List[List[int]]:
        """ Given segments, return valid frames for each segment
        a valid frame mean both hand pose and object mask&pose are available.
        """
        valid_frames = self.reader.get_valid_frames()
        ret_valid_frames = []
        for seg in segments:
            if seg['ref'] == INHAND or (seg['ref'] == SCENE_DYNAMIC and seg['side'] in {'left', 'right'}):
                hamer_avail_frames = self.reader.hamer_loader.avail_side_frames(
                    self.reader.vid)
                hamer_avail_frames = hamer_avail_frames[seg['side']]
                _valid_frames = list(set(valid_frames) & set(hamer_avail_frames))
            else:
                _valid_frames = valid_frames.copy()
            _valid_frames = torch.tensor(_valid_frames)
            _valid_frames = _valid_frames[
                (seg['st'] <= _valid_frames) &
                (_valid_frames <= seg['ed'])]
            _valid_frames = _valid_frames.tolist()
            ret_valid_frames.append(_valid_frames)
        return ret_valid_frames

    """ Open3d Viz """
    def prepare_o3d_hover_data(self, D):
        from potim.utils.open3d.helper import get_material, get_frustum
        from potim.utils.o3d_viewcontrol import calc_front_viewing_cam # calc_scene_viewing_cam
        sun_light = True
        yellow = get_material('yellow', shader='defaultLit')
        purple = get_material('purple', shader='defaultLit')
        red = get_material('red', shader='unlitLine')  # 'unlitLine' necessary to avoid verbose warning
        white = get_material('white', point_size=0.01) # , shader='defaultLit')
        if not sun_light:
            yellow = get_material('yellow', shader='defaultUnlit')  # 'Unlit' necessary, otherwise object too dim

        data = DotMap()
        m = DotMap()
        m.white = white
        m.yellow = yellow
        m.purple = purple
        m.red = red
        data.m = m
        data.sun_light = sun_light

        # Don't show pcd as they are just too messy.
        # data.pcd_mesh = self.reader.get_sfm_pcd_open3d(
        #     as_mesh=False, voxel_down_sample=0.01)  # PCD, 1cm
        
        # Build w2c_frames from segments across all segments, not just current D's segments
        if (use_all_segments := True):
            segments = self.tl['segments']
        else:
            segments = D.segments
        valid_frames = self.prepare_valid_frames(segments)
        all_potim_segments = [
            PotimSegment(
                st=seg['st'], ed=seg['ed'], ref=seg['ref'], side=seg['side'])
            for seg in segments]
        meta_samples = fixed_segment_sampling(
            all_potim_segments, valid_frames,
            max_samples=self.max_samples_per_seg, force_max_samples=self.force_max_samples)
        w2c_frames = meta_samples.nonunique_frames

        all_w2cs = self.reader.read_w2c(frames=w2c_frames)
        all_c2ws = torch.inverse(all_w2cs)
        data.view_cam = calc_front_viewing_cam(all_c2ws) # view_cam = calc_scene_viewing_cam(c2ws)
        data.all_c2ws = all_c2ws  # store this in case we want to modify view_cam later
        data.fov = 60

        # Frustums have to be current D's
        w2cs = self.reader.read_w2c(frames=D.meta_samples.nonunique_frames)
        c2ws = torch.inverse(w2cs)
        data.frustums = []
        for c2w in c2ws:
            ego_frustum = get_frustum(c2w)
            data.frustums.append(ego_frustum)
        return data


class EPICOnDemandDataset(Dataset):

    def __init__(self,
                 timeline_json_path,
                 **dataset_kwargs):
        super().__init__()
        self.timelines = io.read_json(timeline_json_path)
        self.dataset_kwargs = dataset_kwargs
        verbose = False
        if verbose:
            print(f"Loaded {len(self.timelines)} timelines")

        # Filter SAM2 mask quality
        df = pd.read_csv(SAM2_QUALITY_CSV)
        video_names = set(df[df['usable_masks'] == 1].video_name)
        self.timelines = [tl for tl in self.timelines if tl['mp4_name'] in video_names]
        if verbose:
            print(f"Filtered {len(self.timelines)} timelines with SAM2 mask quality")

        # Filter valid epic-fields-metric-transform
        valid_metric = set([v.replace('.json', '')
             for v in os.listdir(EPICFIELD_METRIC_TRANSFORM_DIR) if v.endswith('.json')])
        self.timelines = [tl for tl in self.timelines if tl['vid'] in valid_metric]
        if verbose:
            print(f"Filtered {len(self.timelines)} timelines with valid epic-fields-metric-transform")

        # Filter tl that has None in 'scene_dynamic'
        tls = []
        for tl in self.timelines:
            skip = False
            for seg in tl['segments']:
                if seg['ref'] == SCENE_DYNAMIC and seg['side'] not in {'left', 'right'}:
                    skip = True
                    break
            if not skip:
                tls.append(tl)
        self.timelines = tls
        if verbose:
            print("Filtered tl that has None in 'scene_dynamic'")

    def __len__(self):
        return len(self.timelines)

    def locate_index_from_output(self, timeline_name) -> int:
        return [tl['timeline_name'] for tl in self.timelines].index(timeline_name)

    def get_num_segments(self, idx):
        skip_dynamic = self.dataset_kwargs['skip_dynamic']
        _segments = self.timelines[idx]['segments']
        if skip_dynamic:
            _segments = [
                seg for seg in _segments if seg['ref'] != SCENE_DYNAMIC]
        return len(_segments)

    def __getitem__(self, idx):
        tl = self.timelines[idx]
        reader = EPICTimelineReader(tl['timeline_name'], tl['mp4_name'])

        dataset_object = _EPICSingleVideo(
            reader=reader,
            timeline=tl,
            **self.dataset_kwargs)
        return dataset_object


if __name__ == '__main__':
    # dataset_kwargs =
    dataset_kwargs = dict(
        max_samples_per_seg=30,
        roi_box_expand=0.4,
        valid_frame_pixel_thr=25,
        occlude_level='all',
    )
    ds = EPICOnDemandDataset(
        timeline_json_path='./code_epichor/timelines/epic_hit.json',
        **dataset_kwargs)
    from pprint import pprint
    video_ds = ds[0]

    for D in video_ds:
        pprint(D)
