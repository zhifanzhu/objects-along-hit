from typing import Union, List
import os.path as osp
from libzhifan import io
import torch
import numpy as np
from einops import rearrange, repeat
from nnutils.image_utils import square_bbox_xywh, batch_crop_resize
from potim.defs.types import (
    PotimSegment,
    INHAND, SCENE_STATIC, SCENE_DYNAMIC)
from potim.defs.sim3 import Sim3
from torch.utils.data import Dataset
from potim.defs.types import PotimSegment, get_sample_indices, fixed_segment_sampling
from dotmap import DotMap
from potim.utils.scene_static import setup_inits_o2w_upright, setup_inits_o2w_priors
from potim.utils.bop_pose_error import get_symmetry_transformations
from potim.utils.cmd_logger import getLogger
from potim.utils.open3d.helper import get_material, get_frustum
from potim.utils.o3d_viewcontrol import calc_front_viewing_cam # calc_scene_viewing_cam
from potim.utils.rand_inhand_pose_init import spiral_inhand_upright

from code_hot3d.hot3d_reader import HOT3DReader, MASK_FMT, HOT3DDATA_ROOT
from code_hot3d.hot3d_preextract_reader import HOT3DPreExtractReader

logger = getLogger(__name__)

def load_symmetry_transforms(cat):
    models_info = io.read_json('./weights/hot3d/models_info.json')
    models_info = {v['name']: v for v in models_info.values()}
    trans_list = get_symmetry_transformations(models_info[cat])
    return trans_list

class _HOT3DSingleVideo(Dataset):
    """ Each dataset object is a single video """
    def __init__(self,
                 reader: Union[HOT3DReader, HOT3DPreExtractReader],
                 timeline,
                 max_samples_per_seg=30,
                 roi_box_expand=0.4,
                 valid_frame_pixel_thr=25,
                 occlude_level='all',
                 inhand_pose_init='learnt-10',
                 inhand_only=False,
                 use_gt_init=False,
                 static_init_method='multi_upright',
                 skip_dynamic=False,
                 display_raw_hand=False,
                 **kwargs,
                 ):
        """
        Args:
            display_raw_hands: in case we need to debug GT hand in o3d views, set this to True
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
        self.inhand_only = inhand_only
        self.use_gt_init = use_gt_init
        self.static_init_method = static_init_method
        self.display_raw_hand = display_raw_hand

        if skip_dynamic:
            timeline['segments'] = [
                seg for seg in timeline['segments'] if seg['ref'] != SCENE_DYNAMIC]
        else:
            # Assigning handside according to contact frames
            seq_name = timeline['seq_name']
            cat = timeline['cat']
            segments = timeline['segments']
            info = io.read_pickle(f'./DATA_STORAGE/hot3d_dataset/grasp_information_v2/{seq_name}.pkl')
            new_segs = []
            for i, seg in enumerate(segments):
                if seg['ref'] != 'scene_dynamic':
                    new_segs.append(seg)
                    continue
                l_cnt, r_cnt = 0, 0
                for f in range(seg['st'], seg['ed']+1):
                    if info[f]['left_object'] == cat:
                        l_cnt += 1
                    if info[f]['right_object'] == cat:
                        r_cnt += 1
                side = 'left' if l_cnt > r_cnt else 'right'
                # if not (l_cnt > 0 or r_cnt > 0):
                #     print("Bad", tl['timeline_name'], seq_name, cat, i, seg)
                new_seg = seg.copy()
                new_seg['side'] = side
                new_segs.append(new_seg)
            timeline['segments'] = new_segs

        # replicate timeline for each segment
        self.single_segment_timelines = []
        for seg in timeline['segments']:
            single_seg_tl = timeline.copy()
            single_seg_tl['segments'] = [seg]
            self.single_segment_timelines.append(single_seg_tl)
        self.num_segments = len(self.single_segment_timelines)
        self.tl = timeline  # Still keep the raw timeline
        self.segments = self.tl['segments']

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        tl = self.single_segment_timelines[idx]
        reader = self.reader

        D = DotMap()
        D.dataset_name = 'hot3d'
        D.has_3d_gt = True
        D.timeline = tl
        D.timeline_name = tl['timeline_name']
        D.cat = tl['cat']

        # Special case, fix after cvpr: we sometimes have (st=50, ed=2) due to imperfect handling of first 50 frames
        # Return None for such cases, so as to keep the same segi
        tl['segments'][0]['st'] = max(50, tl['segments'][0]['st'])
        tl['segments'][0]['ed'] = max(50, tl['segments'][0]['ed'])
        if tl['segments'][0]['st'] <= 50 and  tl['segments'][0]['ed'] <= 50:
            return None

        all_potim_segments = [
            PotimSegment(
                st=seg['st'], ed=seg['ed'], ref=seg['ref'], side=seg['side'])
            for seg in tl['segments']]
        valid_frames = self.prepare_valid_frames(
            D.cat, tl['seq_name'], tl['segments'])
        D.meta_samples = fixed_segment_sampling(
            all_potim_segments, valid_frames, max_samples=self.max_samples_per_seg)
        D.segments = D.meta_samples.segments
        D.num_samples = D.meta_samples.num_samples
        if D.num_samples == 0:
            return None

        # Read w2cs for all segments
        T_c2w = reader.T_c2w
        T_w2c = T_c2w.inverse()
        D.w2cs = []
        for i, seg in enumerate(D.segments): # tl['segments']):
            frames_seg = D.meta_samples.frames_per_seg[i]
            w2c = T_w2c[frames_seg]
            D.w2cs.append(w2c)

        # Optimisation targets (mask, verts, faces, hands=None for now)
        mesh = reader.obj_mesh_dict[D['cat']]
        v_obj, f_obj = mesh.vertices, mesh.faces
        D.obj = DotMap()
        D.obj.verts = torch.from_numpy(v_obj).float()
        D.obj.faces = torch.from_numpy(f_obj).long()
        D.obj.diameter = reader.obj_diameters[D['cat']]
        D.obj.sym_transforms = load_symmetry_transforms(D['cat'])

        D.obj.mask = reader.load_fg_masks(
            D['cat'], D.meta_samples.nonunique_frames,
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
        global_cam = reader.batch_camera_manager[frames]
        D.global_cam = global_cam
        D.roi_h = torch.ones([len(global_cam)]) * D.rend_size
        D.roi_w = torch.ones([len(global_cam)]) * D.rend_size
        D.roi_cam = D.global_cam.crop(D.rois).resize(D.roi_h, D.roi_w)
        D.roi_camintr = D.roi_cam.to_nr(return_mat=True)

        # Setup inits o2w, used by static
        D.static_inits_o2w_list = []
        for i, seg in enumerate(D.segments):  # tl['segments']):
            sta_seg_init = None
            if seg.ref == SCENE_STATIC:
                if self.static_init_method in {'upright', 'upright-10', 'multi_upright'}:
                    num_rots = 10 if self.static_init_method in {'upright-10', 'multi_upright'} else 4
                    sta_seg_init = setup_inits_o2w_upright(
                        D, index_of_static=i,  num_rots=num_rots,
                        estimate_scale=False)
                    D.num_inits = len(sta_seg_init.rot)
                elif self.static_init_method == 'priors':
                    sta_seg_init = setup_inits_o2w_priors(
                        D, index_of_static=i, estimate_scale=False)
                    D.num_inits = 10
                elif self.static_init_method == 'passed':
                    sta_seg_init = None
                    D.num_inits = 1
                else:
                    raise ValueError(f"Unkown {self.static_init_method=}")
            D.static_inits_o2w_list.append(sta_seg_init)

        # Init o2h for testing NO-INIT-PASSING
        if self.inhand_pose_init == 'learnt-10' or self.inhand_pose_init == 'learnt-6':
            if self.inhand_pose_init == 'learnt-10':
                pose_priors = torch.load('./weights/pose_priors/hot3d/pose_priors_10.pth')
                D.num_inits = 10
            elif self.inhand_pose_init == 'learnt-6':
                pose_priors = torch.load('./weights/pose_priors/hot3d/pose_priors_6.pth')
                D.num_inits = 6
            pose_prior = pose_priors[D.cat]
            D.init_o2h_list = []
            for i, seg in enumerate(D.segments):
                # each o2h is (N, 4, 4) as we only init to static
                if seg.ref == SCENE_STATIC:
                    D.init_o2h_list.append(None)
                    continue
                elif seg.ref == SCENE_DYNAMIC:
                    # Mar-02 2025, to save computation, just init to identity.
                    T_o2h = torch.eye(4).view(1, 4, 4)
                    D.init_o2h_list.append(T_o2h)
                    D.num_inits = 1
                    continue

                if seg.side == 'left':
                    R_o2h = pose_prior['R_o2l'].clone()
                    t_o2h = pose_prior['t_o2l'].clone()
                elif seg.side == 'right':
                    R_o2h = pose_prior['R_o2r'].clone()
                    t_o2h = pose_prior['t_o2r'].clone()
                T_o2h = torch.zeros(len(R_o2h), 4, 4)
                T_o2h[:, :3, :3] = R_o2h
                T_o2h[:, :3, 3] = t_o2h
                T_o2h[:, -1, -1] = 1.0
                D.init_o2h_list.append(T_o2h)

        elif self.inhand_pose_init == 'gt':
            D.num_inits = 1
            D.init_o2h_list = []
            for seg, frames in zip(D.segments, D.meta_samples.frames_per_seg):
                gt_o2h = None
                if seg.ref == INHAND:
                    gt_o2h = reader.poses_obj2hand_batch(frames, D.cat, seg.side)
                D.init_o2h_list.append(gt_o2h)
        
        elif self.inhand_pose_init in {'random', 'random-10'}:
            num_inits = 10 if self.inhand_pose_init == 'random-10' else 4
            D.init_o2h_list = []
            for i, seg in enumerate(D.segments):
                # each o2h is (N, 4, 4) as we only init to static
                if seg.ref != INHAND:
                    D.init_o2h_list.append(None)
                    continue
                T_o2h = spiral_inhand_upright(num_inits)
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
        D.scale_hand = torch.ones([D.num_inits])
        D.scale_obj = torch.ones([D.num_inits])

        # Setup T_h2c for inhand, also setup v_hand for inhand
        D.T_h2cs = []
        D.vhand_ego_list = []
        D.f_hand_list = []
        D.vhand_untransl_list = []  # eventually we can try to remove them to simplify the code
        D.t_h2c_untransl_list = []  # eventually we can try to remove them to simplify the code
        for i, seg in enumerate(D.segments):
            frames_seg = D.meta_samples.frames_per_seg[i]
            T_h2c = None
            faces_hand = None
            vh_egos = None
            vh_untransl = None
            t_h2c_untransl = None
            if seg.ref == INHAND or (seg.ref == SCENE_DYNAMIC and seg.side is not None):
                # When using mano_tracer, T_h2c will be different from use_api.
                T_h2c = reader.poses_hand2ego_batch(frames_seg, seg.side)
                fh = reader.fl if seg.side == 'left' else reader.fr

                vh_egos = []
                for f in frames_seg:
                    vh_ego = reader.hand_verts_from_api(f, seg.side, space='ego', as_mesh=False)
                    vh_egos.append(vh_ego)
                vh_egos = repeat(vh_egos, 't v d -> n t v d', n=D.num_inits).float()
                T_h2c = repeat(T_h2c, 't v d -> n t v d', n=D.num_inits).float()
                t_h2c_untransl = T_h2c[:, :, :3, [3]].transpose(-2, -1)[[0], ...]
                vh_untransl = vh_egos[[0], ...] - t_h2c_untransl
                faces_hand = rearrange(
                    torch.from_numpy(fh.astype(np.int64)),
                    'v d -> 1 v d')

            D.T_h2cs.append(T_h2c)
            D.vhand_ego_list.append(vh_egos)
            D.f_hand_list.append(faces_hand)
            D.vhand_untransl_list.append(vh_untransl)
            D.t_h2c_untransl_list.append(t_h2c_untransl)

        # GT o2w for evaluaion
        D.gt_o2ws = []
        for frames in D.meta_samples.frames_per_seg:
            gt_o2w = reader.pose_obj2world(D.cat, frames)
            D.gt_o2ws.append(gt_o2w)
        D.gt_o2hs = []
        for seg, frames in zip(D.segments, D.meta_samples.frames_per_seg):
            gt_o2h = None
            if seg.ref == INHAND:
                gt_o2h = reader.poses_obj2hand_batch(frames, D.cat, seg.side)
            D.gt_o2hs.append(gt_o2h)

        if self.use_gt_init:
            D.static_inits_o2w_list = [Sim3.from_matrix(gt_o2w) for gt_o2w in D.gt_o2ws]
        
        if self.display_raw_hand:
            D.hands = self.prepare_gt_hands(D)
        else:
            D.hands = None
        D.o3d = self.prepare_o3d_hover_data(D)
        return D
    
    def get_gt_hand_avail_frames(self, cat, seq_name) -> List:
        """ avail_frames: dict(left=[], right=[]) for abs frame indices """
        mask_info_path = osp.join(
            osp.dirname(osp.dirname(MASK_FMT)),
            f"{seq_name}_mask_info.json")
        mask_info = io.read_json(mask_info_path)  # contains: pixels, inview_frames
        handpose_info_path = osp.join(
            HOT3DDATA_ROOT, 'cached_infos', seq_name,
            'hand_pose_available.json')
        handpose_info = io.read_json(handpose_info_path)
        num_pixels = np.asarray(mask_info[cat]['pixels'])
        num_pixel_thr = self.valid_frame_pixel_thr
        inview_frames = (num_pixels > num_pixel_thr).nonzero()[0]
        left_avail_frames = handpose_info['left']['has_pose']
        right_avail_frames = handpose_info['right']['has_pose']
        left_valid_frames = sorted(list(set(left_avail_frames).intersection(inview_frames)))
        right_valid_frames = sorted(list(set(right_avail_frames).intersection(inview_frames)))
        avail_frames = dict(left=left_valid_frames, right=right_valid_frames)
        return avail_frames

    def prepare_valid_frames(self, cat, seq_name, segments):
        """ Given segments, return valid frames for each segment 
        a valid frame mean both hand pose and object mask&pose are available.
        """
        mask_info_path = osp.join(
            osp.dirname(osp.dirname(MASK_FMT)),
            f"{seq_name}_mask_info.json")
        mask_info = io.read_json(mask_info_path)  # contains: pixels, inview_frames
        num_pixels = np.asarray(mask_info[cat]['pixels'])
        num_pixel_thr = self.valid_frame_pixel_thr
        inview_frames = (num_pixels > num_pixel_thr).nonzero()[0]
        avail_frames = self.get_gt_hand_avail_frames(cat, seq_name)
        left_valid_frames = np.asarray(avail_frames['left'])
        right_valid_frames = np.asarray(avail_frames['right'])
        ret_valid_frames = []
        for seg in segments:
            if seg['side'] == 'left':
                valid_frames = left_valid_frames
            elif seg['side'] == 'right':
                valid_frames = right_valid_frames
            else:
                assert seg['ref'] == SCENE_STATIC, "Only static can have side=None"
                # assert seg['ref'] == SCENE_DYNAMIC or seg['ref'] == SCENE_STATIC
                valid_frames = inview_frames
            _valid_frames = valid_frames[
                (seg['st'] <= valid_frames) &
                (valid_frames <= seg['ed'])]
            ret_valid_frames.append(_valid_frames)
        return ret_valid_frames

    """ Open3d Viz """
    def prepare_o3d_hover_data(self, D):
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

        # Build w2c_frames from segments across all segments, not just current D's segments
        if (use_all_segments := True):
            segments = self.tl['segments']
        else:
            segments = D.segments
        valid_frames = self.prepare_valid_frames(D.cat, D.timeline['seq_name'], segments)
        all_potim_segments = [
            PotimSegment(
                st=seg['st'], ed=seg['ed'], ref=seg['ref'], side=seg['side'])
            for seg in segments]
        meta_samples = fixed_segment_sampling(
            all_potim_segments, valid_frames,
            max_samples=self.max_samples_per_seg)
        w2c_frames = meta_samples.nonunique_frames
        all_c2ws = self.reader.T_c2w[w2c_frames]
        data.view_cam = calc_front_viewing_cam(all_c2ws) # view_cam = calc_scene_viewing_cam(c2ws)
        data.all_c2ws = all_c2ws
        data.fov = 60

        # Frustums have to be current D's
        c2ws = self.reader.T_c2w[D.meta_samples.nonunique_frames]
        data.frustums = []
        for c2w in c2ws:
            ego_frustum = get_frustum(c2w)
            data.frustums.append(ego_frustum)
        return data
    
    def prepare_gt_hands(self, D):
        reader = self.reader
        hands = DotMap(fl=reader.fl, fr=reader.fr)  # (1, F, 3)
        logger.debug("Reading GT hands")
        hands.vl_list = []
        hands.vr_list = []
        gt_avail_frames = self.get_gt_hand_avail_frames(D.cat, D.timeline['seq_name'])
        for i, seg in enumerate(D.segments):
            for f in D.meta_samples.frames_per_seg[i]:
                vl, vr = None, None
                for side in ('left', 'right'):
                    if f in gt_avail_frames[side]:
                        # _, vh, _, _ = reader.read_hamer([f], side)
                        vh = reader.hand_verts(f, side, space='ego', as_mesh=False, use_api=True)
                        vh = vh.view(-1, 3)
                        if side == 'left':
                            vl = vh
                        else:
                            vr = vh
                hands.vl_list.append(vl)
                hands.vr_list.append(vr)
        logger.debug("DONE Reading GT hands")
        return hands


class HOT3DOnDemandDataset(Dataset):
    """ On-demand dataset for HOT3D, segment-wise
    The job of this class is to return a HOT3DSingleVideo object,
    via splitting at each timeline.

    example timeline:
    {
        "seq_name": "P0002_016222d1",
        "cat": "spatula_red",
        "total_start": 0,
        "total_end": 3666,
        "timeline_name": "P0002_016222d1_spatula_red",
        "segments": [
            {"st": 0, "ed": 100, "ref": "static", "side": "left"},
            {"st": 100, "ed": 200, "ref": "inhand", "side": "left"},
            ]
    }
    """

    def __init__(self,
                 timeline_json_path: str,
                 use_preextract: bool,
                 **dataset_kwargs):
        super().__init__()
        self.timelines = io.read_json(timeline_json_path)
        self.use_preextract = use_preextract
        self.seq_reader_cache = dict()  # safe to store in memory?

        self.dataset_kwargs = dataset_kwargs
        # Removing categories for which we don't have reduced meshes
        self.timelines = [item for item in self.timelines if item['cat'] not in self.dataset_kwargs['ignore_cats']]

    def __len__(self):
        return len(self.timelines)

    def locate_index_from_output(self, full_key) -> int:
        """ e.g. full_key = P0002_016222d1_spatula_red_00000_03666 """
        for i, info in enumerate(self.timelines):
            fmt = info['timeline_name']
            # fmt = f'{info["seq_name"]}_{info["cat"]}_{info["total_start"]:05d}_{info["total_end"]:05d}'
            if fmt in full_key or full_key in fmt:
                return i
        return None

    def get_seq_reader(self, seq_name):
        if seq_name in self.seq_reader_cache:
            return self.seq_reader_cache[seq_name]
        if self.use_preextract:
            seq_reader = HOT3DPreExtractReader(
                seq_name=seq_name, obj_version='reduced')
        else:
            seq_reader = HOT3DReader(
                seq_name=seq_name, obj_version='reduced')
        self.seq_reader_cache[seq_name] = seq_reader
        return seq_reader
    
    def get_num_segments(self, idx):
        skip_dynamic = self.dataset_kwargs['skip_dynamic']
        _segments = self.timelines[idx]['segments']
        if skip_dynamic:
            _segments = [
                seg for seg in _segments if seg['ref'] != SCENE_DYNAMIC]
        return len(_segments)

    def __getitem__(self, idx):
        tl = self.timelines[idx]
        reader = self.get_seq_reader(tl['seq_name'])

        dataset_object = _HOT3DSingleVideo(
            reader=reader, timeline=tl,
            **self.dataset_kwargs)
        return dataset_object


if __name__ == '__main__':
    import fire
    import tqdm
    from pprint import pprint
    def main(check_all=False, skip_dynamic=True):
        ignore_cats = [
            'aria_small',
            'vase',
            'flask',
            'dumbbell_5lb',
            'spoon_wooden',
            'dvd_remote',
            # Something is wrong with this object, it is not showing up in optimisation
            'holder_black',
        ]                                       # Catogories for which we do not have reduced meshes
        dataset_kwargs = dict(
            max_samples_per_seg=30,
            roi_box_expand=0.4,
            valid_frame_pixel_thr=25,
            occlude_level='all',
            ignore_cats=ignore_cats,
        )
        ds = HOT3DOnDemandDataset(
            timeline_json_path='./code_hot3d/timelines/hot3d_hit.json',
            use_preextract=True,
            skip_dynamic=skip_dynamic,
            **dataset_kwargs)
        if check_all:
            for vds in tqdm.tqdm(ds, total=len(ds)):
                for segi in tqdm.trange(len(vds)):
                    # print(segi)
                    _ = vds[segi]
        else:
            video_ds = ds[0]
            pprint(video_ds[1])
    fire.Fire(main)
