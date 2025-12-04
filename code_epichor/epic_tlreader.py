from typing import List
import glob
import os
import os.path as osp
import numpy as np
import trimesh
import torch
from einops import rearrange, repeat
from PIL import Image
from libzhifan import io
from libzhifan.geometry import CameraManager
from pytorch3d.transforms import rotation_6d_to_matrix
from homan.homan_ManoModel import HomanManoModel
from code_epichor.hamer_loader import HamerLoader
from code_epichor.epic_fields import load_metric_sfm
from datasets.epic_local_reader import EpicLocalReader
from potim.utils.cmd_logger import getLogger


# Images
EPIC_DATA_ROOT = 'DATA_STORAGE/epic'
# Cached frame indices with valid sam-2 masks
MASK_VALID_FRAME_DIR = 'DATA_STORAGE/epic/cache_mask_valid_frames'  # source: code_epichor/scripts/build_timeline.py
# Sam 2 mask
TIMELINE_MASK_DIR = 'DATA_STORAGE/epic/timeline_sam_masks'  # mask source: externally computed
# Cached epic fields in metric-unit and gravity
CACHE_METRIC_EPIC_FIELDS_DIR = 'DATA_STORAGE/epic/cache_metric_epic_fields'   # computed from: code_epichor/epic_fields.py: load_metric_sfm
# Object CAD models
OBJECT_CAD_DIR = 'weights/obj_models/epichor_export_getagrip'
# Object to hand initial poses
O2H_INIT_POSE_DIR = 'weights/pose_priors/epic'


logger = getLogger(__name__)


def read_sam_mask(mp4_name: int, frame: int) -> np.ndarray:
    """
    Args:
        mp4_name: mp4_name from epicgrasps
    
    Returns:
        omask, lmask, rmask. (256, 456)
    """
    obj_mask_fmt = osp.join(
        TIMELINE_MASK_DIR, 'obj',
        '%s/%012d.png'
    )
    lh_mask_fmt = osp.join(
        TIMELINE_MASK_DIR, 'left-hand',
        '%s/%012d.png'
    )
    rh_mask_fmt = osp.join(
        TIMELINE_MASK_DIR, 'right-hand',
        '%s/%012d.png'
    )
    o_mask = Image.open(obj_mask_fmt % (mp4_name, frame))
    l_mask = None
    r_mask = None
    l_mask_path = lh_mask_fmt % (mp4_name, frame)
    r_mask_path = rh_mask_fmt % (mp4_name, frame)
    if osp.exists(l_mask_path):
        l_mask = Image.open(l_mask_path)
    if osp.exists(r_mask_path):
        r_mask = Image.open(r_mask_path)
    return o_mask, l_mask, r_mask


class EPICTimelineReader:
    """ This reads everything in a timeline-level:

    - Masks. including occlusion-aware stuff.
    - EPIC-Fields
    - Hamer
    - Hand-object bounding boxes. [Q: use cache or from mask? but cache is from mask too]
    - Object Meshes for one category
    - Object-to-hand initial poses
    - Camera Manager
    """
    IMG_SIZE = (854, 480)

    def __init__(self, tl_name: str, mp4_name):
        """
        Args:
            tl_name: e.g. P35_105_left_pan_71777_72870
            mp4_name: from epicgrasps. P01_01_21784_22242_22872_left_pan
        """
        _vid0, _vid1, self.side, self.cat, self.total_start, self.total_end = \
            tl_name.split('_')
        vid = '_'.join([_vid0, _vid1])
        self.vid = vid
        self.tl_name = tl_name
        self.mp4_name = mp4_name  # mainly for mask reading
        self.mask_avail_frames = io.read_json(
            f'{MASK_VALID_FRAME_DIR}/{self.mp4_name}.json')
        self.img_reader = EpicLocalReader(EPIC_DATA_ROOT)
        self.total_start = int(self.total_start)
        self.total_end = int(self.total_end)

        logger.debug("Loading SFM...")
        self.sfm = self._load_metric_sfm(vid)
        logger.debug("Loading SFM Done.")
        self.sfm_avail_frames = self._calc_sfm_avail_frames()
        obj_path = f'{OBJECT_CAD_DIR}/{self.cat}.obj'
        obj_mesh = trimesh.load(obj_path, force='mesh')
        self.obj_verts = obj_mesh.vertices
        self.obj_faces = obj_mesh.faces

        epicfields_fx = self.sfm['camera']['params'][0]
        epicfields_fy = self.sfm['camera']['params'][1]
        out_w, out_h = self.IMG_SIZE
        fx = epicfields_fx / self.sfm['camera']['width'] * out_w
        fy = epicfields_fy / self.sfm['camera']['height'] * out_h
        self.global_cam = CameraManager(
            fx=fx, fy=fy, cx=out_w//2, cy=out_h//2,
            img_h=out_h, img_w=out_w)

        self.hamer_loader = HamerLoader(ho_version='hoa_potim', device='cpu')
        self.lmano_homan = HomanManoModel(
            "externals/mano", side='left', pca_comps=45)  # Note(zhifan): in HOMAN num_pca = 16
        self.rmano_homan = HomanManoModel(
            "externals/mano", side='right', pca_comps=45)
        self.fl = self.lmano_homan.hand_faces
        self.fr = self.rmano_homan.hand_faces
    
    def _load_metric_sfm(self, vid: str):
        cache_epic_fields_path = f'{CACHE_METRIC_EPIC_FIELDS_DIR}/{vid}.pkl'
        if osp.exists(cache_epic_fields_path):
            return io.read_pickle(cache_epic_fields_path)
        os.makedirs(osp.dirname(cache_epic_fields_path), exist_ok=True)
        logger.info(f"Caching metric epic fields for {vid}")
        sfm = load_metric_sfm(vid)
        io.write_pickle(sfm, cache_epic_fields_path)
        return sfm
    
    def read_image(self, frame_ind: int) -> np.ndarray:
        """ Read image from the video

        Args:
            frame_ind (int): frame index

        Returns:
            np.ndarray: (854, 480, 3)
        """
        return self.img_reader.read_image(self.vid, frame_ind)
        # image_format = f'<path-to-epic-100>/rgb/%s/%s/frame_%010d.jpg'
        # image_path = image_format % (self.vid[:3], self.vid, frame_ind)
        # if not osp.exists(image_path):
        #     return image_path
        # img_pil = Image.open(image_path).resize(self.IMG_SIZE)
        # img = np.asarray(img_pil)
        # return img
    
    def load_fg_masks(self, 
                      frames: List[int], 
                      ignore_other_objects=False) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: (T, H, W) int32
        """
        masks = [self.read_mask(frame, ignore_other_objects)
                 for frame in frames]
        masks = rearrange(masks, 't h w -> t h w')
        return masks

    def read_mask(self, frame_ind: int, ignore_other_objects=False) -> torch.Tensor:
        """

        Returns:
            mask_obj: torch.Tensor (H, W) int32
                1 fg, -1 ignore, 0 bg
        """
        W, H = self.IMG_SIZE
        om, lm, rm = read_sam_mask(self.mp4_name, frame_ind)
        om = om.resize((W, H))
        om = torch.from_numpy(np.asarray(om))
        obj_mask = torch.zeros([*om.shape[:2]])
        if lm is not None:
            lm = lm.resize((W, H))
            lm = torch.from_numpy(np.asarray(lm))
            obj_mask[lm == 1] = -1
        if rm is not None:
            rm = rm.resize((W, H))
            rm = torch.from_numpy(np.asarray(rm))
            obj_mask[rm == 1] = -1
        obj_mask[om == 1] = 1

        return obj_mask
    
    def read_w2c(self, frames: List[int]) -> torch.Tensor:
        """
        Args:
            frames: List of frame indices
        
        Returns:
            torch.Tensor: (T, 4, 4)
        """
        w2cs = []
        for f in list(frames):
            w2c = self.sfm['images'][int(f)]
            w2cs.append(torch.from_numpy(w2c).float())
        w2c = rearrange(w2cs, 't d e -> t d e')
        return w2c
    
    def _read_verts_hand_og(self, frames: List[int], side: str) -> torch.Tensor:
        """
        Args:
            frames: List of frame indices
            side: 'left' or 'right'
        
        Returns:
            torch.Tensor: (T, V, 3)
            params
        """
        frames = list(frames)
        num_samples = len(frames)
        assert side in ['left', 'right']
        mano_model = self.lmano_homan if side == 'left' else self.rmano_homan
        params = self.hamer_loader.get_hamer_parmas(
            self.global_cam, self.vid, frames, 
            is_left=(side == 'left'), ortho2persp=True)
        
        mano_rot = torch.zeros([num_samples, 3]) # , device=mano_pca_pose.device)
        mano_trans = torch.zeros([num_samples, 3]) # , device=mano_pca_pose.device)
        mano_res = mano_model.forward_pca(
            params.pca_pose,
            rot=mano_rot,
            betas=params.betas,
            side=side)
        vertices = mano_res["verts"]
        verts_hand_og = vertices + mano_trans.view(-1, 1, 3)

        return verts_hand_og, params

    def read_hamer(self, frames: List[int], side, scale_hand: float = None):
        """
        Returns:
            T_h2c: torch.Tensor (T, 4, 4)
            vh_egos: torch.Tensor (T, V, 3)
            vh_untranslated: torch.Tensor (T, V, 3)
            t_h2c: torch.Tensor (T, 3) untranslated
        """
        verts_hand_og, params = self._read_verts_hand_og(frames, side)

        num_samples = len(frames)
        R_h2c = rotation_6d_to_matrix(params.rot6d)
        t_h2c = params.transl
        verts_hand_ego = verts_hand_og @ R_h2c.permute(0, 2, 1) + t_h2c
        if scale_hand is not None:
            t_h2c = t_h2c * scale_hand
            verts_hand_ego = (verts_hand_og * scale_hand) @ R_h2c.permute(0, 2, 1) + t_h2c
        verts_hand_untranslated = verts_hand_og @ R_h2c.permute(0, 2, 1)
        t_h2c_untransl = params.transl

        T_h2c = repeat(torch.eye(4), 'd e -> t d e', t=num_samples).clone()
        T_h2c[:, :3, :3] = R_h2c
        T_h2c[:, :3, 3] = t_h2c.view(-1, 3)

        return T_h2c, verts_hand_ego, verts_hand_untranslated, t_h2c_untransl

    def read_hamer_poly_scales(self, frames: List[int], side, scale_hand: torch.Tensor):
        """ Read hamer when scale_hand is a list of scales.

        Args:
            scale_hand: (N,)

        Returns:
            T_h2c: torch.Tensor (N, T, 4, 4)
            vh_egos: torch.Tensor (N, T, V, 3)
            vh_untranslated: torch.Tensor (1, T, V, 3)
            t_h2c_untransl: torch.Tensor (1, T, 3) untranslated
        """
        verts_hand_og, params = self._read_verts_hand_og(frames, side)

        num_samples = len(verts_hand_og)  # len(frames)
        num_inits = len(scale_hand)
        verts_hand_og = rearrange(verts_hand_og, 't v d -> 1 t v d')
        R_h2c = rearrange(
            rotation_6d_to_matrix(params.rot6d), 't d e -> 1 t d e')
        t_h2c = rearrange(params.transl, 't d e -> 1 t d e')
        scale_hand = rearrange(scale_hand, 'n -> n 1 1 1')

        t_h2c = t_h2c * scale_hand
        verts_hand_ego = (verts_hand_og * scale_hand) @ R_h2c.transpose(-2, -1) + t_h2c  # (N, T, V, 3)

        verts_hand_untranslated = verts_hand_og @ R_h2c.transpose(-2, -1)
        t_h2c_untransl = rearrange(params.transl, 't d e -> 1 t d e')

        T_h2c = repeat(torch.eye(4), 'd e -> n t d e', n=num_inits, t=num_samples).clone()
        T_h2c[..., :3, :3] = R_h2c
        T_h2c[..., :3, [3]] = t_h2c.transpose(-2, -1)

        return T_h2c, verts_hand_ego, verts_hand_untranslated, t_h2c_untransl
    
    def o2h_init_poses(self, side: str, cat: str) -> torch.Tensor:
        """
        Returns:
            np.ndarray: (N, 4, 4)
        """
        assert side in ['left', 'right']

        glob_list = glob.glob(
            f'{O2H_INIT_POSE_DIR}/{cat}/{side}_*.npy')
        poses = [np.load(v) for v in glob_list]
        poses = torch.from_numpy(np.stack(poses))

        rot_o2h = poses[:, :3, :3]
        transl_o2h = poses[:, :3, 3]

        num_inits = len(rot_o2h)
        rot_o2h = rot_o2h.float()  # (N, 3, 3)
        transl_o2h = transl_o2h.view(-1, 3).float()  # (N, 3)

        T_o2h = torch.zeros(num_inits, 4, 4)
        T_o2h[:, :3, :3] = rot_o2h
        T_o2h[:, :3, 3] = transl_o2h
        T_o2h[:, 3, 3] = 1
        return T_o2h
    
    def _calc_sfm_avail_frames(self) -> List:
        """ Returns frames that are available in the sfm.
        and within the total_start and total_end.

        Returns:
            torch.Tensor: (T,)
        """
        frames = list(sorted(self.sfm['images'].keys()))
        sfm_avail_frames = torch.tensor(frames)
        sfm_avail_frames = sfm_avail_frames[
            (sfm_avail_frames >= self.total_start) & (sfm_avail_frames <= self.total_end)]
        if len(sfm_avail_frames) < (self.total_end - self.total_start + 1):
            _total_frames = self.total_end - self.total_start + 1
            logger.info(f"Warning: {len(sfm_avail_frames)=} < {_total_frames=}")
        return sfm_avail_frames.tolist()
    
    """ Debug utils """
    def get_sfm_pcd_open3d(self, as_mesh, voxel_down_sample=-1):
        import open3d as o3d
        points = self.sfm['points']
        verts = np.asarray([p[:3] for p in points])
        colors = np.asarray([p[3:] for p in points])
        if as_mesh:
            pcd_mesh = o3d.geometry.TriangleMesh()
            pcd_mesh.vertices = o3d.utility.Vector3dVector(verts)
            pcd_mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.)
        else:
            pcd_mesh = o3d.geometry.PointCloud()
            pcd_mesh.points = o3d.utility.Vector3dVector(verts)
            pcd_mesh.colors = o3d.utility.Vector3dVector(colors / 255.)
            if voxel_down_sample > 0:
                pcd_mesh = pcd_mesh.voxel_down_sample(voxel_size=voxel_down_sample)
        return pcd_mesh
    
    def get_hamer_avail_frames(self) -> List:
        """ avail_frames: dict(left=[], right=[]) for abs frame indices """
        sfm_avail_frames = set(self.sfm_avail_frames)
        avail_frames = self.hamer_loader.avail_side_frames(self.vid)
        for side in ['left', 'right']:
            avail_frames[side] = list(set(avail_frames[side]).intersection(sfm_avail_frames))
        return avail_frames

    def get_valid_frames(self) -> torch.Tensor:
        sfm_avail_frames = set(self.sfm_avail_frames)
        mask_avail_frames = set(self.mask_avail_frames)
        avail_frames = list(sfm_avail_frames & mask_avail_frames)
        return sorted(avail_frames)


if __name__ == '__main__':
    pass