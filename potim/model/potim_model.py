from functools import reduce
import numpy as np
import cv2
from collections import namedtuple
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import neural_renderer as nr
from open3d.visualization import rendering

from homan.metrics import batch_mask_iou
from potim.defs.types import (
    PotimSegment, ScaleGetter,
    INHAND, SCENE_STATIC, SCENE_DYNAMIC)
from potim.model.scene_static import SceneStatic
from potim.model.proxy_scene_static import ProxySceneStatic
from potim.model.inhand_dynamic import InHandDynamicVis
from potim.model.inhand import InHand, InHandImpl, InHandVis
from potim.defs.sim3 import Sim3
from potim.utils.open3d.helper import (
    get_global_o3d_render, inc_render_usage_counter,
    update_render_geom, update_render_geom_from_trimesh)

from libzhifan.geometry import (
    BatchCameraManager, CameraManager, SimpleMesh,
    projection)


""" There are two options for initilisation.
1. Out of N segments, we say one of them is stronger than others,
    hence number of total inits we need to test == num_inits of that segment.
2. If we treat all segments equally strong,
    then we need to test sum of all N segments.
    so maybe implement num_potim_inits = calculate_potim_inits(segs),

In both cases, we should implement subset initialisation, and both
    requires ref_seg_idx as an argument.
"""

class POTIM_SC(nn.Module):
    """
    POTIM_SC: Posed Object TIMeline model with Segments Constracints
    """

    def __init__(self,
                 meta_samples,
                 ndof: str,
                 num_inits: int,
                 update_scale: bool,
                 rend_size=256,
                 use_proxy_static=False):
        super().__init__()
        """ each segment are closed both left and right. """
        self.nonunique_frames = meta_samples.nonunique_frames
        self.frames_per_seg = meta_samples.frames_per_seg
        self.segments = meta_samples.segments
        self.num_samples = meta_samples.num_samples
        self.num_segments = len(self.segments)

        # concat seg inds
        self._sampled_abs_inds = torch.tensor(
            reduce(lambda a, b: a + b,
             [list(range(seg.st, seg.ed+1)) for seg in self.segments],
             []))
        self._abs_ind_to_segi = reduce(lambda a, b: a + b,
            [[segi for _ in range(seg.st, seg.ed+1)] for segi, seg in enumerate(self.segments)],
            [])  # used to get ref for any abs_ind
        self._abs_ind_to_data_idx = {
            int(abs_ind): i for i, abs_ind in enumerate(self._sampled_abs_inds)
        }

        self.num_inits = num_inits
        self.rend_size = rend_size
        self.update_scale = update_scale

        self.scale_obj_getter = ScaleGetter(
            torch.empty([self.num_inits, 1, 1]), 
            update_scale=self.update_scale)
        self.use_proxy_static = use_proxy_static
        if not use_proxy_static:
            self.M = self.create_modules(
                self.segments, ndof=ndof)
        else:
            self.M = self.create_proxy_modules(
                self.segments)

        # Setup renderer
        self.renderer = nr.renderer.Renderer(
            image_size=self.rend_size,
            K=None,
            R=torch.eye(3)[None],
            t=torch.zeros([1, 3]),
            dist_coeffs=torch.tensor([[0., 0., 0., 0., 0.]]),
            orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = torch.as_tensor(0.3)
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]

        self.set_video_frames('keyframe')

    def to(self, device):
        super().to(device)
        self.renderer.R = self.renderer.R.to(device)
        self.renderer.t = self.renderer.t.to(device)
        self.renderer.dist_coeffs = self.renderer.dist_coeffs.to(device)
        self.scale_obj_getter.to(device)
        self.M.to(device)
        self.device = device

    def create_modules(self,
                       segments: List[PotimSegment],
                       ndof: str) -> List[Union[InHandVis, InHandDynamicVis, SceneStatic]]:
        # Setup submodules, according to input segments
        modules = []
        for seg in segments:
            if seg.ref == INHAND:
                mod = InHandVis(
                    ndof=ndof,
                    num_inits=self.num_inits,
                    num_samples=seg.ed-seg.st+1,
                    scale_obj_getter=self.scale_obj_getter,
                    rend_size=self.rend_size,
                    seg_st=seg.st,
                    seg_ed=seg.ed,
                    ref=INHAND,
                    )
            elif seg.ref == SCENE_STATIC:
                mod = SceneStatic(
                    num_inits=self.num_inits,
                    num_samples=seg.ed-seg.st+1,
                    seg_st=seg.st,
                    seg_ed=seg.ed,
                    ref=SCENE_STATIC,
                )
            elif seg.ref == SCENE_DYNAMIC:
                mod = InHandDynamicVis(
                    ndof=ndof,
                    num_inits=self.num_inits,
                    num_samples=seg.ed-seg.st+1,
                    scale_obj_getter=self.scale_obj_getter,
                    rend_size=self.rend_size,
                    seg_st=seg.st,
                    seg_ed=seg.ed,
                    ref=SCENE_DYNAMIC,
                    )
            else:
                raise ValueError(f"Unkown ref {seg.ref}")
            modules.append(mod)
        return nn.ModuleList(modules)
    
    def create_proxy_modules(self,    
                             segments) -> List[Union[InHandVis, ProxySceneStatic]]:
        # Setup submodules, according to input segments
        assert len(segments) == 2
        assert set([seg.ref for seg in segments]) == {INHAND, SCENE_STATIC}

        inhand_segi = 0 if segments[0].ref == INHAND else 1
        seg = segments[inhand_segi]
        inhand_module = InHandVis(
            ndof='N6',
            num_inits=self.num_inits,
            num_samples=seg.ed-seg.st+1,
            scale_obj_getter=self.scale_obj_getter,
            rend_size=self.rend_size,
            seg_st=seg.st,
            seg_ed=seg.ed,
            ref=INHAND)
        
        static_segi = 1 - inhand_segi
        seg = segments[static_segi]
        # if segi of A is 0, then we have A-C, then proxy-A will take first(0) pose of C
        inhand_index = 0 if static_segi == 0 else -1
        proxy_static_module = ProxySceneStatic(
            inhand=inhand_module,
            inhand_index=inhand_index,
            num_inits=self.num_inits,
            num_samples=seg.ed-seg.st+1,
            seg_st=seg.st,
            seg_ed=seg.ed,
            ref=SCENE_STATIC)
        
        if inhand_segi == 0:
            modules = [inhand_module, proxy_static_module]
        else:
            modules = [proxy_static_module, inhand_module]
        return nn.ModuleList(modules)

    def get_local_inds(self, abs_inds, seg_idx):
        """
        Args:
            abs_inds: (T,) absolute index wrt the entire sequence.
        Returns:
            local_inds: (T,) local index wrt a child segment.
        """
        seg = self.segments[seg_idx]
        abs_st, abs_ed = seg.st, seg.ed
        local_inds = abs_inds[
            (abs_st <= abs_inds) & (abs_inds <= abs_ed)] - abs_st
        return local_inds

    def get_seg_layout(self, seg_idx):
        """ return absolute indices that are covered by this segment """
        return self.segments[seg_idx]

    def register_camera(self, camintr: torch.Tensor):
        """
        camintr: (1, T, 3, 3)
        """
        camintr = camintr.detach().clone().requires_grad_(False)
        camintr = camintr.view(1, self.num_samples, 3, 3)
        self.register_buffer('camintr', camintr)
        for i, mod in enumerate(self.M):
            if isinstance(mod, InHand):
                seg = self.segments[i]
                inds = torch.arange(seg.st, seg.ed+1)
                mod.register_camera(camintr[:, inds, ...])

    def register_obj_buffer(self, verts_object_og, faces_object):
        """
        verts_object_og: (1, 1, V, 3)
        """
        N, T = self.num_inits, self.num_samples
        verts_object_og = verts_object_og.view(1, 1, -1, 3)
        self.register_buffer(
            "verts_object_og", verts_object_og)
        """ Do not attempt to copy tensor too early, which will get OOM. """
        self.register_buffer(
            "faces_object", faces_object)
        self.register_buffer(
            "textures_object",
            torch.ones(faces_object.shape[0], 1, 1, 1, 3))
        for mod in self.M:
            if isinstance(mod, InHand):
                mod.register_obj_buffer(
                    verts_object_og, faces_object)

    def register_obj_target(self, target_masks_object: torch.Tensor):
        """
        target_masks_object: (1, num_samples, W, W)
        """
        target_masks_object = target_masks_object.view(
            1, self.num_samples, self.rend_size, self.rend_size)
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        for mod in self.M:
            if isinstance(mod, InHand):
                mod.register_obj_target(target_masks_object)

    def register_w2cs(self, w2cs: List[torch.Tensor]):
        for i, mod in enumerate(self.M):
            mod.register_w2c(w2cs[i])

    def _index_obj_target(self, inds):
        """
        returns: (N, t, W, W)
        """
        ref_mask = self.ref_mask_object.expand(self.num_inits, -1, -1, -1)
        keep_mask = self.keep_mask_object.expand(self.num_inits, -1, -1, -1)
        ref_mask, keep_mask = map(
            lambda x: x[:, inds, ...].view(self.num_inits, -1, self.rend_size, self.rend_size),
            (ref_mask, keep_mask))
        return ref_mask, keep_mask

    def _expand_obj_faces(self, n, t) -> torch.Tensor:
        """
        e.g. self._expand_obj_faces(n, t) -> shape (n*t, f, 3)
        Args:

        Returns:
            obj_faces: (n * t, F_o, 3)
        """
        num_obj_faces = self.faces_object.size(0)
        return self.faces_object.expand(n*t, num_obj_faces, 3)

    def get_verts_object(self, inds) -> torch.Tensor:
        """
        Args:
            inds: (T,)
        Returns:
            (N, T, V, 3)
        """
        n, t = self.num_inits, len(inds)
        if max(inds) >= self.num_samples:
            raise ValueError(f"{inds} Out of Bound: self.num_samples = {self.num_samples}")
        verts_obj = self.verts_object_og.expand(n, t, -1, -1)
        rots, transl, _ = self.get_obj_transform_ego(inds)
        transl = transl.view(n, t, 1, 3)
        scale_obj = self.scale_obj_getter()
        scale = scale_obj.tile(1, t, 1).view(n, t, 1, -1)
        return (verts_obj*scale) @ rots.permute(0, 1, 3, 2) + transl

    def init_hand(self, D, Np_start, Np_end):
        segs = self.segments
        for i, mod in enumerate(self.M):
            if (
                mod.ref == INHAND 
                or 
                (mod.ref == SCENE_DYNAMIC and segs[i].side is not None)
            ):
                vhand_ego = D.vhand_ego_list[i][Np_start:Np_end, ...]
                T_h2cs = D.T_h2cs[i][Np_start:Np_end, ...]
                mod.init_hand(vhand_ego, T_h2cs, D.f_hand_list[i])

    def init_scale_obj(self, scale_obj: torch.Tensor, Np_start, Np_end):
        """ scale_obj: (num_inits,) """
        self.scale_obj_getter.scale.data = scale_obj[Np_start:Np_end, ...].view(self.num_inits, 1, 1)

    def init_dynamic_obj_transform(self, dyn_inits_o2c_list):
        raise NotImplementedError

    def init_static_obj_transform(self, static_inits_o2w_list):
        for i, mod in enumerate(self.M):
            if mod.ref == SCENE_STATIC:
                mod.init_obj_transform(static_inits_o2w_list[i], in_coord='o2w')

    def get_obj_transform_ego(self, abs_inds) -> Sim3:
        """
        Args:
            abs_inds: indices into potim sample layout.
            see desc.md for more details.

        Returns:
            rot: (N, T, 3, 3)
            transl: (N, T, 3)
            scale: (N, 1)
        """
        rot_mats = []
        transl = []
        for i, mod in enumerate(self.M):
            seg_inds = self.get_local_inds(abs_inds, i)
            if len(seg_inds) == 0:
                continue
            # abs_frames = self.frames_per_seg[i][seg_inds]
            # print(f"{i=}, {mod.ref=}, {seg_inds=}, {abs_frames=}, {mod.seg_st=}, {mod.seg_ed=}")
            T = mod.get_obj_transform_ego(seg_inds)
            rot_mats.append(T.rot)
            transl.append(T.t)
        rot_mat_obj = torch.cat(rot_mats, dim=1)
        transl = torch.cat(transl, dim=1)
        scale_obj = self.scale_obj_getter()
        return Sim3(rot_mat_obj, transl, scale_obj)

    def get_obj_transform_world(self, abs_inds) -> Sim3:
        """
        Args:
            abs_inds: indices into potim sample layout.
            see desc.md for more details.

        Returns:
            rot: (N, T, 3, 3)
            transl: (N, T, 3)
            scale: (N, 1)
        """
        rot_mats = []
        transl = []
        scale = None
        for i, mod in enumerate(self.M):
            seg_inds = self.get_local_inds(abs_inds, i)
            if len(seg_inds) == 0:
                continue
            T = mod.get_obj_transform_world(seg_inds)
            rot_mats.append(T.rot)
            transl.append(T.t)
            if scale is None:
                scale = T.s
        rot_mat_obj = torch.cat(rot_mats, dim=1)
        transl = torch.cat(transl, dim=1)
        return Sim3(rot_mat_obj, transl, scale)
    
    def get_w2cs(self, abs_inds) -> torch.Tensor:
        """ returns: (1, T, 4, 4) """
        w2cs = []
        for i, mod in enumerate(self.M):
            seg_inds = self.get_local_inds(abs_inds, i)
            if len(seg_inds) == 0:
                continue
            T = mod.get_w2cs(seg_inds)
            w2cs.append(T)
        w2cs = torch.cat(w2cs, dim=1)
        return w2cs

    """ Losses """
    def render_obj(self, inds):
        """
        Args:
            inds: (T,)
        Returns:
            images: (N, T, W, W)
        """
        N, t = self.num_inits, len(inds)
        batch_faces = self._expand_obj_faces(
            self.num_inits, t)  # (N*T, F, 3)
        batch_K = self.camintr[:, inds, :, :].expand(N, t, 3, 3)

        verts = self.get_verts_object(inds)
        # if check:= True == True:
        #     if verts.isnan().any():
        #         import pudb;
        #         pudb.set_trace()
            # print(verts.shape, batch_faces.shape, batch_K.shape)
        images = self.renderer(
            verts.view(N*t, -1, 3), batch_faces, K=batch_K.reshape(N*t, 3, 3), mode='silhouettes')
        images = images.view(N, t, self.rend_size, self.rend_size)
        return images

    def forward_obj_pose_render(self,
                                inds,
                                loss_only=True,
                                clamp_max=100.0):
        """
        Args:
            inds: (T,)

        Returns:
            l_mask: (N, T)
            [Optional] iou: (N, T)
        """
        if not isinstance(inds, torch.Tensor):
            inds = torch.tensor(inds)
        images = self.render_obj(inds)
        image_ref, keep = self._index_obj_target(inds)
        image = keep * images

        loss_mask = torch.sum((image - image_ref)**2, dim=(-2, -1))
        loss_mask = loss_mask / (image_ref.sum(dim=(-2, -1)) + 1e-7)  # Fix: 2024/Feb/26, remove the normaliser 'n', so we align with physical loss that we always sum, not mean!
        loss_mask = loss_mask.clamp_max_(clamp_max)

        if not loss_only:
            with torch.no_grad():
                iou = batch_mask_iou(
                    image.detach(), image_ref.detach())  # (N, T)
            return loss_mask, iou, image
        return loss_mask

    def train_loss(self, inds, cfg, debug_check_nan=True):
        """
        Args:
            inds: (T,)
            cfg: optim_cfg. OmegaConf
        Returns:
            loss: scalar if ret_dict else dict of (N, T)
        """
        _C = cfg.loss
        retVal = namedtuple('TrainLoss', 'loss_dict loss metrics_dict')
        l_dict = dict()
        metrics = dict()

        l_mask, iou, _ = self.forward_obj_pose_render(inds, loss_only=False)
        l_dict['mask'] = l_mask * cfg.loss.mask.weight
        metrics['oiou'] = iou.mean(dim=-1)  # (N,)

        for i, mod in enumerate(self.M):
            seg_inds = self.get_local_inds(inds, i)
            if len(seg_inds) == 0:
                continue
            if mod.ref == INHAND:
                l_phy_dict = mod.train_loss(seg_inds, cfg, debug_check_nan)  # physical
                self.update_l_dict(l_dict, l_phy_dict.loss_dict, i)
            elif mod.ref == SCENE_DYNAMIC:
                l_phy_dict = mod.train_loss_dynamic(seg_inds, cfg, debug_check_nan)  # physical
                self.update_l_dict(l_dict, l_phy_dict.loss_dict, i)

        if _C.connection.weight > 0:
            l_dict['connection'] = _C.connection.weight * self.loss_connection()

        return retVal(loss_dict=l_dict, loss=None, metrics_dict=metrics)

    def loss_connection(self):
        """ Constraint connection in world vertex frame

        Alternatively, I can implemented using potim.get_obj_transform_world()

        Returns:
            l_connection: (N, 1) avg over vertices
        """
        vo_orig = self.verts_object_og.view(-1, 3)  # (1, 1, V, 3) -> (V, 3)
        
        phy_factor = (self.camintr[:, :, 0, 0]).mean(dim=1) * self.rend_size # (N, 1)

        loss = torch.zeros([self.num_inits, 1], device=self.device)
        pairs = [(i, i+1) for i in range(len(self.segments)-1)]
        for i, j in pairs:
            T_prev = self.M[i].get_obj_transform_world([-1])  # (N, 1, 4, 4)
            T_next = self.M[j].get_obj_transform_world([0])
            vo_prev = T_prev.apply(vo_orig)  # (N, 1, V, 3)
            vo_next = T_next.apply(vo_orig)
            l = torch.mean((vo_prev - vo_next)**2, dim=(-1, -2, -3))
            loss = loss + l * phy_factor
        return loss

    def get_best_pose_idx(self):
        return None
    
    @staticmethod
    def update_l_dict(d_all, d_new, segi: int):
        for k, v in d_new.items():
            k = f"{segi}.{k}"
            d_all[k] = v
        return d_all


    """ Visualisation """
    def set_video_frames(self, mode: str):
        """ mode: one of {'all', 'keyframe'}"""
        if mode == 'all':
            self._out_video_inds = self._sampled_abs_inds
            self._make_video_fps = 10
        elif mode == 'keyframe':
            # sample 3 frames each segment
            key_frames = []
            for seg in self.segments:
                _frames = np.linspace(seg.st, seg.ed, num=3, endpoint=True, dtype=int)
                key_frames.extend(_frames)
            self._out_video_inds = key_frames
            self._make_video_fps = 2
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def set_display_segi(self, segi: int):
        self._display_segi = segi

    def get_meshes(self, pose_idx, sample_idx, **mesh_kwargs) -> Tuple[SimpleMesh]:
        """
        Args:
            pose_idx:
            sample_idx: index of sample (timestep)

        Returns:
            mleft: SimpleMesh
            mright: SimpleMesh
            mobj: SimpleMesh, or None if obj_idx < 0
        """
        hand_color = mesh_kwargs.pop('hand_color', 'light_blue')
        obj_color = mesh_kwargs.pop('obj_color', 'yellow')

        mleft, mright = None, None
        for i, seg in enumerate(self.segments):
            if (seg.ref == INHAND and seg.st <= sample_idx <= seg.ed) or \
                (seg.ref == SCENE_DYNAMIC and seg.st <= sample_idx <= seg.ed):
                mhand = SimpleMesh(
                    self.M[i].v_hand[0, sample_idx - seg.st, :, :],
                    self.M[i].faces_hand, tex_color=hand_color)
                if seg.side == 'left':
                    mleft = mhand
                else:
                    mright = mhand

        with torch.no_grad():
            verts_obj = self.get_verts_object(
                torch.tensor([sample_idx]))[pose_idx, 0, :, :]
            mobj = SimpleMesh(
                verts_obj, self.faces_object, tex_color=obj_color)

        return mleft, mright, mobj

    def render_scene(self, pose_idx, scene_idx,
                     with_hand=True, overlay_gt=False,
                     bg_image_patch=True,
                     **mesh_kwargs) -> np.ndarray:
        """ 
        pose_idx: init_ind
        scene_idx: frame_ind
        returns: (H, W, 3) """
        # nt_ind = pose_idx * self.train_size + scene_idx
        nt_ind = scene_idx
        n, t = pose_idx, scene_idx
        if not with_hand:
            img = self.image_patches[nt_ind] / 255
        else:
            mleft, mright, mobj = self.get_meshes(pose_idx=pose_idx,
                                          sample_idx=scene_idx, **mesh_kwargs)
            if pose_idx < 0:
                mesh_list = [mleft, mright]
            else:
                mesh_list = [mleft, mright, mobj]
            bg_image = self.image_patches[nt_ind] if bg_image_patch else None
            vis_rend_size = self.rend_size
            img = projection.perspective_projection_by_camera(
                mesh_list,
                # CameraManager.from_nr(
                #     self.camintr.detach().cpu().numpy()[nt_ind], vis_rend_size),
                CameraManager.from_nr(
                    self.camintr.detach().cpu().numpy()[n, t, ...], vis_rend_size),
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                ),
                image=bg_image,
            )

        if overlay_gt:
            all_mask = np.zeros_like(img, dtype=np.float32)
            mask_hand = self.ref_mask_hand[nt_ind].cpu().numpy().squeeze()
            all_mask = np.where(
                mask_hand[..., None], (0, 0, 0.8), all_mask)
            # if obj_idx >= 0:
            mask_obj = self.ref_mask_object[nt_ind].cpu().numpy()
            all_mask = np.where(
                mask_obj[..., None], (0.6, 0, 0), all_mask)
            all_mask = np.uint8(255*all_mask)
            img = cv2.addWeighted(np.uint8(img*255), 0.9, all_mask, 0.5, 1.0)
        return img

    def render_grid_np(self, pose_idx=0, with_hand=True,
                       *args, **kwargs) -> np.ndarray:
        """ low resolution but faster """
        num_grids = min(30, len(self.image_patches))
        cam_inds = np.linspace(
            0, len(self.image_patches), num_grids, endpoint=False, dtype=np.int)
        num_cols = min(num_grids, 5)
        num_rows = (num_grids + num_cols - 1) // num_cols
        imgs = []
        for cam_idx in cam_inds:
            img = self.render_scene(
                pose_idx=pose_idx,
                scene_idx=cam_idx,
                with_hand=with_hand, *args, **kwargs)
            imgs.append(img)
            if cam_idx == num_grids-1:
                break

        h, w, _ = imgs[0].shape
        out = np.empty(shape=(num_rows*h, num_cols*w, 3), dtype=imgs[0].dtype)
        for row in range(num_rows):
            for col in range(num_cols):
                idx = row*num_cols+col
                if idx >= num_grids:
                    break
                out[row*h:(row+1)*h, col*w:(col+1)*w, :] = imgs[idx]

        return out

    def render_triview(self, pose_idx, scene_idx,
                       views=['front', 'left', 'back'],
                       hstack=True,
                       rend_size=256,
                       **mesh_kwargs) -> np.ndarray:
        """
        Returns:
            (H, W, 3)
        """
        mleft, mright, mobj = self.get_meshes(
            pose_idx=pose_idx, sample_idx=scene_idx, **mesh_kwargs)
        meshes = [mleft, mright, mobj]

        direction_map = {'front': '+z', 'left': '+x', 'back': '-z', 'right': '-x'}
        rends = []
        for view in views:
            direction = direction_map[view]
            rend = projection.project_standardized(
                meshes,
                direction=direction,
                image_size=rend_size,
                method=dict(
                    name='pytorch3d',
                    coor_sys='nr',
                    in_ndc=False
                )
            )
            rends.append(rend)
        rends = np.hstack(rends) if hstack else np.vstack(rends)
        return rends

    def render_global(self,
                      global_cam: BatchCameraManager,
                      global_images: np.ndarray,
                      pose_idx: int,
                      sample_idx: int,
                      **mesh_kwargs,
                      ) -> np.ndarray:
        """ returns: (H, W, 3) """
        mleft, mright, mobj = self.get_meshes(
            pose_idx=pose_idx,
            sample_idx=sample_idx, **mesh_kwargs)
        global_image = global_images[self._abs_ind_to_data_idx[sample_idx]]
        img = projection.perspective_projection_by_camera(
            [mleft, mright, mobj],
            global_cam[sample_idx],
            method=dict(
                name='pytorch3d',
                coor_sys='nr',
                in_ndc=False
            ),
            image=global_image,
        )
        return img

    def make_compare_video(self,
                           global_cam: BatchCameraManager,
                           global_images: np.ndarray,
                           pose_idx: int = 0,
                           segi_overwrite: int = None,
                           putText=True) -> List[np.ndarray]:
        """
        Args:
            pose_idx: usually 0 for drawing eval frames
        """
        frames = []
        image_h, image_w = global_images[0].shape[:2]
        if low_reso:= True:
            new_h = 384  # new_h, new_w = 384, 384
            new_w = int((new_h / image_h) * image_w)
            image_h = new_h
            resized_images = []
            for i, img in enumerate(global_images):
                resized_images.append(cv2.resize(img, (new_w, new_h)))
            new_h = torch.ones_like(global_cam.fx) * new_h
            new_w = torch.ones_like(global_cam.fx) * new_w
            global_cam = global_cam.resize(new_h=new_h, new_w=new_w)
            global_images = resized_images
            fontsize = 1.0
        else:
            fontsize = 2.0

        font = cv2.FONT_HERSHEY_DUPLEX
        for _i, abs_ind in enumerate(self._out_video_inds):
            abs_ind = int(abs_ind)
            global_image = global_images[self._abs_ind_to_data_idx[abs_ind]]
            img_mesh = self.render_global(
                global_cam=global_cam,
                global_images=global_images,
                pose_idx=pose_idx,
                sample_idx=abs_ind,
                with_hand=True)
            frame = self.nonunique_frames[_i]

            sideview = self.render_triview(
                pose_idx, abs_ind, views=['left', 'back'], hstack=False,
                rend_size=image_h//2)
            img = np.hstack([global_image, img_mesh * 255 , sideview * 255])
            segi = self._abs_ind_to_segi[abs_ind]
            ref = self.segments[segi].ref
            segi_display = segi_overwrite if segi_overwrite is not None else segi
            if putText:
                # img = cv2.putText(img, f'{_i=} {abs_ind=} {frame=} [{segi}] {ref}', (10, 60), font, fontsize, (0, 255, 0), 2, cv2.LINE_AA)
                # img = cv2.putText(
                #     img, f'{_i=} {abs_ind=} {frame=} [{self._display_segi}] {ref}', 
                #     (10, 60), font, fontsize, (0, 255, 0), 2, cv2.LINE_AA)
                img = cv2.putText(
                    img, f'{_i=} {abs_ind=} {frame=} [{segi_display}] {ref}', 
                    (10, 60), font, fontsize, (0, 255, 0), 2, cv2.LINE_AA)
            frames.append(img)
        return frames

    @torch.no_grad()
    def make_hover_video(self, D, pose_idx: int):
        """
        Visualise object scale & depth w.r.t cam in open3d's step-back observer
        """
        render = get_global_o3d_render()
        render.scene.clear_geometry()
        background_color = [1, 1, 1, 1.0]
        render.scene.set_background(background_color)
        if D.sun_light:
            render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
            render.scene.scene.enable_sun_light(True)
        else:
            render.scene.set_lighting(rendering.Open3DScene.NO_SHADOWS, (0, 0, 0))
        if 'pcd_mesh' in D.o3d:
            render.scene.add_geometry('pcd', D.o3d.pcd_mesh, D.o3d.m.white)
        render.setup_camera(
            D.o3d.fov, 
            D.o3d.view_cam.lookat, D.o3d.view_cam.eye, D.o3d.view_cam.up)

        # Add object to render
        abs_inds = self._out_video_inds

        images = []
        w2cs = self.get_w2cs(abs_inds).view(-1, 4, 4)
        c2ws = w2cs.inverse().view(-1, 4, 4).cpu().numpy()
        frustums = D.o3d.frustums
        for i, _abs_ind in enumerate(abs_inds):
            ml, mr, mo = self.get_meshes(pose_idx=pose_idx, sample_idx=_abs_ind)  # in ego coord.
            c2w = c2ws[i]
            mo = mo.apply_transform(c2w)
            update_render_geom_from_trimesh(render, mo, 'obj', D.o3d.m.yellow)
            update_render_geom(render, frustums[i], 'ego', D.o3d.m.red)  # frustum
            if ml is not None:
                ml = ml.apply_transform(c2w)
                update_render_geom_from_trimesh(render, ml, 'left', D.o3d.m.purple)
            if mr is not None:
                mr = mr.apply_transform(c2w)
                update_render_geom_from_trimesh(render, mr, 'right', D.o3d.m.purple)
            
            if D.hands is not None:
                # Add Raw(unscaled) hand
                for side, vh_list, fh in zip(
                    ['left', 'right'], [D.hands.vl_list, D.hands.vr_list], 
                    [D.hands.fl, D.hands.fr]):
                    vh = vh_list[_abs_ind]
                    if vh is not None:
                        mh = SimpleMesh(vh, fh).apply_transform(c2w)
                        update_render_geom_from_trimesh(
                            render, mh, f'raw_{side}', D.o3d.m.purple)

            img_buf = render.render_to_image()
            img = np.asarray(img_buf)
            images.append(img)
        
        render = None  # Release render awkwardly to avoid SIGSEGV at the end of the code
        inc_render_usage_counter()
        return images