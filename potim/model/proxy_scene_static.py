import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from potim.model.component_base import ComponentBase
from potim.model.inhand import InHand, InHandVis
from potim.defs.sim3 import Sim3


class ProxySceneStatic(ComponentBase):
    """ Different from SceneStatic, 
    this class does not store the object poses,
    it fetches the first/last pose from inhand model and it returns that pose when requested.
    """
    def __init__(self,
                 inhand: InHandVis,
                 inhand_index: int,
                 **kwargs):
        """
        Args:
            inhand_index: 0 or -1. 
                0: A-C, -1: C-A
        """
        super().__init__(**kwargs)
        self.inhand = inhand
        self.inhand_index = inhand_index
        assert self.inhand_index in {0, -1}, "Can only take the first or the last pose"

    def register_w2c(self, w2c: torch.Tensor):
        """
        w2c: (1, T, 4, 4)
        """
        assert len(w2c) == self.num_samples
        w2c = w2c.view(1, self.num_samples, 4, 4)
        self.register_buffer('w2c', w2c)

    def init_obj_transform(self,
                           inits: Sim3,
                           in_coord: str):
        raise ValueError("Set self.inhand instead")

    def get_obj_transform_ego(self, inds) -> Sim3:
        """ o2c () = w2c x o2w ()
        Args:
            inds: (T,)
        Returns:
            R_o2c: (N, T, 3, 3)
            t_o2c: (N, T, 3)
        """
        R_o2w, t_o2w, _ = self.get_obj_transform_world(inds)
        N, T = self.num_inits, len(inds)
        w2c = self.w2c[:, inds, ...].view(1, T, 4, 4)

        R_o2c = w2c[..., :3, :3] @ R_o2w
        t_o2c = w2c[..., :3, :3] @ t_o2w.view(N, T, 3, 1) + w2c[..., :3, [-1]]
        t_o2c = t_o2c.view(N, T, 3)
        return Sim3(R_o2c, t_o2c, None)

    def get_obj_transform_world(self, inds) -> Sim3:
        """ 
        Returns:
            R_o2w: (N, T, 3, 3)
            t_o2w: (N, T, 3)
        """
        inhand_ind = 0 if self.inhand_index == 0 else self.inhand.num_samples - 1
        R_o2w, t_o2w, _ = self.inhand.get_obj_transform_world([inhand_ind])  # (N, 1, 3, 3), (N, 1, 3)

        T = len(inds)
        R_o2w = R_o2w.tile(1, T, 1, 1)  # (N, T, 3, 3)
        t_o2w = t_o2w.tile(1, T, 1) # (N, T, 3)
        return Sim3(R_o2w, t_o2w, None)

    def get_w2cs(self, inds) -> torch.Tensor:
        """
        Args:
            inds: (T,)

        Returns:
            w2cs: (1, T, 4, 4)
        """
        return self.w2c[:, inds, ...].view(1, len(inds), 4, 4)