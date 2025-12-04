import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from potim.model.component_base import ComponentBase
from potim.defs.sim3 import Sim3


class SceneStatic(ComponentBase):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

        I_6d = torch.tensor([1., 0, 0, 0, 1, 0]).view(1, 1, 6)
        self.R_o2w_6d = nn.Parameter(
            I_6d.tile(self.num_inits, 1, 1), requires_grad=True)  # (N, 1(T), 6)
        self.t_o2w = nn.Parameter(
            torch.zeros([self.num_inits, 1, 3]), requires_grad=True)  # (N, 1(T), 3)

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
        """ <del>Set o2w by constructing from the LAST element.</del>
        Args:
            in_coord: must be 'o2c'. Input coordinates.
        """
        if in_coord == 'o2c':
            raise NotImplementedError
        elif in_coord == 'o2w':
            T_o2w = inits.to_matrix()  # (N, 1, 4, 4)
            self.R_o2w_6d.data = matrix_to_rotation_6d(
                T_o2w[..., :3, :3]).view(-1, 1, 6)
            self.t_o2w.data = T_o2w[..., :3, -1].view(-1, 1, 3)
        else:
            raise ValueError("Unknown in_coord")

    def get_obj_transform_ego(self, inds) -> Sim3:
        """ o2c () = w2c x o2w ()
        Args:
            inds: (T,)
        Returns:
            R_o2c: (N, T, 3, 3)
            t_o2c: (N, T, 3)
            t: (N, 1)
        """
        N, T = self.num_inits, len(inds)
        w2c = self.w2c[:, inds, ...].view(1, T, 4, 4)

        R_o2w = rotation_6d_to_matrix(self.R_o2w_6d).tile(1, T, 1, 1)  # (N, T, 3, 3)
        t_o2w = self.t_o2w
        R_o2c = w2c[..., :3, :3] @ R_o2w
        t_o2c = w2c[..., :3, :3] @ t_o2w.view(N, 1, 3, 1) + w2c[..., :3, [-1]]
        t_o2c = t_o2c.view(N, T, 3)
        return Sim3(R_o2c, t_o2c, None)

    def get_obj_transform_world(self, inds) -> Sim3:
        """ 
        Returns:
            R_o2w: (N, T, 3, 3)
            t_o2w: (N, T, 3)
        """
        T = len(inds)
        R_o2w = rotation_6d_to_matrix(self.R_o2w_6d).tile(1, T, 1, 1)  # (N, T, 3, 3)
        t_o2w = self.t_o2w.tile(1, T, 1)  # (N, T, 3)
        return Sim3(R_o2w, t_o2w, None)

    def get_w2cs(self, inds) -> torch.Tensor:
        """
        Args:
            inds: (T,)

        Returns:
            w2cs: (1, T, 4, 4)
        """
        return self.w2c[:, inds, ...].view(1, len(inds), 4, 4)