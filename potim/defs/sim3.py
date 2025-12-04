import torch
from typing import NamedTuple


# Sim3 using matrix repr of rotation
class Sim3(NamedTuple):
    rot: torch.Tensor
    t: torch.Tensor
    s: torch.Tensor = None
    """
    rot: (..., 3, 3)
    transl: (..., 3)
    scale: (N, 1)
    """
    def to_matrix(self):
        """
        Returns: (..., 4, 4)
        """
        if self.rot.ndim == 3:
            N = len(self.rot)
            mat = torch.zeros([N, 4, 4], device=self.rot.device)
            mat[:, :3, :3] = self.rot[..., :3, :3]  # (..., 3, 3)
            mat[:, :3, [-1]] = self.t.view(N, 3, 1)
            mat[:, -1, -1] = 1.0
            return mat
        elif self.rot.ndim == 4:
            N, T = self.rot.shape[:2]
            mat = torch.zeros([N, T, 4, 4], device=self.rot.device)
            mat[..., :3, :3] = self.rot[..., :3, :3]  # (..., 3, 3)
            mat[..., :3, [-1]] = self.t.view(N, T, 3, 1)
            mat[..., -1, -1] = 1.0
            return mat
        else:
            raise NotImplementedError
    
    @staticmethod
    def from_matrix(mat):
        """
        Args:
            mat: (..., 4, 4)
        """
        if mat.ndim == 3:
            rot = mat[..., :3, :3]
            t = mat[..., :3, [-1]]
        elif mat.ndim == 4:
            rot = mat[..., :3, :3]
            t = mat[..., :3, [-1]]
        else:
            raise NotImplementedError
        return Sim3(rot, t, None)
    
    @property
    def shape(self):
        return self.rot.shape
    
    def __len__(self):
        return len(self.rot)
    
    def apply(self, pts):
        """
        Args:
            pts: (V, 3)
        Returns:
            pts: (..., V, 3) depends on rot.shape
        """
        assert len(pts.shape) == 2 and pts.shape[1] == 3
        _shape = [1] * (self.rot.ndim - 2) + [-1, 3]
        pts = pts.view(_shape)
        if self.t.shape[-1] == 3:
            t = self.t.unsqueeze(-2)
        else:
            t = self.t
        pts = pts @ self.rot.transpose(-1, -2) + t
        return pts
    
    def print_shape(self):
        print(
            f"rot: {self.rot.shape}\n"
            f"t: {self.t.shape}"
        )
        if self.s is not None:
            print(f"s: {self.s.shape}")
        else:
            print(f"s: None")


def loss_sim3(T1: Sim3, T2: Sim3):
    """
    Args:
        any: (N, T, ...)
        - except scale: (N,)
    Returns:
        l: (N, T)
    """
    # Squared Chordal distanc3
    # https://math.stackexchange.com/questions/2281374/distance-so3-rotation-matrix/4652060#4652060
    l_rot = torch.sum((T1.rot - T2.rot)**2, dim=(-1, -2)) 
    l_transl = torch.sum((T1.t - T2.t)**2, dim=-1)
    loss = l_rot + l_transl  # (N, T)
    if T1.s is not None and T2.s is not None:
        l_scale = torch.sum((T1.s - T2.s)**2, dim=1, keepdim=True)  # (N, 1)
        loss = loss + l_scale
    return loss
