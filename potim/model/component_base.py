import torch.nn as nn
from potim.defs.sim3 import Sim3

""" Base class for segment components in the model

All inherited classes must implement the same signature.
"""

class ComponentBase(nn.Module):
    def __init__(self,
                 num_inits: int,
                 num_samples: int,
                 seg_st: int,
                 seg_ed: int,
                 ref: str):
        """
        Args:
            ref: one of ['inhand' 'scene_static']
        """
        super().__init__()
        self.num_inits = num_inits
        self.num_samples = num_samples
        self.seg_st = seg_st
        self.seg_ed = seg_ed
        self.ref = ref
    
    def get_obj_transform_ego(self, inds) -> Sim3:
        """
        Args:
            inds: (T,) LOCAL index wrt this segment.

        Returns:
            R_o2c: (N, T, 3, 3)
            t_o2c: (N, T, 3)
            t: (N, 1)
        """
        raise NotImplementedError