import torch
from homan.utils.geometry import spiral_rotations
from potim.defs.sim3 import Sim3
from potim.utils.scene_static_epic import get_R_o2w_upright_epic


def random_inhand_upright_epic(cat: str) -> torch.Tensor:
    """ This is to mimic static' initialisation.
    Returns:
        T_o2h: (N, 4, 4) where
            R_o2h: (N, 3, 3)
                N = 4 for cup, mug, saucepan and pan
                N = 1 otherwise
            t_o2h
    """
    R_o2h = get_R_o2w_upright_epic(cat)
    t_o2h = torch.zeros(len(R_o2h), 3)
    T_o2h = Sim3(R_o2h, t_o2h, None).to_matrix()
    return T_o2h


def spiral_inhand_upright(num_inits=4) -> torch.Tensor:
    """
    Returns:
        T_o2h: (N, 4, 4)
    """
    R_o2h = spiral_rotations(
        num_inits, num_sym_rots=1, sym_axis='+z', lim_ratio=1.0)
    t_o2h = torch.zeros(len(R_o2h), 3)
    T_o2h = Sim3(R_o2h, t_o2h, None).to_matrix()
    return T_o2h