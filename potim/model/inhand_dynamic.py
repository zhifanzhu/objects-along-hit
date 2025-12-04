from functools import reduce
from collections import namedtuple
import torch
from einops import rearrange, repeat
from torch_scatter import scatter_min, scatter_mean
from nnutils.mesh_utils_extra import compute_vert_normals
from potim.model.inhand import InHandVis
from pytorch3d.ops import knn_gather, knn_points


class InHandDynamicVis(InHandVis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss_contact_consistency(self,
                                 inds,
                                 v_hand=None, v_obj=None,
                                 squared_dist=False,
                                 num_priors=8,
                                #  reduce_type='avg',
                                 num_nearest_points=1,
                                 contact_consistency_weight=0,
                                 contact_consistency_method='min_over_pairs',
                                 contact_consistency_region_reduce_type='none',
                                 verbose=False):
        """ To bring the object closer to the hand.
        L = distance from finger tips to their nearest vertices
        average over 8(=5+3) regions.
        Options:
            (5 regions vs 8 regions) x (min vs avg)

        Args:
            v_hand: (N*T, V, 3)
            v_obj: (N*T, V, 3)
            squared_dist: whether to calc loss as squared distance
            num_priors: 5 or 8
            reduce: 'min' or 'avg'

            w_smoothing: ablation, added Sep-17 2024,
                we ask the nearest-dist vec to be smooth between frames.
            w_smoothing_allpairs: ablation, added Sep-19 2024,
                we ask the nearest-dist vec to be smooth between all pairs of frames.

        Returns:
            loss: (N, T)
        """
        k1 = num_nearest_points
        v_hand = self.v_hand if v_hand is None else v_hand
        v_obj = self.get_verts_object() if v_obj is None else v_obj

        n = self.num_inits
        t = len(v_obj) // n
        # vn_obj = compute_vert_normals(v_obj, faces=self.faces_object)

        ph_idx = reduce(lambda a, b: a + b, self.contact_regions.verts, [])
        ph = v_hand[:, ph_idx, :]  # (N*T, CONTACT, 3)
        _, idx, nn = knn_points(ph, v_obj, K=k1, return_nn=True)
        # idx: (N*T, CONTACT, k1),  nn: (N*T, CONTACT, k1, 3)
        # vn_obj_nn = knn_gather(vn_obj, idx)  # (N*T, CONTACT, k1, 3)

        ph = ph.view(n*t, -1, 1, 3).expand(-1, -1, k1, -1)
        # (N*T, CONTACT, k1, 3) => (N*T, CONTACT, k1)
        # prod = torch.sum((ph - nn) * vn_obj_nn, dim=-1)

        _sqdist = ((ph - nn)**2).sum(dim=-1)  # (N*T, CONTACT, k1), absolute distance squared

        if contact_consistency_method == 'fullmat_over_smooth':
            _vobj = rearrange(v_obj, '(n t) vo d -> n t 1 vo d', n=n)
            _ph = rearrange(v_hand[:, ph_idx, :], '(n t) CONTACT d -> n t CONTACT 1 d', n=n)
            fullmat = ((_vobj - _ph)**2).sum(dim=-1)  # (N, T, CONTACT, VO)
            fullmat_diff = fullmat[:, 1:] - fullmat[:, :-1]  # (N, T-1, CONTACT, VO)
            l_consist = fullmat_diff.abs().mean(dim=(-2, -1))  # (N, T-1)
            l_consist = torch.cat([l_consist, l_consist.new_zeros(n, 1)], dim=1)

        elif contact_consistency_method == 'fullmat_over_pairs':
            _vobj = rearrange(v_obj, '(n t) vo d -> n t 1 vo d', n=n)
            _ph = rearrange(v_hand[:, ph_idx, :], '(n t) CONTACT d -> n t CONTACT 1 d', n=n)
            fullmat = ((_vobj - _ph)**2).sum(dim=-1)  # (N, T, CONTACT, VO)
            _fullmat_1 = rearrange(fullmat, 'n t CONTACT vo -> n t 1 CONTACT vo', n=n)
            _fullmat_2 = rearrange(fullmat, 'n t CONTACT vo -> n 1 t CONTACT vo', n=n)
            fullmat_diff_pairs = _fullmat_1 - _fullmat_2  # (N, T, T, CONTACT, k1)
            l_consist = fullmat_diff_pairs.abs().mean(dim=(-3, -2, -1))  # (N, T)

        elif contact_consistency_method == 'min_over_smooth':
            _sqdist = rearrange(_sqdist, '(n t) CONTACT k1 -> n t CONTACT k1', n=n)
            _sqdist_diff = _sqdist[:, 1:] - _sqdist[:, :-1]  # (N, T-1, CONTACT, k1)

            if contact_consistency_region_reduce_type != 'none':
                index = torch.cat(
                    [_sqdist.new_zeros(len(v), dtype=torch.long) + i
                    for i, v in enumerate(self.contact_regions.verts)])
                if contact_consistency_region_reduce_type == 'min':
                    _sqdist_diff, _ = scatter_min(src=_sqdist_diff, index=index, dim=-2)
                    _sqdist_diff = _sqdist_diff[..., :num_priors, :]
                elif contact_consistency_region_reduce_type == 'avg':
                    _sqdist_diff = scatter_mean(src=_sqdist_diff, index=index, dim=-2)
                    _sqdist_diff = _sqdist_diff[..., :num_priors, :]
            # _sqdist_diff: (N, T-1, CONTACT, k1) or (N, T-1, num_priors, k1)
            l_consist = _sqdist_diff.abs().mean(dim=(2, 3))  # (N, T-1)
            l_consist = torch.cat([l_consist, l_consist.new_zeros(n, 1)], dim=1)

        elif contact_consistency_method == 'min_over_pairs':
            _sqdist = rearrange(_sqdist, '(n t) CONTACT k1 -> n t CONTACT k1', n=n)
            _sqdist_1 = rearrange(_sqdist, 'n t CONTACT k1 -> n t 1 CONTACT k1', n=n)
            _sqdist_2 = rearrange(_sqdist, 'n t CONTACT k1 -> n 1 t CONTACT k1', n=n)
            _sqdist_diff = _sqdist_1 - _sqdist_2  # (N, T, T, CONTACT, k1)

            if contact_consistency_region_reduce_type != 'none':
                index = torch.cat(
                    [_sqdist.new_zeros(len(v), dtype=torch.long) + i
                    for i, v in enumerate(self.contact_regions.verts)])
                if contact_consistency_region_reduce_type == 'min':
                    _sqdist_diff, _ = scatter_min(src=_sqdist_diff, index=index, dim=-2)
                    _sqdist_diff = _sqdist_diff[..., :num_priors, :]
                elif contact_consistency_region_reduce_type == 'avg':
                    _sqdist_diff = scatter_mean(src=_sqdist_diff, index=index, dim=-2)
                    _sqdist_diff = _sqdist_diff[..., :num_priors, :]
            # _sqdist_diff: (N, T, T, CONTACT, k1) or (N, T, T, num_priors, k1)
            l_consist = _sqdist_diff.abs().mean(dim=(-3, -2, -1))  # (N, T)

        # prod = prod**2 if squared_dist else prod.abs_()
        # index = torch.cat(
        #     [prod.new_zeros(len(v), dtype=torch.long) + i
        #      for i, v in enumerate(self.contact_regions.verts)])

        # Use mean for k1 nearest points
        # prod = prod.mean(-1)    # (N*T, CONTACT, k1) => (N*T, CONTACT)
        # regions_min, _ = scatter_min(src=prod, index=index, dim=1)  # (N*T, 8)
        # regions_min = regions_min[..., :num_priors]

        # if reduce_type == 'min':
        #     loss = regions_min.min(dim=-1).values
        # elif reduce_type == 'avg':
        #     loss = regions_min.mean(dim=-1)

        # loss = loss.view(n, t)
        # phy_factor = self.physical_factor(inds)
        # loss = loss * phy_factor

        # if contact_consistency_weight > 0:
        l_consist = l_consist.view(n, t) * self.physical_factor(inds) * contact_consistency_weight
        # loss = l_consist
        # loss = loss + l_consist

        return l_consist

    def train_loss_dynamic(self, inds, optim_cfg, debug_check_nan=True) -> dict:
        """
        Args:
            inds: (T,)
            cfg: `optim` section of the config
        Returns:
            loss_dict:
                - keys: 'inside', 'close'
                - values: (N, T)
        """
        retVal = namedtuple('InhandLoss', 'loss_dict loss')
        nt = self.num_inits * len(inds)
        v_hand = self.v_hand[:, inds, ...]  # (1, T, V, 3)
        v_hand = v_hand.expand(self.num_inits, -1, -1, -1)
        v_obj = self.get_verts_object(inds)
        v_hand_sqz = v_hand.reshape(nt, -1, 3)
        v_obj_sqz = v_obj.reshape(nt, -1, 3)

        # _C = optim_cfg.loss_dynamic.contact_consistency
        loss_dict = {}
        l_inside_full = self.loss_insideness(
            inds=inds,
            v_hand=v_hand_sqz, v_obj=v_obj_sqz, v_obj_select=None,
            num_nearest_points=optim_cfg.loss.inside.num_nearest_points)
        l_inside = optim_cfg.loss.inside.weight * l_inside_full # .sum(-1)
        loss_dict['inside'] = l_inside

        l_close_full, l_consist = self.loss_closeness(
            inds=inds,
            v_hand=v_hand_sqz, v_obj=v_obj_sqz, v_obj_select=None,
            num_priors=optim_cfg.loss.close.num_priors,
            reduce_type=optim_cfg.loss.close.reduce,
            num_nearest_points=optim_cfg.loss.close.num_nearest_points,
            contact_consistency_weight=optim_cfg.loss.close.contact_consistency.weight,
            contact_consistency_method=optim_cfg.loss.close.contact_consistency.method,
            contact_consistency_region_reduce_type=optim_cfg.loss.close.contact_consistency.regional_reduce,
            ablation_no_closeness=optim_cfg.loss.close.get('ablation_no_closeness', False),
            )
        l_close = optim_cfg.loss.close.weight * l_close_full # .sum(-1)
        loss_dict['close'] = l_close
        l_consist = optim_cfg.loss.close.contact_consistency.weight * l_consist
        loss_dict['contact_consistency'] = l_consist

        # l_consist = self.loss_contact_consistency(
        #     inds=inds,
        #     v_hand=v_hand_sqz, v_obj=v_obj_sqz,
        #     num_priors=_C.num_priors,
        #     # reduce_type=_C.reduce,
        #     num_nearest_points=_C.num_nearest_points,
        #     contact_consistency_weight=_C.contact_consistency_weight,
        #     contact_consistency_method=_C.method,
        #     contact_consistency_region_reduce_type=_C.regional_reduce,
        #     )
        # l_consist = _C.weight * l_consist # .sum(-1)
        if debug_check_nan:
            from potim.debug_utils import check_nan
            check_nan(l_inside)
            check_nan(l_close)
        else:
            if torch.isnan(l_inside).any():
                print(f"WARNING: cleaning l_inside nan")
                l_inside[torch.isnan(l_inside)] = 0
            if torch.isnan(l_close).any():
                print(f"WARNING: cleaning l_close nan")
                l_close[torch.isnan(l_close)] = 0
        # loss_dict['consist'] = l_consist

        return retVal(loss_dict=loss_dict, loss=None)