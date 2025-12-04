import torch
from itertools import product
from collections import namedtuple
import numpy as np
from potim.defs.sim3 import Sim3
from potim.log_managerV2 import LogManagerV2
from potim.model.potim_model import (
    POTIM_SC, SCENE_STATIC, SCENE_DYNAMIC, INHAND
)


def segi_scheduler(segments, strategy: str):
    """ Chooses LONGEST static to start, output segi_order.
    
    1. normal: 0 to N-1
    2. one-way: <- S, -> S, S->
    3. circle: one-way with final to S. <- S, -> S, S->, S <-
    4. two-circle: 2x circle
    
    Later, might need to iterate twice
    1. Disperse: <- S ->
    2. Disperse: <- S ->; -> S <-
    
    Args:
        strategy: {'one-way'}

    Returns:
        segi_list: segi optimisation order.
            e.g. [2, 1, 0, 1, 2, 3, 4]. 
                Future might return a list of list.
    """
    num_segments = len(segments)
    num_A = sum([1 for seg in segments if seg['ref'] == 'scene_static'])
    if not num_segments > 1:
        print(f"Warning: {num_segments=}")
    if num_A == 0:
        print("No static segment found. Fall back to normal strategy.")
        return segi_normal_scheduler(segments)

    seg_lens = [seg['ed'] - seg['st'] + 1 
                if seg['ref'] == 'scene_static' else 0
                for seg in segments ]
    A0 = np.argmax(seg_lens)  # segi of initial A
    L, R = 0, num_segments
    full = list(range(L, R))

    if strategy == 'normal':
        return segi_normal_scheduler(segments)
    elif strategy == 'one-way':
        segi_list = full[A0:0:-1] + full[0:R:1]  # [A0 to 0), then [0 to R)

    elif strategy == 'circle':
        segi_list = []
        segi_list = full[A0:0:-1] + full[:-1] + full[R:A0:-1] # [A0 to 0), then [0 to R-1), then [R-1 to A0)
        segi_list = segi_list + [A0]  # Finally end with A0

    elif strategy == 'two-circle':
        segi_list = []
        circle = full[A0:0:-1] + full[:-1] + full[R:A0:-1] # [A0 to 0), then [0 to R-1), then [R-2 to A0)
        segi_list = circle + circle + [A0]
    
    if (debug := False):
        for segi in segi_list:
            print(segments[segi])
    return segi_list

def segi_normal_scheduler(segments):
    segi_list = list(range(len(segments)))
    return segi_list


def calc_propagation_frame_and_pose(all_prev_o2w: dict, cur_frames: list):
    """ Find the nearest two frames from two segments, respectively

    Args:
        all_prev_o2w: dict mapping frame number to (4, 4) o2w
        cur_frames: list of current frames

    Returns:
        prev_T_o2w: (4, 4), the pose at the nearest frame of the previous segment
        put_ind: the index of the nearest frame of the current segments.
            cur_frames[put_ind] is the nearest frame from the current segment
    """
    prev_frames = sorted(all_prev_o2w.keys())
    prev_frame, cur_frame = min(product(prev_frames, cur_frames), key=lambda t: abs(t[0] - t[1]))
    prev_T_o2w = all_prev_o2w[prev_frame]
    put_ind = cur_frames.index(cur_frame)
    return prev_T_o2w, put_ind

def make_o2w_mapping(saved_o2w: dict) -> dict:
    """ 
    Args:
        saved_o2w:
            - 'frames': list of N
            - 'o2w': (N, 4, 4)
    Returns: 
        dict of frame to (4, 4) 
    """
    frames = saved_o2w['frames']
    o2w = saved_o2w['o2w'].view(len(frames), 4, 4)
    o2w_mapping = dict(zip(frames, o2w))
    return o2w_mapping


def collect_and_transform_pose_inits(all_prev_o2w, 
                                     prev_scale_hand,
                                     prev_scale_obj,
                                     D,
                                     which_only):
    """ The main passing-control function.
    This function handles the passed-in poses according to the current segment type.

    Args:

    Returns:
        D  where
            D.init_o2h_list, D.static_inits_o2w_list concatenated accordingly
            D.scale_hand and D.scale_obj concatenated accordingly
            D.num_inits: number of total inits for the current segment
            D.pass_init_idx: the index of the passed-in init. -123 if no passing.

    """
    CUR = 0
    cur_ref = D.segments[CUR].ref

    # Determine prev_T_o2w at the previous boundary and which current frame to 'put'
    prev_T_o2w = None
    if all_prev_o2w is not None:
        prev_T_o2w, put_ind = calc_propagation_frame_and_pose(
            all_prev_o2w, D.meta_samples.frames_per_seg[CUR])
        prev_T_o2w = prev_T_o2w.view(1, 1, 4, 4)
        # print(prev_scale_hand, prev_scale_obj)

    # Transform w.r.t the current segment
    if cur_ref == SCENE_STATIC:
        D.init_types = ['Init'] * D.num_inits
        if prev_T_o2w is not None and which_only != SCENE_STATIC:
            D.scale_hand = torch.cat([D.scale_hand, prev_scale_hand], dim=0)
            D.scale_obj = torch.cat([D.scale_obj, prev_scale_obj], dim=0)

            # Only concat when we have previous pose and we are not running A-only
            o2w = D.static_inits_o2w_list[CUR].to_matrix().to(prev_T_o2w.device)
            o2w = torch.cat([o2w, prev_T_o2w], dim=0)
            D.static_inits_o2w_list[CUR] = Sim3.from_matrix(o2w)
            D.num_inits = len(o2w)
            D.init_types.append('Pass')

    elif cur_ref in {INHAND, SCENE_DYNAMIC}:
        D.init_types = ['Init'] * D.num_inits
        if prev_T_o2w is not None and which_only != INHAND:
            D.scale_hand = torch.cat([D.scale_hand, prev_scale_hand], dim=0)
            D.scale_obj = torch.cat([D.scale_obj, prev_scale_obj], dim=0)
            vhand_ego, T_h2c = make_vhand(
                D.T_h2cs[CUR], D.vhand_untransl_list[CUR], D.t_h2c_untransl_list[CUR], D.scale_hand)
            D.vhand_ego_list[CUR] = vhand_ego
            D.T_h2cs[CUR] = T_h2c

            # Similar to SCENE_STATIC, we only concat when we have previous pose and we are not running C-only
            T_o2h = D.init_o2h_list[CUR].to(prev_T_o2w.device) # (N, 4, 4)
            T_w2c = D.w2cs[CUR][[put_ind], ...].to(prev_T_o2w.device)  # (1, 4, 4)
            # we take the last one that was produced from prev_scale_hand
            prev_T_c2h = D.T_h2cs[CUR][-1, [put_ind], ...].inverse().to(prev_T_o2w.device)  # (1, 4, 4)
            prev_T_w2h = prev_T_c2h @ T_w2c
            prev_T_o2h = prev_T_w2h @ prev_T_o2w.view(1, 4, 4)
            T_o2h = torch.cat([T_o2h, prev_T_o2h], dim=0)  # (N+1, 4, 4)
            D.init_o2h_list[CUR] = T_o2h
            D.num_inits = len(T_o2h)
            D.init_types.append('Pass')

    else:
        raise ValueError(f"Unknown ref: {cur_ref}")

    return D


def make_vhand(T_h2c, verts_hand_untranslated, t_h2c_untransl, scale_hand):
    """ It seems that we can live without verts_hand_untranslated and t_h2c_untranslated, 
    eventually we can try to remove them to simplify the code.

    Args:
        verts_hand_untranslated: (1, T, V, 3)
        t_h2c_untransl: (1, T, 1, 3)
        scale_hand: (N,)
    
    Returns:
        verts_hand_ego: (N, T, V, 3)
        T_h2c: (N, T, 4, 4)
    """
    verts_hand_ego = (verts_hand_untranslated + t_h2c_untransl) * scale_hand.view(-1, 1, 1, 1)
    out_T_h2c = torch.cat([T_h2c, T_h2c[[0], ...]], dim=0)  # Some hacking as R_h2c are the same
    out_T_h2c[..., :3, [3]] = (t_h2c_untransl * scale_hand.view(-1, 1, 1, 1)).transpose(-2, -1)
    return verts_hand_ego, out_T_h2c


def pad_to_multiple(D, Np):
    """ Pad D's inits to be multiple of Np.
    """
    def repeat_last_slice(tensor, repeat):
        """ 
        Args:
            tensor: (N, ...)
            bsize: int
        Returns:
            (bsize, ...)
        """
        last_slice = tensor[-1:]  # shape (1, ...)
        expanded_slices = last_slice.expand(repeat, *last_slice.shape[1:])
        return torch.cat([tensor, expanded_slices], dim=0)

    if D.num_inits % Np == 0:
        return D

    pad_size = Np - D.num_inits % Np
    cur_ref = D.segments[0].ref
    CUR = 0
    if cur_ref == SCENE_STATIC:
        D.scale_hand = repeat_last_slice(D.scale_hand, pad_size)
        D.scale_obj = repeat_last_slice(D.scale_obj, pad_size)

        o2w = D.static_inits_o2w_list[CUR].to_matrix()
        o2w = repeat_last_slice(o2w, pad_size)
        D.static_inits_o2w_list[CUR] = Sim3.from_matrix(o2w)
        D.num_inits = len(o2w)
        D.init_types.extend(['Pad'] * pad_size)
    elif cur_ref in {INHAND, SCENE_DYNAMIC}:
        D.scale_hand = repeat_last_slice(D.scale_hand, pad_size)
        D.scale_obj = repeat_last_slice(D.scale_obj, pad_size)

        D.vhand_ego_list[CUR] = repeat_last_slice(D.vhand_ego_list[CUR], pad_size)
        D.T_h2cs[CUR] = repeat_last_slice(D.T_h2cs[CUR], pad_size)
        D.init_o2h_list[CUR] = repeat_last_slice(D.init_o2h_list[CUR], pad_size)
        D.num_inits = len(D.init_o2h_list[CUR])
        D.init_types.extend(['Pad'] * pad_size)
    else:
        raise ValueError(f"Unknown ref: {cur_ref}")
    
    return D


class RecordTracer:
    """ Trace which pose_init index has been optimised for each segment.
    controls (based on iou) which best prev_all_o2w to route out to the next segment.

    It also adds a `init_type` column to csv, which can be helpful for debugging.

    Current implementation caches in memory. Later we might want to save to disk if OOM.
    """

    def __init__(self):
        self.best_record = dict()  # segi -> RecordType for each segi.
        self.best_oiou = dict() # segi -> best_oiou for each segi.
        self.num_visited_inits = dict()  # segi -> number of visited inits

    def is_indep_init_completed(self, segi):
        if segi not in self.num_visited_inits:
            return False
        if self.num_visited_inits[segi] == 0:
            return False
        return True
    
    def update_record(self, 
                      segi: int, 
                      blob: tuple,
                      logmanager: LogManagerV2):
        """
        Clean-up entry, add to logmanger, and update best_record.
        """
        RecordType = namedtuple(
            'RecordType', 
            ['init_idx', 'sd_before_optim', 'sd_after_optim', 'entry',
            'eval_entry_before', 'all_o2w', 'scale_hand', 'scale_obj',
            ])

        (eval_entry, eval_entry_before, all_o2w, sd_before_optim, sd_after_optim,
            scale_hand, scale_obj, work_id, init_types, Np_start, Np_end) = blob

        if segi not in self.best_oiou:
            self.best_oiou[segi] = -1.0

        Np = Np_end - Np_start
        for np_ind in range(Np):
            if segi not in self.num_visited_inits:
                self.num_visited_inits[segi] = 0
            init_idx = self.num_visited_inits[segi]
            entry = self.make_entry(eval_entry, np_ind, init_idx, Np)  # After
            entry['init_type'] = init_types[np_ind + Np_start]
            entry['work_id'] = work_id
            entry_before = self.make_entry(eval_entry_before, np_ind, init_idx, Np)

            if entry['oiou'] > self.best_oiou[segi]:
                self.best_oiou[segi] = entry['oiou']
                self.best_record[segi] = RecordType(
                    init_idx=init_idx,
                    sd_before_optim=sd_before_optim,
                    sd_after_optim=sd_after_optim,
                    entry=entry,
                    eval_entry_before=entry_before,
                    all_o2w=all_o2w[np_ind, ...],  # (T, 4, 4)
                    scale_hand=scale_hand[[np_ind], ...],  # (1,)
                    scale_obj=scale_obj[[np_ind], ...], # (1,)
                )

            logmanager.add_entry(entry)
            self.num_visited_inits[segi] += 1

    def fetch_best(self, segi):
        return self.best_record[segi]

    @staticmethod
    def make_entry(eval_entry, np_ind, init_idx, Np):
        entry = dict(init_idx=init_idx)
        for k, v in eval_entry.items():
            if isinstance(v, torch.Tensor):
                assert v.shape[0] == Np
                entry[k] = v[np_ind].item()
            elif isinstance(v, list):
                assert len(v) == Np
                entry[k] = v[np_ind]
            else:
                entry[k] = v
        return entry