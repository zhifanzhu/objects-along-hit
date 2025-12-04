from collections import namedtuple
from typing import NamedTuple, Union, List
import torch
import numpy as np
import bisect
from copy import deepcopy

INHAND = 'inhand'
SCENE_STATIC = 'scene_static'
SCENE_DYNAMIC = 'scene_dynamic'

class PotimSegment(NamedTuple):
    st: int
    ed: int
    ref: str
    side: str = None  # 'left' or 'right'


class ScaleGetter(torch.nn.Module):
    """ For getting the global scale that is shared
    by all the segments and potim itself.
    """
    def __init__(self, scale: float, update_scale: bool):
        super().__init__()
        self.scale = torch.nn.Parameter(
            scale, requires_grad=update_scale)
    def forward(self):
        return self.scale


def get_sample_indices(segments: List[PotimSegment],
                       sample_ratio: int):
    """
    Given a list of segments,
        we will sample evenly from min(st) to max(ed),
        we compute such sample indices for each segments.
        Also the PotimSegments are updated
            by downsample (raw_st, raw_ed).
    Args:
        segments: list

        inds: list of indices
        sample_ratio: int
    """
    _Segments = namedtuple(
        'SegmentSample',
        'segments nonunique_frames frames_per_seg offset num_samples'
    )

    min_st = min([seg.st for seg in segments])
    max_ed = max([seg.ed for seg in segments])
    n_total = (max_ed - min_st + 1) // sample_ratio
    total_frames = np.linspace(
        min_st, max_ed, num=n_total, endpoint=True,
        dtype=int)
    _total_inds = list(total_frames)
    # print(len(_total_inds))
    # num_samples = len(sample_frames)

    num_samples = 0
    sample_frames = []
    new_segs = []
    frames_per_seg = []
    offset = min_st
    for seg in segments:
        st, ed = seg.st, seg.ed
        frames_seg = total_frames[(total_frames >= st) & (total_frames <= ed)]
        st = _total_inds.index(frames_seg[0])
        ed = _total_inds.index(frames_seg[-1])
        seg = PotimSegment(st, ed, seg.ref, seg.side)
        new_segs.append(seg)
        frames_per_seg.append(frames_seg)
        sample_frames.extend(frames_seg)
        num_samples += len(frames_seg)

    res = _Segments(
        new_segs, sample_frames, frames_per_seg, offset, num_samples)
    return res


def fixed_segment_sampling(segments: List[PotimSegment],
                           valid_frames: List[List[int]],
                           max_samples: int,
                           force_max_samples: bool = False):
    """ Version of get_sample_indices with fixed number of samples per segment.
    Given a list of segments,
        we will sample a fixed number for every segment,
        we compute such sample indices for each segments.
        Also the PotimSegments are updated
            by downsample (raw_st, raw_ed).
    Args:
        segments: list, of S segments
        available_frames: list of S lists,
            each list contains the available frames for each segment
        max_samples: int, maximum number of samples per segment
        force_max_samples: bool, if True, force to sample max_samples
    """
    _Segments = namedtuple(
        'SegmentSample',
        'segments nonunique_frames frames_per_seg num_samples'
    )

    new_segs = []
    sample_frames = []
    frames_per_seg = []
    total_samples = 0
    for si, seg in enumerate(segments):
        st, ed = seg.st, seg.ed
        if force_max_samples:
            n_samples = max_samples
        else:
            n_samples = min(ed - st + 1, max_samples)
        valid_frames_seg = valid_frames[si]
        if len(valid_frames_seg) == 0:
            continue
        frames_seg = keep_valid_frames(valid_frames_seg, n_samples)
        # frames_seg = np.linspace(st, ed, num=n_samples, endpoint=True, dtype=int)
        new_inds = np.arange(len(frames_seg)) + total_samples
        new_st = new_inds[0]
        new_ed = new_inds[-1]
        new_seg = PotimSegment(new_st, new_ed, seg.ref, seg.side)
        new_segs.append(new_seg)
        frames_per_seg.append(frames_seg)
        sample_frames.extend(frames_seg)
        total_samples += n_samples

    res = _Segments(
        new_segs, sample_frames, frames_per_seg, total_samples)
    return res


def keep_valid_frames(avail_frames: List,
                      num_frames:int) -> List:
    """
    Given available frames,
    take the frames at the uniformly sampled indices.

    Args:
        avail_frames: available absolute frame numbers
    """
    avail_frames = sorted(avail_frames)
    probes = np.linspace(
        0, len(avail_frames)-1, num_frames, endpoint=True, dtype=int)
    keep_frames = [avail_frames[p] for p in probes]
    return sorted(keep_frames)
