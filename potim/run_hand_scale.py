"""
For epic,
after joint_static, we also need to learn a global hand scale,
given the contact frames at the static boundary.
Optimse multi frames push & pull & centroid-dist to fit a hand scale.
Pick the one that has smallest optim loss.
"""
import torch
from potim.data.epic_segwise_ondemand_dataset import EPICOnDemandDataset, _EPICSingleVideo
from potim.utils.hand_scale_optim import optim_hand_scale
from potim.utils.cmd_logger import getLogger

logger = getLogger(__name__)


def pick_side_frames(segments, frame_thr):
    """ ret: list of (side, frame, dist) """
    tuples = []

    A_segments = [seg for seg in segments if seg['ref'] == 'scene_static']
    other_segments = [seg for seg in segments if seg['side'] in {'left', 'right'}]
    for A in A_segments:
        for q in (A['st'], A['ed']):
            for oseg in other_segments:
                d = dist_f_to_seg(q, oseg)
                if d < frame_thr:
                    tuples.append((oseg['side'], q, d))
    return tuples

def dist_f_to_seg(f: int, seg: dict):
    if seg['st'] <= f <= seg['ed']:
        return 0
    return min(abs(seg['st']-f), abs(seg['ed']-f))


def fit_hand_scale_for_v6(prev_all_o2w: dict, 
                          scale_hand: torch.Tensor,
                          scale_obj: torch.Tensor,
                          vds: _EPICSingleVideo):
    """  For v6
    There can be multiple static-contact-frames, 
    fit to all of them, and use the one having smallest push & pull loss 

    Args:
        scale_hand: torch.Tensor, (1,) 
            Although this is a Tensor, but it is best_prev_scale_hand, hence only 1 element
        scale_obj: torch.Tensor, (1
    """

    # Skip if no static segments
    if len([seg for seg in vds.tl['segments'] if seg['ref'] == 'scene_static']) == 0:
        return False

    o2w_map = prev_all_o2w
    scale_hand = scale_hand.clone()
    scale_obj = scale_obj.clone()

    hamer_avail_frames = vds.reader.get_hamer_avail_frames()
    tuples = pick_side_frames(vds.tl['segments'], frame_thr=10)  # 10 frames == 0.16s
    best_final_loss = 1e8
    best_scale_h = scale_hand
    for side, frame_to_fit, _d in tuples:
        if frame_to_fit not in o2w_map:
            logger.warning(f"frame {frame_to_fit} not in o2w_map, skipping")
            continue
        if frame_to_fit not in hamer_avail_frames[side]:
            logger.warning(f"frame {frame_to_fit} not in hamer_avail_frames, skipping")
            continue
        T_o2w = o2w_map[frame_to_fit].view(4, 4).to('cpu')
        vo_o = torch.from_numpy(vds.reader.obj_verts).float()  # (V, 3)
        fo = torch.from_numpy(vds.reader.obj_faces)

        _, _, vh_untransl, t_h2c = vds.reader.read_hamer([frame_to_fit], side)
        fh = vds.reader.fl if side == 'left' else vds.reader.fr
        fh = fh.view(-1, 3)
        vh_untransl = vh_untransl.view(1, -1, 3)
        t_h2c = t_h2c.view(1, 1, 3)

        T_w2c = vds.reader.read_w2c([frame_to_fit]).view(4, 4)  # (1, 4, 4)
        vo_world = (vo_o * scale_obj) @ T_o2w[:3, :3].T + T_o2w[:3, -1]
        vo_cam = vo_world @ T_w2c[:3, :3].T + T_w2c[:3, -1]
        vo_cam = vo_cam.view(1, -1 ,3).detach()
        scale_h, final_loss = optim_hand_scale(
            vo_cam, fo, vh_untransl, fh, t_h2c, scale_h=scale_hand)
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            best_scale_h = scale_h

    return best_scale_h