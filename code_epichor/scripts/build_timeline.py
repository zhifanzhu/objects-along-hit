"""
From 
/media/eve/SCRATCH/sid/extend_epic_grasp/final-batch-jsons_28-10-2024/output_timelines

we build the timelines and export to ./code_epichor/image_sets/subset_<date>.json

We merge consecutive segments of the same type into one segment,
while merging, we also generate SAM valid frames for merge segments, 
by both checking object existence in sam and avoiding the gap between segments.
"""
import os
import re
import os.path as osp
from libzhifan import io
import numpy as np
from PIL import Image


def build_sam_valid_frames(mp4_name, raw_segments):
    fmt_root = '/media/eve/SCRATCH/sid/extend_epic_grasp/'
    obj_mask_fmt = osp.join(
        fmt_root, 'data-scaling-only-obj_29-10-2024/',
        '%s/output_sam_masks/%012d.png') 
    valid_frames = []
    for seg in raw_segments:
        st = seg['st']
        ed = seg['ed']
        for f in range(st, ed+1):
            omask_path = obj_mask_fmt % (mp4_name, f)
            if not osp.exists(omask_path):
                print(f"{omask_path} not found")
                continue
            om = np.asarray(Image.open(omask_path))
            if (om == 1).sum() > 0:
                valid_frames.append(f)
    return valid_frames

def get_raw_tls():
    indir = '/media/eve/SCRATCH/sid/extend_epic_grasp/final-batch-jsons_28-10-2024/output_timelines'
    files = sorted([v for v in os.listdir(indir) if v.endswith('.json')])

    all_tls = []
    # src_dir = './DATA_STORAGE/annotated_epic_grasp/final_output/'
    # files = sorted(os.listdir(src_dir))

    # f: e.g. P01_01_27334_28208_28633_left_glass_batch-0008.json
    for f in files:
        infos = io.read_json(osp.join(indir, f))['absolute']
        mp4_name = re.sub(r'_batch-\d+\.json$', '', f)

        vid = '_'.join(mp4_name.split('_')[:2])
        side = mp4_name.split('_')[-2]
        cat = mp4_name.split('_')[-1]

        key2ref = {'static': 'scene_static', 'unstable_grasp': 'scene_dynamic', 'grasp': 'inhand'}
        tl = dict(
            timeline_name=None,
            vid=vid,
            mp4_name=mp4_name,
            cat=cat,
            main_side=side,
            segments=[],
        )
        total_start = 1e11
        total_end = -1
        for key, ref in key2ref.items():
            for st, ed in infos[key]:
                seg = dict(st=st, ed=ed, side=None, ref=ref)
                if ref == 'inhand':
                    seg['side'] = side
                tl['segments'].append(seg)
                total_start = min(total_start, st)
                total_end = max(total_end, ed)
        tl['timeline_name'] = f"{vid}_{side}_{cat}_{total_start}_{total_end}"
        tl['total_start'] = total_start
        tl['total_end'] = total_end

        all_tls.append(tl)
    return all_tls

def merge_consecutive_segments(segments):
    merged_segments = []
    if len(segments) == 0:
        return merged_segments
    segments = sorted(segments, key=lambda x: x['st'])
    cur_seg = segments[0]
    for seg in segments[1:]:
        if seg['ref'] == cur_seg['ref']:
            cur_seg['ed'] = seg['ed']
        else:
            merged_segments.append(cur_seg)
            cur_seg = seg
    merged_segments.append(cur_seg)
    return merged_segments


def main():
    all_tls = get_raw_tls()

    valid_frames_save_dir = './outputs/epic_cache/valid_frames'
    build_valid_frames = False
    for tl in all_tls:
        vid = tl['vid']
        mp4_name = tl['mp4_name']

        if build_valid_frames:
            valid_frames = build_sam_valid_frames(mp4_name, tl['segments'])

        tl['segments'] = merge_consecutive_segments(tl['segments'])

        if build_valid_frames:
            valid_frames_path = osp.join(
                valid_frames_save_dir, tl['mp4_name'] + ".json")
            os.makedirs(osp.dirname(valid_frames_path), exist_ok=True)
            io.write_json(valid_frames, valid_frames_path, indent=2)

    io.write_json(all_tls, './code_epichor/image_sets/subset_Nov1.json', indent=2)

if __name__ == '__main__':
    main()