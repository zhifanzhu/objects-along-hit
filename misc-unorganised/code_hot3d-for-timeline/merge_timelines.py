
"""
Code to merge short B's and C's in the timelines saved in the json files
"""

import os
import json
import copy
import argparse
from tqdm import tqdm
from pprint import pprint # debugging


class MergeTimelines():
    def __init__(self, timeline_path, threshold):
        self.timeline_path = timeline_path
        self.threshold = threshold
        self.timeline_data = json.load(open(timeline_path))
        self.merged_timelines = list()
        self.window = 4
        save_path = os.path.join(args.save_dir, os.path.basename(timeline_path)).replace('_longest.json', '_merged.json')
        assert not os.path.exists(save_path), f"Timeline already exists at {save_path}"
        for timeline in self.timeline_data:
            segments = sorted(timeline['segments'], key=lambda x: x['st'])
            merged_segments = self.merge_timeline(copy.deepcopy(segments))
            final_segments = self.check_timeline(copy.deepcopy(merged_segments))
            # print(f'Segments before merging. {len(segments)=}')
            # pprint(segments)
            # print(f'Segments after merging. {len(final_segments)=}')
            # pprint(final_segments)
            self.merged_timelines.append({
                'timeline_name': timeline['timeline_name'],
                'seq_name': timeline['seq_name'],
                'cat': timeline['cat'],
                'total_start': timeline['total_start'],
                'total_end': timeline['total_end'],
                'segments': final_segments
            })
        json.dump(self.merged_timelines, open(save_path, 'w'), indent=2)
    
    def merge_timeline(self, segments):
        merged_segments = list()
        i = 0
        while i < len(segments):
            current_segment = segments[i]
            duration = current_segment['ed'] - current_segment['st']
            if i == 0:
                merged_segments.append(current_segment)
                i += 1
                continue
            if duration < self.threshold and current_segment['ref'] in ['inhand', 'scene_dynamic']:
                if i > 0 and segments[i-1]['ref'] != current_segment['ref']:
                    # Making sure we do not merge into A
                    if segments[i-1]['ref'] in ['inhand', 'scene_dynamic']:
                        if segments[i]['st'] == segments[i-1]['ed']:
                            # print(f'Merging {segments[i-1]} and {current_segment}')
                            merged_segments[-1]['ed'] = segments[i]['ed']
                        else:
                            merged_segments.append(current_segment)
                    else:
                        merged_segments.append(current_segment)
                elif i < len(segments) - 1 and segments[i+1]['ref'] != current_segment['ref']:
                    # Making sure we do not merge into A
                    if segments[i+1]['ref'] in ['inhand', 'scene_dynamic']:
                        if segments[i+1]['ed'] == segments[i]['st']:
                            # print(f'Merging {current_segment} and {segments[i+1]}')
                            current_segment['ed'] = segments[i+1]['ed']
                            merged_segments.append(current_segment)
                            i += 1
                        else:
                            merged_segments.append(current_segment)
                    else:
                        merged_segments.append(current_segment)
                else:
                    merged_segments.append(current_segment)
            else:
                merged_segments.append(current_segment)
            i += 1
        return merged_segments
    
    def check_timeline(self, merged_segments):
        """
        Now there could be cases where segments with same ['ref'] would be next to each other. We need to merge them
        """
        prev_len = len(merged_segments)
        new_len = 0
        # We want to make sure that all the segments are merged
        while prev_len != new_len:
            i = 0
            final_segments = list()
            while i < len(merged_segments):
                current_segment = merged_segments[i]
                # Only merging if there are two scene_dynamic segments next to each other
                if i < len(merged_segments) - 1 and current_segment['ref'] == merged_segments[i+1]['ref'] and current_segment['ref'] == 'scene_dynamic':
                    if current_segment['side'] == merged_segments[i+1]['side']:
                        current_segment['ed'] = merged_segments[i+1]['ed']
                        i += 1
                # Merging inhand segments with same hand side and same start and end time
                if i < len(merged_segments) - 1 and current_segment['ref'] == merged_segments[i+1]['ref'] and current_segment['ref'] == 'inhand' and current_segment['side'] == merged_segments[i+1]['side'] and current_segment['ed'] == merged_segments[i+1]['st']:
                    current_segment['ed'] = merged_segments[i+1]['ed']
                    i += 1
                final_segments.append(current_segment)
                i += 1
            if new_len == 0:
                prev_len = len(merged_segments)
            else:
                prev_len = new_len
            new_len = len(final_segments)
            merged_segments = copy.deepcopy(final_segments)
            # print(f'Prev: {prev_len}, New: {new_len}')
        return final_segments
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--timeline_dir',
        type=str,
        default='/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_v2_timelines/',
        help='Path to the directory containing the timelines',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_v2_timelines_merged/',
        help='Path to the directory where the merged timelines will be saved',
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=4,
        help='Number of timestamps to merge',
    )
    args = parser.parse_args()
    print(args)
    timelines = [os.path.join(args.timeline_dir, item) for item in os.listdir(args.timeline_dir) if 'json' in item]
    for timeline in tqdm(timelines):
        merger = MergeTimelines(timeline, args.threshold)
