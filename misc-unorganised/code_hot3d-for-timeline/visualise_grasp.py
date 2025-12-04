"""
This file contains the code to visualise the stable graps obtained after running get_grasp.py
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

from hot3d_utils import HOT3DUTILS
from hot3d_constants import DATA_CONSTANTS


def get_grasp_information(sequence):
    data = pd.read_csv(sequence)
    data['left_grasp_smooth'] = data['left_grasp_smooth'].fillna(False)
    data['right_grasp_smooth'] = data['right_grasp_smooth'].fillna(False)

    left_grasp_count = 0
    right_grasp_count = 0
    left_grasp_timestamps = {}
    right_grasp_timestamps = {}
    previous_left_grasp = None
    previous_right_grasp = None
    current_left_grasp = None
    current_right_grasp = None

    for idx, row in data.iterrows():
        if idx <= 1:
            previous_left_grasp = row['left_grasp_smooth']
            previous_right_grasp = row['right_grasp_smooth']
            continue
        current_left_grasp = row['left_grasp_smooth']
        current_right_grasp = row['right_grasp_smooth']
        if current_left_grasp and not previous_left_grasp:
            left_grasp_count += 1
            left_grasp_timestamps[left_grasp_count] = {
                'start': data.loc[idx-1]['timestamp_ns'],
                'end': None,
            }
        elif not current_left_grasp and previous_left_grasp:
            left_grasp_timestamps[left_grasp_count]['end'] = data.loc[idx-1]['timestamp_ns']
        if current_right_grasp and not previous_right_grasp:
            right_grasp_count += 1
            right_grasp_timestamps[right_grasp_count] = {
                'start': data.loc[idx-1]['timestamp_ns'],
                'end': None,
            }
        elif not current_right_grasp and previous_right_grasp:
            right_grasp_timestamps[right_grasp_count]['end'] = data.loc[idx-1]['timestamp_ns']
        # handle the last row
        if idx == len(data) - 1:
            if current_left_grasp:
                left_grasp_timestamps[left_grasp_count]['end'] = row['timestamp_ns']
            if current_right_grasp:
                right_grasp_timestamps[right_grasp_count]['end'] = row['timestamp_ns']
        previous_left_grasp = current_left_grasp
        previous_right_grasp = current_right_grasp
    return left_grasp_timestamps, right_grasp_timestamps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--grasp_video',
        action='store_true',
        help='Flag to save individual video of grasp sequences',
    )
    parser.add_argument(
        '--video',
        action='store_true',
        help='Flag to save the entire video with grasp labels',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='/home/cibo/sid/hot3d_results/obtained_grasps/13_07_2024',
        help='Path to save the video',
    )
    args = parser.parse_args()
    print(args)
    
    info_csv = os.listdir(DATA_CONSTANTS['grasp_information_save_dir'])
    sequences = [f'{DATA_CONSTANTS["grasp_information_save_dir"]}/{seq}' for seq in info_csv]
    for sequence in sequences:
        left_grasp_timestamps, right_grasp_timestamps = get_grasp_information(sequence)
        if args.video:
            seq_info = HOT3DUTILS(sequence=sequence.split('/')[-1].split('.')[0])
            seq_info.create_entire_video(
                left_grasp_timestamps,
                right_grasp_timestamps,
                f'{args.save_path}/{sequence.split("/")[-1].split(".")[0]}.mp4',
            )
        if args.grasp_video:
            sequence_name = sequence.split('/')[-1].split('.')[0]
            seq_info = HOT3DUTILS(sequence=sequence_name)
            for key, value in tqdm(left_grasp_timestamps.items(), desc='Left Grasp Videos'):
                save_path = f'{args.save_path}/{sequence_name}/left_grasp/'
                os.makedirs(save_path, exist_ok=True)
                video_path = f'{save_path}/{key}.mp4'
                if not os.path.exists(video_path):
                    seq_info.create_video(value['start'], value['end'], video_path)
            for key, value in tqdm(right_grasp_timestamps.items(), desc='Right Grasp Videos'):
                save_path = f'{args.save_path}/{sequence_name}/right_grasp/'
                os.makedirs(save_path, exist_ok=True)
                video_path = f'{save_path}/{key}.mp4'
                if not os.path.exists(video_path):
                    seq_info.create_video(value['start'], value['end'], video_path)
