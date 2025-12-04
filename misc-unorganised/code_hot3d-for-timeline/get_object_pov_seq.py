"""
Code to obtain sequences in HOT3D from object's movement POV
"""

import os
import cv2
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset_api import Hot3dDataProvider
from hot3d_constants import DATA_CONSTANTS


def end_occurrence(obj, next_obj, results, timestamp):
    if len(results[obj]) != 0 and results[obj][f'occurance_{len(results[obj])}']['end_timestamp'] is None and (pd.isna(next_obj) or next_obj != obj):
        results[obj][f'occurance_{len(results[obj])}']['end_timestamp'] = timestamp


def initialize_result(obj, results, timestamp):
    if obj not in results:
        results[obj] = {}
    if len(results[obj]) == 0:
        results[obj] = {
            'occurance_1': {
                'start_timestamp': timestamp,
                'end_timestamp': None,
                'stages': [],
            }
        }
    elif results[obj][f'occurance_{len(results[obj])}']['end_timestamp'] is not None:
        results[obj][f'occurance_{len(results[obj]) + 1}'] = {
            'start_timestamp': timestamp,
            'end_timestamp': None,
            'stages': [],
        }


def get_stage(row, side):
    stage = 'unrelated'
    if row[f'{side}_grasp_smooth'] == False and (row[f'{side}_iou'] == 0 or pd.isna(row[f'{side}_iou'])):
        stage = 'static'
    elif row[f'{side}_grasp_smooth'] == False and row[f'{side}_iou'] > 0:
        stage = 'unstable'
    elif row[f'{side}_grasp_smooth'] == True:
        stage = 'grasp'
    else:
        raise NotImplementedError('How can we be here?')
    return stage


if __name__ == '__main__':
    # Get the list of sequences
    sequences = os.listdir(DATA_CONSTANTS['grasp_information_save_dir_smooth'])
    sequences = [seq for seq in sequences if seq.startswith('P')]
    print(f'{len(sequences)} sequences found')
    
    # TODO: Debugging currently
    for sequence in sequences:
    # Check if the sequence is the preferred sequence
        if DATA_CONSTANTS['preferred_sequence'] == sequence.split('.')[0]:
            print(f'Processing sequence: {sequence}')
            # Read the CSV file for the sequence
            data = pd.read_csv(f"{DATA_CONSTANTS['grasp_information_save_dir_smooth']}/{sequence}")
            results = {}
            for idx, row in data.iterrows():
                timestamp = row['timestamp_ns']
                robj = row['right_object']
                lobj = row['left_object']
                
                # Handling the last row
                if idx + 1 == len(data):
                    # Update the end timestamp for each object in results
                    for key, value in results.items():
                        if value[f'occurance_{len(value)}']['end_timestamp'] is None:
                            value[f'occurance_{len(value)}']['end_timestamp'] = timestamp
                    break
                # Case (a): Both right and left objects are not NaN and are the same
                if not pd.isna(robj) and not pd.isna(lobj) and robj == lobj:
                    # print(f'{timestamp} Case (a) {robj}; {lobj}')
                    initialize_result(robj, results, timestamp)
                    if data.iloc[idx+1]['right_object'] == robj or data.iloc[idx+1]['left_object'] == lobj:
                        continue
                    end_occurrence(robj, data.iloc[idx+1]['right_object'], results, timestamp)
                # Case (b): Both right and left objects are NaN
                elif pd.isna(robj) and pd.isna(lobj):
                    # print(f'{timestamp} Case (b) {robj}; {lobj}')
                    pass
                # Case (c): Both right and left objects are not NaN and are different
                elif not pd.isna(robj) and not pd.isna(lobj) and robj != lobj:
                    # print(f'{timestamp} Case (c) {robj}; {lobj}')
                    initialize_result(robj, results, timestamp)
                    end_occurrence(robj, data.iloc[idx+1]['right_object'], results, timestamp)
                    initialize_result(lobj, results, timestamp)
                    end_occurrence(lobj, data.iloc[idx+1]['left_object'], results, timestamp)
                # Case (d): Only left object is not NaN
                elif pd.isna(robj) and not pd.isna(lobj):
                    # print(f'{timestamp} Case (d) {robj}; {lobj}')
                    initialize_result(lobj, results, timestamp)
                    end_occurrence(lobj, data.iloc[idx+1]['left_object'], results, timestamp)
                # Case (e): Only right object is not NaN
                elif not pd.isna(robj) and pd.isna(lobj):
                    # print(f'{timestamp} Case (e) {robj}; {lobj}')
                    initialize_result(robj, results, timestamp)
                    end_occurrence(robj, data.iloc[idx+1]['right_object'], results, timestamp)
                else:
                    raise NotImplementedError('How can we be here?')
            for key, value in results.items():
                for occur_key, occurance in value.items():
                    for idx, row in data.iterrows():
                        right_stage = 'unrelated'
                        left_stage = 'unrelated'
                        if row['timestamp_ns'] >= occurance['start_timestamp'] and row['timestamp_ns'] <= occurance['end_timestamp']:
                            if key == row['right_object'] and key != row['left_object']:
                                right_stage = get_stage(row, 'right')
                            elif key == row['left_object'] and key != row['right_object']:
                                left_stage = get_stage(row, 'left')
                            elif key == row['right_object'] and key == row['left_object']:
                                right_stage = get_stage(row, 'right')
                                left_stage = get_stage(row, 'left')
                            else:
                                raise NotImplementedError('How can we be here?')
                            results[key][occur_key]['stages'].append({
                                'timestamp': row['timestamp_ns'],
                                'right_stage': right_stage,
                                'left_stage': left_stage,
                            })
            # Visualise the stages
            if True:
                seq_folder = f"{DATA_CONSTANTS['dataset_path']}/{sequence.split('.')[0]}"
                hot3d = Hot3dDataProvider(
                    sequence_folder=seq_folder,
                    object_library=DATA_CONSTANTS['object_library_path'],
                    mano_hand_model=None,
                )
                img_stream_id = hot3d.device_data_provider.get_image_stream_ids()[0]
                for key, value in results.items():
                    for occur_key, occurance in value.items():
                        video_images = list()
                        save_path = f'{key}-{occur_key}.mp4'
                        for stage in tqdm(occurance['stages'], desc=f'Video: {save_path}'):
                            image = hot3d.device_data_provider.get_image(stage['timestamp'], img_stream_id) 
                            image = np.rot90(image.copy(), -1).copy()
                            object_state = 'ERROR'
                            if stage['right_stage'] == 'unrelated' and stage['left_stage'] == 'unrelated':
                                object_state = 'unrelated (CHECK)'
                            elif stage['right_stage'] == 'static' and stage['left_stage'] == 'static':
                                object_state = 'both hands static'
                            elif stage['right_stage'] == 'unstable' and stage['left_stage'] == 'unstable':
                                object_state = 'both hands unstable'
                            elif stage['right_stage'] == 'grasp' and stage['left_stage'] == 'grasp':
                                object_state = 'both hands grasp'
                            elif stage['right_stage'] == 'grasp' and stage['left_stage'] == 'unstable':
                                object_state = 'right H grasp (left H unstable)'
                            elif stage['right_stage'] == 'unstable' and stage['left_stage'] == 'grasp':
                                object_state = 'left H grasp (right H unstable)'
                            elif stage['right_stage'] == 'grasp' and stage['left_stage'] == 'static':
                                object_state = 'right H grasp (left H static)'
                            elif stage['right_stage'] == 'static' and stage['left_stage'] == 'grasp':
                                object_state = 'left H grasp (right H static)'
                            elif stage['right_stage'] == 'unstable' and stage['left_stage'] == 'static':
                                object_state = 'right H unstable (left H static)'
                            elif stage['right_stage'] == 'static' and stage['left_stage'] == 'unstable':
                                object_state = 'left H unstable (right H static)'
                            elif stage['right_stage'] == 'unrelated' and stage['left_stage'] == 'static':
                                object_state = 'left H static (right H unrelated)'
                            elif stage['right_stage'] == 'static' and stage['left_stage'] == 'unrelated':
                                object_state = 'right H static (left H unrelated)'
                            elif stage['right_stage'] == 'unrelated' and stage['left_stage'] == 'unstable':
                                object_state = 'left H unstable (right H unrelated)'
                            elif stage['right_stage'] == 'unstable' and stage['left_stage'] == 'unrelated':
                                object_state = 'right H unstable (left H unrelated)'
                            elif stage['right_stage'] == 'grasp' and stage['left_stage'] == 'unrelated':
                                object_state = 'right H grasp (left H unrelated)'
                            elif stage['right_stage'] == 'unrelated' and stage['left_stage'] == 'grasp':
                                object_state = 'left H grasp (right H unrelated)'
                            else:
                                breakpoint()
                                raise NotImplementedError('How can we be here?')
                            cv2.putText(image, object_state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                            video_images.append(image)
                        imageio.mimsave(save_path, video_images, fps=15)
                        print(f'Saved: {save_path}')
