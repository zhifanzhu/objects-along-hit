
import os
import cv2
import json
import imageio
import numpy as np
from tqdm import tqdm

from code_hot3d.hot3d_utils import HOT3DUTILS

timeline_path = '/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_v2_timelines/P0003_b573833a_longest.json'
video_save_path = '/media/cibo/DATA/Zhifan/hot3d/hot3d/dataset/grasp_information_v2_timelines/vis'
timeline_data = json.load(open(timeline_path))

hot3d_utils = HOT3DUTILS('P0003_b573833a')

data_provider = hot3d_utils.hot3d_data_provider.device_data_provider
image_stream_ids = data_provider.get_image_stream_ids()
timestamps = data_provider.get_sequence_timestamps()

def get_frames(timestamps, data_provider, ref, side):
    images = list()
    for timestamp in timestamps:
        image = data_provider.get_image(timestamp, data_provider.get_image_stream_ids()[0])
        if image is None:
            continue
        image = np.rot90(image.copy(), -1).copy()
        image = cv2.putText(image, f"{ref} - {side}; {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        images.append(image)
    return images

for timeline in timeline_data:
    video_path = f"{video_save_path}/{timeline['timeline_name']}.mp4"
    if os.path.exists(video_path):
        print(f"Video already exists at {video_path}. Skipping...")
        continue
    timelines_images = list()
    for segment in tqdm(timeline['segments'], desc=f"Processing {timeline['timeline_name']}"):
        selected_timestamps = timestamps[segment['st']:segment['ed']]
        timelines_images.extend(get_frames(selected_timestamps, data_provider, segment['ref'], segment['side']))
    print('Creating video...')
    imageio.mimsave(video_path, timelines_images, fps=30)
    print(f"Video saved at {video_path}")
