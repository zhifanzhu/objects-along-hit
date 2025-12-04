from typing import List
import numpy as np
import torch
import json
from homan.math import qvec2rotmat


EPIC_FIELDS_DATA_ROOT = 'DATA_STORAGE/epic_fields'
METRIC_TRANSFORM_DIR = 'weights/epicfields_metric_transform'


def load_metric_sfm(vid: str):
    """ Load epic field that has been manually aligned to metric scale.
    Gravity is aligned to neg_Z.

    Args:
        sample_frames: List of (T,)

    Returns:
        dict.
            - points: List of [x, y, z, r, g, b] for each point
            - images: 
                key: int of frame number
                value: (4, 4) of w2c. np.ndarray
    """
    path = f'{EPIC_FIELDS_DATA_ROOT}/{vid}.json'
    with open(path, 'r') as f:
        sfm = json.load(f)
    metric_transform_path = f'{METRIC_TRANSFORM_DIR}/{vid}.json'
    with open(metric_transform_path, 'r') as f:
        metric_transform = json.load(f)
    T = np.array(metric_transform['T'])  # T = [sR|t]
    _scale = np.cbrt(np.linalg.det(T[:3, :3]))  # det(sR) = s^3, hence cube-root

    # transform points xyz
    points = np.asarray([p[:3] for p in sfm['points']]).reshape(-1, 3)
    points = np.dot(points, T[:3, :3].T) + T[:3, -1].reshape(1, 3)
    new_xyz = points.reshape(-1, 3).tolist()
    sfm['points'] = [xyz + p[3:] for xyz, p in zip(new_xyz, sfm['points'])]

    # transform camera extrinsics
    # (qvec, tvec) -> w2c -> c2w -> c2w' -> w2c' -> (qvec', tvec')
    new_sfm_images = dict()
    for key, param in sfm['images'].items():
        frame_num = int(key.split('_')[-1].split('.')[0])
        w2c = get_sfm_w2c(param)
        c2w = np.linalg.inv(w2c)
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = (T[:3, :3] / _scale) @ c2w[:3, :3]
        new_c2w[:3, -1] = np.dot(T[:3, :3], c2w[:3, -1]) + T[:3, -1]
        new_w2c = np.linalg.inv(new_c2w)
        new_sfm_images[frame_num] = new_w2c
    sfm['images'] = new_sfm_images

    return sfm

def extract_w2c_samples(sfm: dict, frames_per_seg: List):
    w2cs = []
    for frames in frames_per_seg:
        _w2c = []
        for f in frames:
            param = sfm['images'][f'frame_{f:010d}.jpg']
            w2c = get_sfm_w2c(param)
            _w2c.append(w2c)
        w2c = torch.from_numpy(np.stack(_w2c)).float()
        w2cs.append(w2c)
    return w2cs


def get_sfm_w2c(image_data: List[float]):
    """ image_data: [qvec, tvec] """
    qvec, tvec = image_data[:4], image_data[4:]
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(qvec)
    w2c[:3, -1] = tvec
    return w2c