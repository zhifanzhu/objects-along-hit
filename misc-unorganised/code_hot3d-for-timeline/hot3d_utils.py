
import os
import cv2
import time
import numpy as np

import torch
import imageio
import trimesh
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d
from pytorch3d.loss import chamfer_distance

from dataset_api import Hot3dDataProvider
from hot3d_constants import DATA_CONSTANTS
from data_loaders.mano_layer import MANOHandModel
from data_loaders.loader_object_library import ObjectLibrary
from code_arctic.mesh_interaction import pcd_mesh_distance_approx
from data_loaders.loader_object_library import load_object_library

from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions


class HOT3DUTILS():
    def __init__(self, sequence='P0001_10a27bf7'):
        self.sequence = sequence
        self.mano_hand_model_path = DATA_CONSTANTS['mano_hand_model_path']
        assert os.path.exists(self.mano_hand_model_path), 'MANO hand model path does not exist'
        self.dataset_path = DATA_CONSTANTS['dataset_path']
        assert os.path.exists(self.dataset_path), 'Dataset path does not exist'
        self.object_library_path = DATA_CONSTANTS['object_library_path']
        assert os.path.exists(self.object_library_path), 'Object library path does not exist'
        self.sequence_path = os.path.join(self.dataset_path, self.sequence)
        assert os.path.exists(self.sequence_path), 'Sequence path does not exist'

        self.mano_hand_model = MANOHandModel(self.mano_hand_model_path)
        self.hot3d_data_provider = Hot3dDataProvider(
            sequence_folder=self.sequence_path,
            object_library=self.object_library_path,
            mano_hand_model=self.mano_hand_model,
        )
        self.device_data_provider = self.hot3d_data_provider.device_data_provider
        self.image_stream_ids = self.device_data_provider.get_image_stream_ids()
        self.timestamps = self.device_data_provider.get_sequence_timestamps()
        self.object_pose_data_provider = self.hot3d_data_provider.object_pose_data_provider
        self.object_library = load_object_library(object_library_folderpath=self.object_library_path)
        self.obj_mesh_cache = dict()
        # There are cases where all the objects are not visible for timestamp at index 0
        temp_obj_pose, _ = self.get_poses_at_timestamp(self.timestamps[2])
        for obj_uid, _ in tqdm(temp_obj_pose.poses.items(), desc='Caching objects'):
            temp_obj_cad_filepath = ObjectLibrary.get_cad_asset_path(
                object_library_folderpath=self.object_library.asset_folder_name,
                object_id=obj_uid,
            )
            temp_obj_mesh = trimesh.load(temp_obj_cad_filepath, force='mesh')
            self.obj_mesh_cache[obj_uid] = {
                'mesh': temp_obj_mesh,
                'cad_filepath': temp_obj_cad_filepath,
            }

    def trimesh_to_pytorch(self, trimesh_mesh):
        vertices = torch.tensor(trimesh_mesh.vertices, dtype=torch.float32).unsqueeze(0)
        faces = torch.tensor(trimesh_mesh.faces, dtype=torch.int64).unsqueeze(0)
        pytorch3d_mesh = Meshes(verts=vertices, faces=faces)
        return pytorch3d_mesh
    
    def get_hand_object_details_at_timestamp(self, timestamp_ns):
       objects_pose, hands_pose = self.get_poses_at_timestamp(timestamp_ns)
       hand_details = self.get_hand_info_dict(hands_pose)
       object_details = self.get_object_info_dict(objects_pose)
       return hand_details, object_details

    def get_poses_at_timestamp(self, timestamp_ns):
        objects_pose_with_dt = self.hot3d_data_provider.object_pose_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        objects_pose = objects_pose_with_dt.pose3d_collection
        hand_poses_with_dt = self.hot3d_data_provider.mano_hand_data_provider.get_pose_at_timestamp(
            timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
            time_domain=TimeDomain.TIME_CODE,
        )
        hands_pose = hand_poses_with_dt.pose3d_collection
        return objects_pose, hands_pose

    def get_object_info_dict(self, objects_pose):
        object_dict = dict()
        object_count = 0
        for object_uid, object_pose3d in objects_pose.poses.items():
            object_name = self.object_library.object_id_to_name_dict[object_uid]
            if object_uid in self.obj_mesh_cache.keys():
                object_cad_filepath = self.obj_mesh_cache[object_uid]['cad_filepath']
                trimesh_mesh = self.obj_mesh_cache[object_uid]['mesh'].copy()
            else:
                object_cad_filepath = ObjectLibrary.get_cad_asset_path(
                    object_library_folderpath=self.object_library.asset_folder_name,
                    object_id=object_uid,
                )
                trimesh_mesh = trimesh.load(object_cad_filepath, force='mesh')
            object_pose = object_pose3d.T_world_object.to_matrix()
            trimesh_mesh_transformed = trimesh_mesh.apply_transform(object_pose)
            assert object_count not in object_dict, 'Object count already exists'
            object_dict[object_count] = {
                'object_name': object_name,
                'object_mesh': trimesh_mesh_transformed,
                'object_cad_filepath': object_cad_filepath,
                'object_uid': object_uid,
            }
            object_count += 1
        return object_dict

    def get_hand_info_dict(self, hands_pose):
        hand_dict = dict()
        for hand_pose_data in hands_pose.poses.values():
            hand_mesh_vertices = self.hot3d_data_provider.mano_hand_data_provider.get_hand_mesh_vertices(
                hand_pose_data,
            )
            hand_triangles, hand_vertex_normals = self.hot3d_data_provider.mano_hand_data_provider.get_hand_mesh_faces_and_normals(
                hand_pose_data,
            )
            mesh = trimesh.Trimesh(
                vertices=hand_mesh_vertices,
                faces=hand_triangles,
                vertex_normals=hand_vertex_normals,
                process=False,
            )
            hand_sidedness = 'right' if hand_pose_data.is_right_hand() else 'left'
            hand_dict[hand_sidedness] = {
                'hand_mesh': mesh,
            }
        return hand_dict
    
    def get_closest_objects_v2(self, timestamp_ns, dist_threshold=0.1):
        info = {
            'closest_left_dist': float('inf'),
            'closest_right_dist': float('inf'),
            'closest_obj_left': None,
            'closest_obj_right': None,
            'closest_obj_left_mesh': None,
            'closest_obj_right_mesh': None,
            'left_hand_mesh': None,
            'right_hand_mesh': None,
        }
        hand_details, object_details = self.get_hand_object_details_at_timestamp(timestamp_ns)
        for hand in hand_details.items():
            hand_side, hand_mesh = hand
            closest_objs = list()
            for value in object_details.values():
                object_name = value['object_name']
                object_mesh = value['object_mesh']
                object_mesh_pytorch = self.trimesh_to_pytorch(object_mesh)
                hand_mesh_pytorch = self.trimesh_to_pytorch(hand_mesh['hand_mesh'])
                loss, _ = chamfer_distance(
                    hand_mesh_pytorch.verts_list()[0].unsqueeze(0).to('cuda'),
                    object_mesh_pytorch.verts_list()[0].unsqueeze(0).to('cuda'),
                )
                if loss < dist_threshold:
                    closest_objs.append([object_name, object_mesh, loss.item(), hand_mesh_pytorch, object_mesh_pytorch])
            if len(closest_objs) > 1:
                for item in closest_objs:
                    min_dist = torch.min(
                        torch.cdist(
                            item[3].verts_list()[0],
                            item[4].verts_list()[0],
                        )
                    )
                    item.append(min_dist.item())
                closest_obj = sorted(closest_objs, key=lambda x: x[-1])[0]
            elif len(closest_objs) == 1:
                closest_obj = closest_objs[0]
            elif len(closest_objs) == 0:
                closest_obj = [None, None, float('inf')]
            else:
                raise AttributeError('How many objects do you have?')
            info[f'closest_{hand_side}_dist'] = closest_obj[2]
            info[f'closest_obj_{hand_side}'] = closest_obj[0]
            info[f'closest_obj_{hand_side}_mesh'] = closest_obj[1]
            info[f'{hand_side}_hand_mesh'] = hand_mesh['hand_mesh']
        return info

    def get_closest_objects(self, timestamp_ns):
        info = {
            'closest_left_dist': float('inf'),
            'closest_right_dist': float('inf'),
            'closest_obj_left': None,
            'closest_obj_right': None,
            'closest_obj_left_mesh': None,
            'closest_obj_right_mesh': None,
            'left_hand_mesh': None,
            'right_hand_mesh': None,
        }
        hand_details, object_details = self.get_hand_object_details_at_timestamp(timestamp_ns)
        # Approximated close, takes half time as the original one. Not used though.
        # right_done = False
        # left_done = False
        # for value in object_details.values():
        #     object_name = value['object_name']
        #     object_mesh = value['object_mesh']
        #     for hand in hand_details.items():
        #         hand_side, hand_mesh = hand
        #         if hand_side == 'right' and right_done:
        #             continue
        #         if hand_side == 'left' and left_done:
        #             continue
        #         object_mesh_pytorch = self.trimesh_to_pytorch(object_mesh)
        #         hand_mesh_pytorch = self.trimesh_to_pytorch(hand_mesh['hand_mesh'])
        #         loss, _ = chamfer_distance(
        #             hand_mesh_pytorch.verts_list()[0].unsqueeze(0),
        #             object_mesh_pytorch.verts_list()[0].unsqueeze(0),
        #         )
        #         if loss < info[f'closest_{hand_side}_dist']:
        #             info[f'closest_{hand_side}_dist'] = loss.item()
        #             if loss < 0.01:
        #                 if hand_side == 'right':
        #                     right_done = True
        #                 else:
        #                     left_done = True
        #             info[f'closest_obj_{hand_side}'] = object_name
        #             info[f'closest_obj_{hand_side}_mesh'] = object_mesh
        #             info[f'{hand_side}_hand_mesh'] = hand_mesh['hand_mesh']
        for hand in hand_details.items():
            hand_side, hand_mesh = hand
            closest_obj = None
            closest_dist = float('inf')
            for value in object_details.values():
                object_name = value['object_name']
                object_mesh = value['object_mesh']
                object_mesh_pytorch = self.trimesh_to_pytorch(object_mesh)
                hand_mesh_pytorch = self.trimesh_to_pytorch(hand_mesh['hand_mesh'])
                loss, _ = chamfer_distance(
                    hand_mesh_pytorch.verts_list()[0].unsqueeze(0),
                    object_mesh_pytorch.verts_list()[0].unsqueeze(0),
                )
                if loss < closest_dist:
                    closest_dist = loss
                    closest_obj = object_name
                    closest_mesh = object_mesh
            if hand_side == 'left':
                info['closest_left_dist'] = closest_dist.item()
                info['closest_obj_left'] = closest_obj
                info['closest_obj_left_mesh'] = closest_mesh
                info['left_hand_mesh'] = hand_mesh['hand_mesh']
            else:
                info['closest_right_dist'] = closest_dist.item()
                info['closest_obj_right'] = closest_obj
                info['closest_obj_right_mesh'] = closest_mesh
                info['right_hand_mesh'] = hand_mesh['hand_mesh']
        return info
    
    def get_contact_points(self, timestamp_ns, hand_side='right', closeness_threshold=0.1, closest_objects=None, use_approx=False, approx_tight=False):
        if closest_objects is None:
            closest_objects = self.get_closest_objects_v2(timestamp_ns)
        selected_index = []
        object_name = None
        if hand_side == 'right' and closest_objects['closest_right_dist'] < closeness_threshold:   
            selected_index = self.get_overlapping_vertices(closest_objects, side='right', approx=use_approx, approx_tight=approx_tight)
            object_name = closest_objects['closest_obj_right']
        elif hand_side == 'left' and closest_objects['closest_left_dist'] < closeness_threshold:
            selected_index = self.get_overlapping_vertices(closest_objects, side='left', approx=use_approx, approx_tight=approx_tight)
            object_name = closest_objects['closest_obj_left']
        elif hand_side != 'right' and hand_side != 'left':
            raise AttributeError('How many hands do you have?')
        return selected_index, object_name

    def get_overlapping_vertices(self, closest_objects, side='right', approx=False, approx_tight=False):
        if approx:
            hand_mesh_torch = torch.from_numpy(closest_objects[f'{side}_hand_mesh'].vertices).view(1, -1, 3).float()
            faces_hand_mesh_torch = torch.from_numpy(closest_objects[f'{side}_hand_mesh'].faces)
            obj_mesh_torch = torch.from_numpy(closest_objects[f'closest_obj_{side}_mesh'].vertices).view(1, -1, 3).float()
            d2 = pcd_mesh_distance_approx(hand_mesh_torch, faces_hand_mesh_torch, obj_mesh_torch)
            d_pred = d2.view(-1).numpy()
            sign_pred = (d_pred <= 1e-3).astype(int)
            selected_index = sign_pred.nonzero()[0]
            if approx_tight and len(selected_index) > 0:
                selected_vertices = closest_objects[f'closest_obj_{side}_mesh'].vertices[selected_index]
                dist = trimesh.proximity.signed_distance(closest_objects[f'{side}_hand_mesh'], selected_vertices)
                sel_approx_tight_ind = selected_index[np.where(dist >= 0)[0]]
                return sel_approx_tight_ind
        else:
            signed_distance = trimesh.proximity.signed_distance(
                    closest_objects[f'{side}_hand_mesh'],
                    closest_objects[f'closest_obj_{side}_mesh'].vertices,
                )
            selected_index = np.where(signed_distance >= 0)[0]
        return selected_index

    def create_video(self, start_timestamp, end_timestamp, save_path):
        device_data_provider = self.hot3d_data_provider.device_data_provider
        image_stream_ids = device_data_provider.get_image_stream_ids()
        timestamps = device_data_provider.get_sequence_timestamps()
        images = []
        for timestamp in timestamps:
            if timestamp < start_timestamp or timestamp > end_timestamp:
                continue
            image = device_data_provider.get_image(timestamp, image_stream_ids[0])
            images.append(np.rot90(image, -1))
        imageio.mimsave(save_path, images)
    
    def create_entire_video(self, left_grasp_timestamps, right_grasp_timestamps, save_path):
        device_data_provider = self.hot3d_data_provider.device_data_provider
        image_stream_ids = device_data_provider.get_image_stream_ids()
        timestamps = device_data_provider.get_sequence_timestamps()
        images = []
        for timestamp in tqdm(timestamps, desc=f'Video: {self.sequence}'):
            left_grasp = False
            right_grasp = False
            for left_count, value in left_grasp_timestamps.items():
                if value['start'] <= timestamp <= value['end']:
                    left_grasp = True
                    break
            for right_count, value in right_grasp_timestamps.items():
                if value['start'] <= timestamp <= value['end']:
                    right_grasp = True
                    break
            image = device_data_provider.get_image(timestamp, image_stream_ids[0])
            image = np.rot90(image.copy(), -1).copy()
            if left_grasp and right_grasp:
                cv2.putText(image, f"left grasp {left_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                cv2.putText(image, f"right grasp {right_count}", (image.shape[1] - 500, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif left_grasp:
                cv2.putText(image, f"left grasp {left_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            elif right_grasp:
                cv2.putText(image, f"right grasp {right_count}", (image.shape[1] - 500, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            images.append(image)
        imageio.mimsave(save_path, images)
        print(f'Video saved at {save_path}')


if __name__ == '__main__':
    hot = HOT3DUTILS(sequence='P0002_016222d1')
    objects_pose, hands_pose = hot.get_poses_at_timestamp(hot.timestamps[540])
    closest_objects = hot.get_closest_objects_v2(hot.timestamps[540])
    selected_index, object_name = hot.get_contact_points(
        timestamp_ns=hot.timestamps[540],
        hand_side='left',
        closeness_threshold=0.1,
        closest_objects=closest_objects,
        use_approx=True,
    )
    # closest_objects_old = hot.get_closest_objects(hot.timestamps[340])
    breakpoint()
