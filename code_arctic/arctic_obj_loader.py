from typing import NamedTuple
from hydra.utils import to_absolute_path
import trimesh
import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import axis_angle_to_quaternion, quaternion_apply
from libzhifan import io
from libzhifan.geometry import SimpleMesh


""" Update 2023/09/09
Use blender to simplify mesh.

- Select non-manifold => F => Cmd-T 
- Select non-manifold => X => collapse
- Add modifier => Decimate (planar / collapse), collapse ratio ~0.05-0.3


In arctic, object model has top and bottom, we process them separately so that we can combine them later.
Each ARCTIC object := top_triangles + bottom_triangles + connecting_triangles, each is not watertight, but the origianl combination is watertight.
Hence, we use Blender to make top and bottom mesh manifold(watertight) and reduced.
Then we combine top_reduce, bot_reduce, and connecting_triangles.
"""

class MeshHolder(NamedTuple):
    vertices: torch.Tensor
    faces: torch.Tensor


def forward_obj_arti(vo, top_idx, arti: torch.Tensor):
    """
    vo: (N, V, 3)
    top_idx: (V,)
    arti: (N, 1)
    """
    vo = torch.as_tensor(vo).clone()  # (N, V, 3)
    Z_AXIS = torch.tensor([0, 0, -1], dtype=torch.float).view(1, 3)
    angles = arti.view(-1, 1)
    quat_arti = axis_angle_to_quaternion(Z_AXIS * angles)  # [N, 4]
    v_top_articulated = quaternion_apply(quat_arti[:, None, :], vo[:, top_idx, ...])  # (N, VT, 3)
    vo[:, top_idx, ...] = v_top_articulated
    return vo


class ArcticOBJLoader:
    original_mesh_fmt = to_absolute_path('DATA_STORAGE/arctic_data/meta/object_vtemplates/{obj_name}/mesh.obj')
    original_parts_fmt = to_absolute_path('DATA_STORAGE/arctic_data/meta/object_vtemplates/{obj_name}/parts.json')

    reduced_top_fmt= to_absolute_path(f'./weights/obj_models/blender_export/{{obj_name}}_top_reduce.obj')
    reduced_bot_fmt= to_absolute_path(f'./weights/obj_models/blender_export/{{obj_name}}_bot_reduce.obj')

    def __init__(self, 
                 version: str):
        """
        """
        self.obj_models_cache = dict()
        self.top_idx_cache = dict()
        self.version = version
        assert version in {'original', 'reduced'}
    
    def load_obj_unarticulated(self, name, return_mesh=False):
        """ Load the mesh without articulation.
        """
        if name not in self.obj_models_cache:
            mesh = trimesh.load(self.original_mesh_fmt.format(obj_name=name))
            vo_orig = np.asarray(mesh.vertices, dtype=np.float32) # / 1000
            parts_ids = io.read_json(self.original_parts_fmt.format(obj_name=name))
            PART_TOP_ID = 0
            vo_top_idx  = np.asarray(parts_ids) == PART_TOP_ID
            fo = mesh.faces

            if self.version == 'original':
                vo_orig = vo_orig / 1000
                obj_mesh = MeshHolder(vertices=vo_orig, faces=mesh.faces)
                self.obj_models_cache[name] = obj_mesh
                self.top_idx_cache[name] = vo_top_idx
            elif self.version == 'reduced':
                verts = mesh.vertices
                top_vis = np.arange(len(verts))[vo_top_idx]
                top_reduce = trimesh.load_mesh(self.reduced_top_fmt.format(obj_name=name))
                bot_reduce = trimesh.load_mesh(self.reduced_bot_fmt.format(obj_name=name))

                top_faces, bot_faces, con_faces = separate_faces(len(verts), fo, vo_top_idx)
                con_mesh, con_vmap = regenerate_mesh(mesh.vertices, con_faces)
                con_top_vis = np.asarray([con_vmap[v] for v in top_vis if v in con_vmap])
                mesh_reduce, top_idx_reduce = recombine_meshes(
                    top_mesh=top_reduce, bot_mesh=bot_reduce, con_mesh=con_mesh, con_top_vinds=con_top_vis,
                    process=False)
                vertices = np.asarray(mesh_reduce.vertices, dtype=np.float32)
                vertices = vertices / 1000
                self.obj_models_cache[name] = MeshHolder(
                    vertices=vertices, faces=mesh_reduce.faces)
                self.top_idx_cache[name] = top_idx_reduce

        return self.obj_models_cache[name], self.top_idx_cache[name]
        
    def load_obj(self, name, arti: float, return_mesh=False, tex_color='light_blue'):
        """
        Args:
            name: str
            arti: float, ARCTIC dataset has additional articulation DoF
        """
        obj_mesh, top_idx = self.load_obj_unarticulated(name, return_mesh=False)
        vo = torch.from_numpy(obj_mesh.vertices).view(1, -1, 3)
        vo = forward_obj_arti(vo, top_idx, torch.as_tensor([arti])).view(-1, 3)

        if return_mesh:
            obj_mesh = SimpleMesh(vo, obj_mesh.faces, tex_color=tex_color)
            return obj_mesh
        else:
            return MeshHolder(vertices=vo, faces=obj_mesh.faces)
    
    def batch_articulate(self, name, artis: torch.Tensor):
        """
        Args:
            artis: (N,) list of articulation

        Returns:
            verts: (N, V, 3)
            faces: (F, 3)
        """
        N = len(artis)
        obj_mesh, top_idx = self.load_obj_unarticulated(name, return_mesh=False)
        vo = torch.from_numpy(obj_mesh.vertices).view(1, -1, 3).repeat(N, 1, 1)
        vo = forward_obj_arti(vo, top_idx, artis.view(N, 1)) # (N, V, 3)
        return vo, obj_mesh.faces

    
    # def load_obj_by_name_frame(self, name, frame_idx: int, return_mesh=False):
    #     pass


def separate_faces(num_verts, faces, top_idx, return_index=False):
    top_vinds = set(np.arange(num_verts)[top_idx])
    bottom_vinds = set(np.arange(num_verts)[~top_idx])
    top_faces = []
    bot_faces = []
    con_faces = []
    top_fi, bot_fi, con_fi = [], [], []
    for i, (i1, i2, i3) in enumerate(faces):
        if (i1 in top_vinds and i2 in top_vinds and i3 in top_vinds):
            top_fi.append(i)
            top_faces.append([i1, i2, i3])
        elif (i1 in bottom_vinds and i2 in bottom_vinds and i3 in bottom_vinds):
            bot_fi.append(i)
            bot_faces.append([i1, i2, i3])
        else:
            con_fi.append(i)
            con_faces.append([i1, i2, i3])
    if return_index:
        return map(np.asarray, (top_fi, bot_fi, con_fi))
    return np.asarray(top_faces), np.asarray(bot_faces), np.asarray(con_faces)


def regenerate_mesh(verts, faces):
    """
    faces are a subset of the faces of the original mesh.
    """
    new_verts = []
    new_faces = []
    v_map = {}
    for fi, (vi1, vi2, vi3) in enumerate(faces):
        if vi1 not in v_map:
            v_map[vi1] = len(new_verts)
            new_verts.append(verts[vi1])
        if vi2 not in v_map:
            v_map[vi2] = len(new_verts)
            new_verts.append(verts[vi2])
        if vi3 not in v_map:
            v_map[vi3] = len(new_verts)
            new_verts.append(verts[vi3])
        new_faces.append([v_map[vi1], v_map[vi2], v_map[vi3]])
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces)
    return new_mesh, v_map


def recombine_meshes(top_mesh, bot_mesh, con_mesh, con_top_vinds,
                     process=False):
    """
    Assuming top_mesh, bot_mesh, con_mesh have no common vertices.

    Returns:
        new_mesh: the combined mesh.
        top_idx: the indices of the top part of the new_mesh.
    """
    new_verts = []
    new_faces = []
    new_top_idx = []
    top_v_map = {}
    bot_v_map = {}
    con_v_map = {}
    con_top_vinds = set(con_top_vinds)
    if top_mesh is not None:
        for fi, (vi1, vi2, vi3) in enumerate(top_mesh.faces):
            if vi1 not in top_v_map:
                top_v_map[vi1] = len(new_verts)
                new_verts.append(top_mesh.vertices[vi1])
                new_top_idx.append(top_v_map[vi1])
            if vi2 not in top_v_map:
                top_v_map[vi2] = len(new_verts)
                new_verts.append(top_mesh.vertices[vi2])
                new_top_idx.append(top_v_map[vi2])
            if vi3 not in top_v_map:
                top_v_map[vi3] = len(new_verts)
                new_verts.append(top_mesh.vertices[vi3])
                new_top_idx.append(top_v_map[vi3])
            new_faces.append([top_v_map[vi1], top_v_map[vi2], top_v_map[vi3]])
    if bot_mesh is not None:
        for fi, (vi1, vi2, vi3) in enumerate(bot_mesh.faces):
            if vi1 not in bot_v_map:
                bot_v_map[vi1] = len(new_verts)
                new_verts.append(bot_mesh.vertices[vi1])
            if vi2 not in bot_v_map:
                bot_v_map[vi2] = len(new_verts)
                new_verts.append(bot_mesh.vertices[vi2])
            if vi3 not in bot_v_map:
                bot_v_map[vi3] = len(new_verts)
                new_verts.append(bot_mesh.vertices[vi3])
            new_faces.append([bot_v_map[vi1], bot_v_map[vi2], bot_v_map[vi3]])
    if con_mesh is not None:
        for fi, (vi1, vi2, vi3) in enumerate(con_mesh.faces):
            if vi1 not in con_v_map:
                con_v_map[vi1] = len(new_verts)
                new_verts.append(con_mesh.vertices[vi1])
                if vi1 in con_top_vinds:
                    new_top_idx.append(con_v_map[vi1])
            if vi2 not in con_v_map:
                con_v_map[vi2] = len(new_verts)
                new_verts.append(con_mesh.vertices[vi2])
                if vi2 in con_top_vinds:
                    new_top_idx.append(con_v_map[vi2])
            if vi3 not in con_v_map:
                con_v_map[vi3] = len(new_verts)
                new_verts.append(con_mesh.vertices[vi3])
                if vi3 in con_top_vinds:
                    new_top_idx.append(con_v_map[vi3])
            new_faces.append([con_v_map[vi1], con_v_map[vi2], con_v_map[vi3]])
    new_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=process)
    new_top_idx = np.asarray(new_top_idx)
    return new_mesh, new_top_idx