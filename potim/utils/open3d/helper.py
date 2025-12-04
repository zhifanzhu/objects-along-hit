from typing import List
import os
import numpy as np
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
from open3d.visualization import rendering

from potim.utils.open3d.o3d_line_mesh import LineMesh
from potim.utils.open3d.offscreen_manage import (
    create_global_o3d_render,
    get_global_o3d_render,
    inc_render_usage_counter)

# export them for easy usage (i.e. user don't have to import from offscreen_manage)
__all__ = [
    create_global_o3d_render, get_global_o3d_render, inc_render_usage_counter]


FREEMONO_PATH = os.path.join(os.path.dirname(__file__), 'FreeMono.ttf')


def get_material(color: str, point_size=None, shader="defaultUnlit") -> rendering.MaterialRecord:
    """
    Args:
        shader: e.g.'defaultUnlit', 'defaultLit', 'depth', 'normal',
            'unlitLine'
            see Open3D: cpp/open3d/visualization/rendering/filament/FilamentScene.cpp#L1109
    """
    base_colors = {
        'white': [1, 1, 1, 0.8],
        'red': [1, 0, 0, 1],
        'blue': [0, 0, 1,1],
        'light_blue': [0.65, 0.74, 0.86, 1],
        'green': [0, 1, 0,1],
        'yellow': [1, 1, 0,1],
        'purple': [0.2, 0.2, 0.8, 1],
        'cyan': [0, 1, 1]
    }
    material = rendering.MaterialRecord()
    material.shader = shader
    material.base_color = base_colors[color]
    if point_size is not None:
        material.point_size = point_size
    return material

def get_cam_pos(c2w: np.ndarray) -> np.ndarray:
     """ Get camera position in world coordinate system
     """
     cen = np.float32([0, 0, 0, 1])
     pos = c2w @ cen
     return pos[:3]


def get_frustum(c2w: np.ndarray,
                sz=0.2,
                camera_height=None,
                camera_width=None,
                frustum_color=[1, 0, 0]) -> o3d.geometry.LineSet:
    """
    Args:
        c2w: np.ndarray, 4x4 camera-to-world matrix
        sz: float, size (width) of the frustum
    Returns:
        frustum: o3d.geometry.TriangleMesh
    """
    cen = [0, 0, 0]
    wid = sz
    if camera_height is not None and camera_width is not None:
        hei = wid * camera_height / camera_width
    else:
        hei = wid
    tl = [wid, hei, sz]
    tr = [-wid, hei, sz]
    br = [-wid, -hei, sz]
    bl = [wid, -hei, sz]

    points = np.float32([cen, tl, tr, br, bl])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]]
    frustum_colors = [frustum_color for l in range(len(lines))]

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(frustum_colors)

    frustum = frustum.transform(c2w)
    return frustum

def get_trajectory(pos_history,
                   num_line=6,
                   line_radius=0.15
                   ) -> o3d.geometry.TriangleMesh:
    """ pos_history: absolute position history
    """
    pos_history = np.asarray(pos_history)[-num_line:]
    colors = [0, 0, 0.6]
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    line_mesh.merge_cylinder_segments()
    path = line_mesh.cylinder_segments[0]
    return path


def get_pretty_trajectory(pos_history,
                          num_line=6,
                          line_radius=0.15,
                          darkness=1.0,
                          ) -> List[o3d.geometry.TriangleMesh]:
    """ pos_history: absolute position history
    """
    def generate_jet_colors(n, darkness=0.6):
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0, vmax=n-1)
        colors = cmap(norm(np.arange(n)))
        # Convert RGBA to RGB
        colors_rgb = []
        for color in colors:
            colors_rgb.append(color[:3] * darkness)

        return colors_rgb

    pos_history = np.asarray(pos_history)[-num_line:]
    colors = generate_jet_colors(len(pos_history), darkness)
    line_mesh = LineMesh(
        points=pos_history, 
        colors=colors, radius=line_radius)
    return line_mesh.cylinder_segments

def update_render_geom(render, geom, name, material):
    if render.scene.has_geometry(name):
        render.scene.remove_geometry(name)
    render.scene.add_geometry(name, geom, material)

def update_render_geom_from_trimesh(render, tri_mesh, name, material):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    m.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    m.compute_vertex_normals()
    update_render_geom(render, m, name, material)


""" Obtain Viewpoint from Open3D GUI """
def parse_o3d_gui_view_status(status: dict, render: rendering.OffscreenRenderer):
    """ Parse open3d GUI's view status and convert to OffscreenRenderer format.
    This will do the normalisation of front and compute eye vector (updated version of front)

    
    Args:
        status: Ctrl-C output from Open3D GUI
        render: OffscreenRenderer
    Output:
       params for render.setup_camera(fov, lookat, eye, up) 
    """
    cam_info = status['trajectory'][0]
    fov = cam_info['field_of_view']
    lookat = np.asarray(cam_info['lookat'])
    front = np.asarray(cam_info['front'])
    front = front / np.linalg.norm(front)
    up = np.asarray(cam_info['up'])
    zoom = cam_info['zoom']
    """ 
    See Open3D/cpp/open3d/visualization/visualizer/ViewControl.cpp#L243: 
        void ViewControl::SetProjectionParameters()
    """
    right = np.cross(up, front) / np.linalg.norm(np.cross(up, front))
    view_ratio = zoom * render.scene.bounding_box.get_max_extent()
    distance = view_ratio / np.tan(fov * 0.5 / 180.0 * np.pi)
    eye = lookat + front * distance
    return fov, lookat, eye, up


def set_offscreen_as_gui(render: rendering.OffscreenRenderer, status: dict):
    """ Set offscreen renderer as GUI's view status
    """
    fov, lookat, eye, up = parse_o3d_gui_view_status(status, render)
    render.setup_camera(fov, lookat, eye, up)