"""
As of February 2025, Open3D's offscreen rendering has an exit issue 
    where the program crashes after ~11,000 calls to add_geometry(). 
    To prevent this, users must manually call inc_render_usage_counter(), 
    which resets the renderer every a few calls. 
Automatic management isn't possible since users may call add_geometry() multiple times, 
    and resetting mid-render could cause crashes.

Note-2 [Optional]:
    To avoid SIGSEGV at the end of the program,
    set render = None.
"""
from open3d.visualization import rendering
import gc

global_o3d_render = None
global_o3d_render_usage_counter = 0

def create_global_o3d_render():
    global global_o3d_render
    out_sizes = {'small': (640, 480), 'middle': (800, 480), 'HD': (1920, 1080)}
    # out_size = out_sizes['middle']
    out_size = out_sizes['small']
    global_o3d_render = rendering.OffscreenRenderer(*out_size)
    return global_o3d_render

def get_global_o3d_render():
    global global_o3d_render
    if global_o3d_render is None:
        global_o3d_render = create_global_o3d_render()
    return global_o3d_render

def inc_render_usage_counter():
    """ Increment the global counter and reset the renderer every 64 times
    This function should be called every time add_geometry() is called.

    Usage:
    ```
    render = get_global_o3d_render()
    ... # Use render
    render = None # Must set to None before calling inc_render_usage_counter()
    inc_render_usage_counter()
    ```

    """
    RESET_COUNT = 64 # 1024
    global global_o3d_render
    global global_o3d_render_usage_counter
    global_o3d_render_usage_counter += 1
    if global_o3d_render_usage_counter % RESET_COUNT == 0:
        if global_o3d_render is not None:
            del global_o3d_render
            gc.collect()
        global_o3d_render_usage_counter = 0
        global_o3d_render = create_global_o3d_render()
    return global_o3d_render