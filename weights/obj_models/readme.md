# Note on reducing number of vertices of an mesh


We use Blender's `decimate` function to reduce the number of vertices.
Before decimation, we use `3D-print` toolbox to process the mesh to be watertight.

We roughly reduce the number of vertices to ~500.

Note during blender exporting, we de-select `normal` export, choose up-axis to be Z-axis.