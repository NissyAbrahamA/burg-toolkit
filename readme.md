# grasp data toolkit

This becomes a Python toolkit for benchmarking and evaluating methods for robotic grasping.
It is based on scenes generated with the sceneGeneration_MATLAB project.

## overview

The project contains:
- **grasp_data_toolkit** - the core Python library, used for io, mesh and point cloud processing, data visualization, etc.
- **scripts** - entry points, scripts for exploring the data, compiling datasets, evaluation
- **config** - configuration files, including e.g. important paths, which meshes to use, scale factors, etc.

## installation

install dependencies by running:
``
pip install -r requirements.txt
``

## plans for the project
### todos
- have a script to export (segmented, partial) point clouds

### longer-term todos:
- move all point clouds to o3d
    - in object_library we could already keep the o3d point clouds, which should save some processing
    - we could also save the point clouds to files, which saves some waiting time during each run
- poisson disk sampling:
    - point densities are not uniform, instead it is relative to the size of the object
    - i think uniform density would be better, but we can skip some part of the table
    - o3d has a voxel-based down-sample method - maybe that could be an approach?