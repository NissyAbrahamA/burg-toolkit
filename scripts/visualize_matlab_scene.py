import os

import argparse
import configparser
import burg_toolkit as burg

parser = argparse.ArgumentParser(description='visualize a scene generetad with MATLAB scene generator project')
parser.add_argument('-c', '--config_fn', default='../config/config.cfg', type=str, metavar='FILE',
                    help='path to config file')

print('hi, it''s me, visualize_matlab_scene.py')

# read config file
cfg_fn = parser.parse_args().config_fn
print('using config file in:', os.path.abspath(cfg_fn))

cfg = configparser.ConfigParser()
cfg.read(cfg_fn)

# object lib
print('read object library')
object_library = burg.io.read_object_library(cfg['General']['object_lib_fn'])
print('found', len(object_library), 'objects')

# read the meshes as point clouds
print('reading object meshes and converting to point cloud')
mesh_fns = [
    os.path.join(
        cfg['General']['models_dir'],
        obj.name +
        cfg['General']['mesh_fn_ext']
    ) for obj in object_library
]
point_clouds = burg.mesh_processing.convert_mesh_to_point_cloud(mesh_fns, with_normals=True)

# add them to object info
for obj, pc in zip(object_library, point_clouds):
    obj.point_cloud = pc

# read bg_obj point cloud
print('reading table mesh and converting to point cloud')
table_path = os.path.join(
    cfg['General']['bg_models_dir'],
    cfg['General']['table_fn']
)
table_scale_factor = float(cfg['General']['table_scale_factor'])
table_pc = burg.mesh_processing.convert_mesh_to_point_cloud(table_path, with_normals=True, scale_factor=table_scale_factor)

# get file names of scene data
file_names = burg.io.get_scene_filenames(cfg['General']['scenes_dir'])

# pick one and read them
files = file_names[0]
print('loading scene data from the following files:')
print('\theap:', os.path.abspath(files['heap_fn']))
print('\timages:', os.path.abspath(files['image_data_fn']))
scene = burg.io.read_scene_files(files)
print('scene has', len(scene.objects), 'objects and', len(scene.views), 'views')

scene.bg_objects[0].point_cloud = table_pc

# visualize point cloud
print('visualizing scene point cloud')
burg.visualization.show_aligned_scene_point_clouds(scene, scene.views, object_library)

