import numpy as np
import open3d as o3d
import os


# point_cloud_array = np.load('C:/Users/anizy/OneDrive - Aston University/Documents/fork/burg-toolkit/tests/dataset/005_tomato_soup_can_pc.npy')
# point_coordinates = point_cloud_array[:, :3]
# normals = point_cloud_array[:, 3:]
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(point_coordinates)
# point_cloud.normals = o3d.utility.Vector3dVector(normals)
# o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
data = np.load('C:/Users/anizy/OneDrive - Aston University/Documents/fork/burg-toolkit/tests/dataset/002_master_chef_can.npz')
# data = np.load('005_tomato_soup_can_mu0.1.npz')
print(max(data['score']))
# print(data['points'][0])
# print(data['normals'][0])
# print(data['angles'][0])
# print(data['score'][0])
# #
# with open("objects.txt", "r") as file:
#     object_names = file.read().splitlines()
# print(object_names)
#
# for object_name in object_names:
#     print(object_name)
#     npz_files = [
#         f'{object_name}_random.npz',
#         f'{object_name}_mu0.5.npz',
#         f'{object_name}_mu0.4.npz',
#         f'{object_name}_mu0.3.npz',
#         f'{object_name}_mu0.2.npz',
#         f'{object_name}_mu0.1.npz'
#     ]
#
#     all_data = []
#     for npz_file in npz_files:
#         print(npz_file)
#         data = np.load(npz_file)
#         print(len(data['score']))
#         all_data.append(data)
#
#         merged_data = {}
#         for data in all_data:
#             for key, value in data.items():
#                 if key in merged_data:
#                     merged_data[key] = np.concatenate((merged_data[key], value), axis=0)
#                 else:
#                     merged_data[key] = value
#                 output_dir = 'C:/Users/anizy/OneDrive - Aston University/Documents/fork/burg-toolkit/tests/output'
#
#         output_file = os.path.join(output_dir, f'{object_name}.npz')
#         np.savez(output_file, **merged_data)
