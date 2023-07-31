import os
from timeit import default_timer as timer
import numpy as np
import configparser

import open3d as o3d

import burg_toolkit as burg

SAVE_FILE = os.path.join('..', 'sampled_grasps.npy')


def test_distance_and_coverage():
    # testing the distance function
    initial_translations = np.random.random((50, 3))
    gs = burg.grasp.GraspSet.from_translations(initial_translations)

    theta = 0 / 180 * np.pi
    rot_mat = np.asarray([[1, 0, 0],
                          [0, np.cos(theta), -np.sin(theta)],
                          [0, np.sin(theta), np.cos(theta)]])

    grasp = gs[0]
    grasp.translation = np.asarray([0, 0, 0.003])
    grasp.rotation_matrix = rot_mat
    gs[0] = grasp
    print(grasp)

    theta = 15 / 180 * np.pi
    rot_mat = np.asarray([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]])

    grasp = gs[1]
    grasp.translation = np.asarray([0, 0, 0])
    grasp.rotation_matrix = rot_mat
    gs[1] = grasp
    print(grasp)

    print('average gripper point distances between 20 and 50 elem graspset')
    print(burg.metrics.avg_gripper_point_distances(gs[0:20], gs).shape)

    dist = burg.metrics.combined_distances(gs[0], gs[1])
    print('computation of pairwise_distances (15 degree and 3 mm)', dist.shape, dist)

    t1 = timer()
    print('computation of coverage 20/50:', burg.metrics.coverage_brute_force(gs, gs[0:20]))
    print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.metrics.coverage(gs, gs[0:20], print_timings=True))
    print('this took:', timer() - t1, 'seconds')

    grasp_folder = 'e:/datasets/21_ycb_object_grasps/'
    grasp_file = '061_foam_brick/grasps.h5'
    grasp_set, com = burg.io.read_grasp_file_eppner2019(os.path.join(grasp_folder, grasp_file))

    t1 = timer()
    # this is unable to allocate enough memory for len(gs)=500
    #print('computation of coverage 20/50:', gdt.grasp.coverage_brute_force(grasp_set, gs))
    #print('this took:', timer() - t1, 'seconds')

    t1 = timer()
    print('coverage kd-tree:', burg.metrics.coverage(grasp_set, gs, print_timings=True))
    print('in total, this took:', timer() - t1, 'seconds')



def test_new_antipodal_grasp_sampling():
    redo = []
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.03,
                                                    finger_thickness=0.003)
    with open("objects.txt", "r") as file:
        object_names = file.read().splitlines()

    for object_name in object_names:
        mesh_fn = f"C:/Users/anizy/OneDrive - Aston University/Desktop/Dissertation/pointcloud/models/ycb/{object_name}/google_16k/textured.obj"

        #mesh_fn= '../input/002_master_chef_can_pointcloud.ply'
        #obj_name = os.path.splitext(os.path.basename(mesh_fn))[0]
        #mesh = o3d.io.read_triangle_mesh(mesh_fn)
        #n_sample = np.max([500, 1000, len(mesh.triangles)])
        #pc_with_normals = burg.util.o3d_pc_to_numpy(burg.mesh_processing.poisson_disk_sampling(mesh, n_points=n_sample))
        #pc_with_normals = burg.util.o3d_pc_to_numpy(burg.mesh_processing.poisson_disk_sampling(mesh))
        #np.save(object_name + '_pc.npy', pc_with_normals)

        ags = burg.sampling.AntipodalGraspSampler()
        ags.mesh = burg.io.load_mesh(mesh_fn)
        ags.gripper = gripper_model
        ags.n_orientations = 1
        ags.verbose = True
        ags.max_targets_per_ref_point = 1
        graspset, contacts, pc_with_normals = ags.sample(500)
        np.save(object_name + '_pc.npy', pc_with_normals)
        #print(contacts[0])
        #contacts_with_mscore = np.copy(contacts)
        # for c in contacts:
        #     nc = np.concatenate((c[:2], np.expand_dims(c[2], axis=0)))
        #     contacts_with_mscore.append(nc)
        # contacts_with_mscore = np.array(contacts_with_mscore)
        #contacts_with_mscore[:, 2:5] = contacts_with_mscore[:, 2:3]
        # contacts_with_mscore = np.array([list(np.unique(row)) for row in contacts])
        # print(contacts_with_mscore[0])
        graspset.scores = ags.check_collisions(graspset, use_width=False)  # need to install python-fcl
        filtered_grasps_contacts = {}
        filtered_grasps_contacts['points'] = []
        filtered_grasps_contacts['normals'] = []
        filtered_grasps_contacts['angles'] = []
        filtered_grasps_contacts['score'] = []
        for grasp, score, contact,normals,angles,cp_score in zip(graspset, graspset.scores, contacts['points'],contacts['normals'],contacts['angles'],contacts['score']):
            if score == 1.0:
                filtered_grasps_contacts['points'].append(contact)
                filtered_grasps_contacts['normals'].append(normals)
                filtered_grasps_contacts['angles'].append(angles)
                filtered_grasps_contacts['score'].append(cp_score)

        if len(filtered_grasps_contacts['points']) < 90:
            redo.append(object_name)
        print(redo)
        np.savez(object_name + "_mu0.1.npz", **filtered_grasps_contacts)

def test_antipodal_grasp_random_sampling():
    redo = []
    gripper_model = burg.gripper.ParallelJawGripper(finger_length=0.03,
                                                    finger_thickness=0.003)
    with open("objects.txt", "r") as file:
        object_names = file.read().splitlines()

    for object_name in object_names:
        mesh_fn = f"C:/Users/anizy/OneDrive - Aston University/Desktop/Dissertation/pointcloud/models/ycb/{object_name}/google_16k/textured.obj"
        ags = burg.sampling.AntipodalGraspSampler()
        ags.mesh = burg.io.load_mesh(mesh_fn)
        ags.gripper = gripper_model
        ags.n_orientations = 1
        ags.verbose = True
        ags.max_targets_per_ref_point = 1
        graspset, contacts = ags.randomsample(5000)
        graspset.scores = ags.check_collisions(graspset, use_width=False)
        filtered_grasps_contacts = {}
        filtered_grasps_contacts['points'] = []
        filtered_grasps_contacts['normals'] = []
        filtered_grasps_contacts['angles'] = []
        filtered_grasps_contacts['score'] = []
        for grasp, score, contact,normals,angles,cp_score in zip(graspset, graspset.scores, contacts['points'],contacts['normals'],contacts['angles'],contacts['score']):
            if score == 1.0:
                filtered_grasps_contacts['points'].append(contact)
                filtered_grasps_contacts['normals'].append(normals)
                filtered_grasps_contacts['angles'].append(angles)
                filtered_grasps_contacts['score'].append(cp_score)

        if len(filtered_grasps_contacts['points']) < 1500:
            redo.append(object_name)
        print(redo)
        np.savez(object_name + "_random.npz", **filtered_grasps_contacts)



def test_rotation_to_align_vectors():
    vec_a = np.array([1, 0, 0])
    vec_b = np.array([0, 1, 0])
    r = burg.util.rotation_to_align_vectors(vec_a, vec_b)
    print('vec_a', vec_a)
    print('vec_b', vec_b)
    print('R*vec_a', np.dot(r, vec_a.reshape(3, 1)))

    vec_a = np.array([1, 0, 0])
    vec_b = np.array([-1, 0, 0])
    r = burg.util.rotation_to_align_vectors(vec_a, vec_b)
    print('vec_a', vec_a)
    print('vec_b', vec_b)
    print('R*vec_a', np.dot(r, vec_a.reshape(3, 1)))


def show_grasp_pose_definition():
    gs = burg.grasp.GraspSet.from_translations(np.asarray([0, 0, 0]).reshape(-1, 3))
    gripper = burg.gripper.ParallelJawGripper(finger_length=0.03, finger_thickness=0.003, opening_width=0.05)
    burg.visualization.show_grasp_set([o3d.geometry.TriangleMesh.create_coordinate_frame(0.02)],
                                      gs, gripper=gripper)


def test_angles():
    vec_a = np.array([1, 0, 0])
    vec_b = np.array([-1, 0, 0])
    mask = np.array([0])
    print(mask)
    a = burg.util.angle(vec_a, vec_b, sign_array=mask)
    print(a)
    print(mask)


def test_cone_sampling():
    axis = [0, 1, 0]
    angle = np.pi/4
    rays = burg.sampling.rays_within_cone(axis, angle, n=100)

    print(rays.shape)


def visualise_perturbations():
    positions = np.zeros((3, 3))
    positions[0] = [0.3, 0, 0]
    positions[1] = [0, 0, 0.3]

    gs = burg.grasp.GraspSet.from_translations(positions)
    gs_perturbed = burg.sampling.grasp_perturbations(gs, radii=[5, 10, 15])
    gripper = burg.gripper.ParallelJawGripper(finger_length=0.03, finger_thickness=0.003, opening_width=0.05)
    burg.visualization.show_grasp_set([], gs_perturbed, gripper=gripper)

def test_scoring_for_grippers():
    mesh_fn = "C:/Users/anizy/OneDrive - Aston University/Desktop/Dissertation/pointcloud/models/ycb/002_master_chef_can/google_16k/textured.obj"
    mesh = burg.io.load_mesh(mesh_fn)
    cp1 = np.array([0.00910593, 0.00076649, 0.01397331])
    cp2 = np.array([-0.03230069,  0.03238435,  0.02414111])
    normal1 = np.array([0.34267149, - 0.89084227, - 0.29828896])
    normal2 = np.array([-0.83048282,  0.53533904,  0.15398183])
    d = (cp2-cp1)
    print(d)
    angle_cp1, angle_cp2, score = burg.util.calc_score(d,normal1, normal2)
    print(angle_cp2)
    print(angle_cp1)
    print('score:' + str(score))
    burg.visualization.plot_contacts_normals(mesh,cp1,cp2,normal1,normal2)



if __name__ == "__main__":
    print('hi')
    # test_distance_and_coverage()
    # test_rotation_to_align_vectors()
    #test_angles()
    # test_cone_sampling()
    #test_new_antipodal_grasp_sampling()
    #show_grasp_pose_definition()
    #visualise_perturbations()
    test_scoring_for_grippers()
    #test_antipodal_grasp_random_sampling()
    print('bye')
