import logging
import os
import tempfile
import copy
from collections import UserDict

import numpy as np
import yaml
import pybullet as p

from . import io
from . import mesh_processing


class ObjectType:
    """
    Describes an Object Type.
    Needs an identifier and a mesh, the latter can be provided either directly or as filename.
    Thumbnail, VHACD and URDF can be created from that.

    :param identifier: object identifier as string
    :param name: name of the object type as string
    :param mesh: open3d.geometry.TriangleMesh associated with the object (leave blank if filename provided)
    :param mesh_fn: filename where to find the mesh
    :param thumbnail_fn: filename of an image of the object
    :param vhacd_fn: filename of the vhacd mesh of the object
    :param urdf_fn: filename of the urdf file of the object
    :param mass: mass of object in kg (defaults to 0, which means fixed in space in simulations)
    :param friction_coeff: friction coefficient, defaults to 0.24
    """
    def __init__(self, identifier, name=None, mesh=None, mesh_fn=None, thumbnail_fn=None, vhacd_fn=None, urdf_fn=None,
                 mass=None, friction_coeff=None):
        self.identifier = identifier
        self.name = name
        if mesh is not None and mesh_fn is not None:
            raise ValueError('Cannot create ObjectType if both mesh and mesh_fn are given. Choose one.')
        if mesh is None and mesh_fn is None:
            raise ValueError('Cannot create ObjectType with no mesh - must provide either mesh or mesh_fn.')
        self._mesh = mesh
        self.mesh_fn = mesh_fn
        self.thumbnail_fn = thumbnail_fn
        self.vhacd_fn = vhacd_fn
        self.urdf_fn = urdf_fn
        self.mass = mass or 0
        self.friction_coeff = friction_coeff or 0.24

        # todo: stable poses, stable poses probs

    @property
    def mesh(self):
        """Loads the mesh from file the first time it is used."""
        if self._mesh is None:
            self._mesh = io.load_mesh(mesh_fn=self.mesh_fn)
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

    # todo: method to save the current mesh to a new, given mesh_fn

    def generate_vhacd(self, vhacd_fn):
        """
        Generates an approximate convex decomposition of the object's mesh and stores it in given filename.

        :param vhacd_fn: Where to store the vhacd mesh. Will be set as property in this object type.
        """
        mesh_file = self.mesh_fn
        # mesh file could be None if object type has been given a mesh directly
        # will store in a temporary file then
        tmp_mesh_file, tmp_mesh_file_handle = None, None
        if mesh_file is None:
            tmp_mesh_file_handle, tmp_mesh_file = tempfile.mkstemp(suffix='.obj', text=True)
            io.save_mesh(tmp_mesh_file, self._mesh)
            mesh_file = tmp_mesh_file

        log_file_handle, log_file = tempfile.mkstemp(suffix='.log', text=True)

        # more documentation on this here: https://github.com/kmammou/v-hacd
        p.connect(p.DIRECT)
        p.vhacd(mesh_file, vhacd_fn, log_file)
        p.disconnect()

        self.vhacd_fn = vhacd_fn

        # dump the temporary files
        os.close(log_file_handle)
        os.remove(log_file)

        if tmp_mesh_file_handle is not None:
            os.close(tmp_mesh_file_handle)
        if tmp_mesh_file is not None:
            os.remove(tmp_mesh_file)

    def generate_urdf(self, urdf_fn, use_vhacd=True):
        """
        Method generates the urdf file for this object type, to be used in simulation.
        It needs to link to a mesh file. This can be either the original mesh or the VHACD. Usually we want this
        to be the VHACD. If it does not exist yet, it will be created in the same dir as the `urdf_fn`.

        :param urdf_fn: Path for the urdf to be generated (directory must exist).
        :param use_vhacd: Whether to use the vhacd (True, default) or the actual mesh (False).
        """
        logging.debug(f'generating urdf for {self.identifier}')
        name = self.identifier
        mass = self.mass
        origin = [0, 0, 0]  # mesh origin will be placed at origin when loading urdf in simulation
        inertia, com = mesh_processing.compute_mesh_inertia(self.mesh, mass)
        logging.debug(f'inertia: {inertia}')
        logging.debug(f'com: {com}')
        logging.debug(f'center: {self.mesh.get_center()}')

        if use_vhacd:
            if self.vhacd_fn is None:
                # we can just generate the vhacd here, assume same path as urdf
                vhacd_fn = os.path.join(os.path.dirname(urdf_fn), f'{name}_vhacd.obj')
                logging.debug(f'creating VHACD at {vhacd_fn}')
                self.generate_vhacd(vhacd_fn)
            rel_mesh_fn = os.path.relpath(self.vhacd_fn, os.path.dirname(urdf_fn))
        else:
            if self.mesh_fn is None:
                raise ValueError('ObjectType has no mesh_fn, but need mesh_fn linked in urdf (or use_vhacd)')
            rel_mesh_fn = os.path.relpath(self.mesh_fn, os.path.dirname(urdf_fn))

        io.save_urdf(urdf_fn, rel_mesh_fn, name, origin, inertia, com, mass)
        self.urdf_fn = urdf_fn

    def __str__(self):
        # todo: add other properties
        return f'ObjectType: {self.identifier}\n\thas mesh: {self.mesh is not None}\n\tmass: {self.mass}\n\t' + \
            f'friction: {self.friction_coeff}'


class ObjectInstance:
    """
    Describes an instance of an object type in the object library and a pose.

    :param object_type: an ObjectType referring to the type of this object instance
    :param pose: (4, 4) np array - homogenous transformation matrix
    """

    def __init__(self, object_type, pose=None):
        self.object_type = object_type
        if pose is None:
            self.pose = np.eye(4)
        else:
            self.pose = pose

    def __str__(self):
        return f'instance of {self.object_type.identifier} object type. pose:\n{self.pose}'

    def get_mesh(self):
        """
        Returns a copy of the mesh of the object type in the pose of the instance.

        :return: open3d.geometry.TriangleMesh
        """
        if self.object_type.mesh is None:
            raise ValueError('no mesh associated with this object type')
        mesh = copy.deepcopy(self.object_type.mesh)
        mesh.transform(self.pose)
        return mesh


class ObjectLibrary(UserDict):
    """
    Contains a library of ObjectType objects and adds some convenience methods to it.
    Acts like a regular python dict.

    :param name: string, name of the ObjectLibrary.
    :param description: string, description of the ObjectLibrary.
    """
    def __init__(self, name=None, description=None):
        super().__init__()
        self.name = name or 'default library'
        self.description = description or 'no description available'

    @classmethod
    def from_yaml(cls, yaml_fn):
        """
        Loads an ObjectLibrary described in the specified yaml file.

        :param yaml_fn: Filename of the YAML file.

        :return: ObjectLibrary containing all the objects.
        """
        with open(yaml_fn, 'r') as stream:
            data = yaml.safe_load(stream)

        logging.debug(f'reading object library from {yaml_fn}')
        logging.debug(f'keys: {[key for key in data.keys()]}')

        library = cls(data['name'], data['description'])

        # need to prepend the base directory of library to all paths, since yaml stores relative paths
        lib_dir = os.path.dirname(yaml_fn)
        for item in data['objects']:
            # item is a dictionary, we can pass it to the ObjectType constructor by unpacking it
            obj = ObjectType(**item)

            # complete the paths of all files
            obj.mesh_fn = cls._get_abs_path(obj.mesh_fn, lib_dir)
            obj.vhacd_fn = cls._get_abs_path(obj.vhacd_fn, lib_dir)
            obj.urdf_fn = cls._get_abs_path(obj.urdf_fn, lib_dir)
            obj.thumbnail_fn = cls._get_abs_path(obj.thumbnail_fn, lib_dir)

            library[obj.identifier] = obj

        return library

    def to_yaml(self, yaml_fn):
        """
        Saves an ObjectLibrary to the specified yaml file.
        All object properties will be saved as well (although no direct changes to the mesh are saved).
        Paths will be made relative to the base directory of the yaml file.

        :param yaml_fn: Filename where to store the object library.
        """
        # create the dictionary structure
        lib_dict = {
            'name': self.name,
            'description': self.description,
            'objects': []
             }

        lib_dir = os.path.dirname(yaml_fn)
        for _, item in self.data.items():
            obj_dict = {
                'identifier': item.identifier,
                'name': item.name,
                'thumbnail_fn': self._get_rel_path(item.thumbnail_fn, lib_dir),
                'mesh_fn': self._get_rel_path(item.mesh_fn, lib_dir),
                'vhacd_fn': self._get_rel_path(item.vhacd_fn, lib_dir),
                'urdf_fn': self._get_rel_path(item.urdf_fn, lib_dir),
                'mass': item.mass,
                'friction_coeff': item.friction_coeff
            }
            lib_dict['objects'].append(obj_dict)

        with open(yaml_fn, 'w') as lib_file:
            yaml.dump(lib_dict, lib_file)

    @staticmethod
    def _get_abs_path(fn, base_dir):
        if fn is None:
            return None
        return os.path.join(base_dir, fn)

    @staticmethod
    def _get_rel_path(fn, base_dir):
        if fn is None:
            return None
        return os.path.relpath(fn, base_dir)

    def generate_urdf_files(self, directory, use_vhacd=True):
        """
        Calls the ObjectType's method to generate a urdf file for all object types in this library.

        :param directory: where to put the urdf files.
        :param use_vhacd: whether to link to vhacd meshes (True, default) or original meshes (False).
        """
        io.make_sure_directory_exists(directory)
        for name, obj in self.data.items():
            urdf_fn = os.path.join(directory, f'{obj.identifier}.urdf')
            obj.generate_urdf(urdf_fn=urdf_fn, use_vhacd=use_vhacd)

    def __len__(self):
        return len(self.data.keys())

    def __str__(self):
        return f'ObjectLibrary: {self.name}, {self.description}\nObjects:\n{[key for key in self.data.keys()]}'

    def print_details(self):
        print(f'ObjectLibrary:\n\t{self.name}\n\t{self.description}\nObjects:')
        for idx, (identifier, object_type) in enumerate(self.data.items()):
            print(f'{idx}: {object_type}')


class Scene:
    """
    contains all information about a scene
    """

    def __init__(self, objects=None, bg_objects=None, views=None):
        self.objects = objects or []
        self.bg_objects = bg_objects or []
        self.views = views or []