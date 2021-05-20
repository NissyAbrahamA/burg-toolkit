import numpy as np
import quaternion


class Grasp:
    """
    Class representing a grasp as the pose in SE(3) in which the gripper attempts to close.
    For efficiency, all information will be stored in an internal numpy array and can be retrieved via property
    functions.
    If necessary, the internal numpy array can also be used, but this has to be done with caution as dimensions might
    change in the future.

    :param np_array: optional, the internal numpy array which is structured as follows:
                     [translation(3), rotation_matrix(3x3)[:], score], length = ARRAY_LEN
    """
    ARRAY_LEN = 13

    def __init__(self, np_array=None):
        if np_array is None:
            np_array = np.zeros(self.ARRAY_LEN)

        assert(len(np_array) == self.ARRAY_LEN), 'provided np_array has wrong length.'

        self._grasp_array = np_array.astype(np.float32)

    @property
    def internal_array(self):
        """
        :return: the internal ndarray representation. only use when you know what you're doing.
        """
        return self._grasp_array

    @internal_array.setter
    def internal_array(self, np_array):
        """
        :param np_array: the new internal array, must be of length Grasp.ARRAY_LEN. use with caution.
        """
        assert(len(np_array) == self.ARRAY_LEN), 'provided np_array has wrong length.'
        self._grasp_array = np_array.astype(np.float32)

    @property
    def translation(self):
        """
        :return: translation as np-array with length 3
        """
        return self._grasp_array[0:3]

    @translation.setter
    def translation(self, translation):
        """
        :param translation: np-array with length 3
        """
        self._grasp_array[0:3] = np.asarray(translation).astype(np.float32)

    @property
    def rotation_matrix(self):
        """
        :return: rotation matrix as np array 3x3
        """
        return self._grasp_array[3:12].reshape((3, 3))

    @rotation_matrix.setter
    def rotation_matrix(self, rotation_matrix):
        """
        :param rotation_matrix: the rotation matrix, 3x3 numpy array
        """
        self._grasp_array[3:12] = rotation_matrix.reshape(9).astype(np.float32)

    @property
    def pose(self):
        """
        :return: the 6d pose as homogenous transformation matrix 4x4 np array
        """
        pose = np.eye(4)
        pose[0:3, 0:3] = self.rotation_matrix
        pose[0:3, 3] = self.translation
        return pose

    @pose.setter
    def pose(self, pose):
        """
        :param pose: the 6d pose as homogenous transformation matrix 4x4 np array
        """
        self.translation = pose[0:3, 3]
        self.rotation_matrix = pose[0:3, 0:3]

    @property
    def score(self):
        """
        :return: the score of this grasp as float value
        """
        return self._grasp_array[12]

    @score.setter
    def score(self, score):
        """
        :param score: a float value as the score
        """
        self._grasp_array[12] = float(score)

    def distance_to(self, other_grasp):
        """
        Computes the distance of this grasp to the other_grasp according to Eppner et al., 2019.

        Translation and rotation terms are weighted so that a distance of 1 equals to either 1 mm translational or
        1 degree rotational distance.

        :param other_grasp: the other grasp of type Grasp

        :return: the distance between this grasp and the other grasp as float value
        """
        # compute using module function
        # could also use quaternion computation:
        # rotation_dist = 2*np.arccos(np.abs(np.dot(q1, q2)))/np.pi*180

        return pairwise_distances(self, other_grasp)[0, 0]

    def as_grasp_set(self):
        """
        Returns the grasp as a 1-element grasp set.

        :return: grasp.GraspSet object containing only this grasp.
        """
        return GraspSet(self._grasp_array.reshape(1, self.ARRAY_LEN))

    def transform(self, tf):
        """
        Applies the given transform to the grasp (in-place).

        :param tf: (4, 4) homogenous transformation matrix
        """
        p = self.pose
        self.pose = np.matmul(tf, p)


class GraspSet:
    """
    A GraspSet is a collection of [0 ... N] grasps.

    :param np_array: optional, the internal numpy array, which is of shape (n, Grasp.ARRAY_LEN) and each row is a Grasp
    """

    def __init__(self, np_array=None):
        if np_array is None:
            np_array = np.zeros((0, Grasp.ARRAY_LEN), dtype=np.float32)

        assert(np_array.shape[1] == Grasp.ARRAY_LEN), 'provided np_array has wrong shape.'

        self._gs_array = np_array.astype(np.float32)

    @classmethod
    def from_translations_and_quaternions(cls, translations, quaternions):
        """
        creates a grasp set from poses specified with translation (3) and quaternion (4).
        the quaternion needs to be in (w, x, y, z) order.

        :param translations: (n, 3) np array with position
        :param quaternions: (n, 4) np array with quaternion (w, x, y, z)

        :return: grasp set with corresponding poses, all other fields are zero-initialised
        """
        if len(translations) != len(quaternions):
            raise ValueError(f'provided translations ({len(translations)}) and quaternions ({len(quaternions)})' +
                             f' arrays must be of same length.')

        gs = cls(np.zeros((translations.shape[0], Grasp.ARRAY_LEN), dtype=np.float32))
        # get translations
        gs.translations = translations

        # get rotation matrices (using the numpy-quaternion package, which offers vectorized implementations)
        quats = quaternion.as_quat_array(quaternions)
        rotation_matrices = quaternion.as_rotation_matrix(quats)
        gs.rotation_matrices = rotation_matrices

        return gs

    @classmethod
    def from_translations(cls, translations):
        """
        creates a grasp set from translations (x, y, z) only - rotation matrices will be eye(3)

        :param translations: (n, 3) np array with position

        :return: grasp set with corresponding grasping points with default orientation (i.e. np.eye(3) as rotation
         matrix), other fields are zero-initialised
        """
        np_array = np.zeros((translations.shape[0], Grasp.ARRAY_LEN), dtype=np.float32)
        np_array[:, 0:3] = translations

        # set canonical orientations (= np.eye(3) for each grasp)
        np_array[:, 3] = 1.0
        np_array[:, 7] = 1.0
        np_array[:, 11] = 1.0

        return cls(np_array)

    @classmethod
    def from_poses(cls, poses):
        """creates a grasp set from given poses (n, 4, 4)

        :param poses: (n, 4, 4) np array with homogenous transformation matrices

        :return: grasp set with corresponding grasps, other fields are zero-initialised
        """
        np_array = np.zeros((poses.shape[0], Grasp.ARRAY_LEN), dtype=np.float32)
        gs = cls(np_array)
        gs.poses = poses
        return gs

    def __len__(self):
        return self._gs_array.shape[0]

    def __getitem__(self, item):
        """
        :param item: can be index, slice or array
        :return: if single index, then grasp object, else grasp set object - note that these will be shallow copies
        """
        if type(item) == int:
            return Grasp(self._gs_array[item])
        elif (type(item) == slice) or (type(item) == list) or (type(item) == np.ndarray):
            return GraspSet(self._gs_array[item])
        else:
            raise TypeError('unknown index type calling GraspSet.__getitem__')

    def __setitem__(self, key, value):
        """
        :param key: can be index, slice or array
        :param value: if single index: Grasp object or GraspSet, else: GraspSet object
        """
        if type(key) == int:
            if type(value) is GraspSet:
                if len(value) != 1:
                    raise ValueError('If type(index) is int, value needs to be Grasp or GraspSet of length 1.')
                value = Grasp(value.internal_array)
            if type(value) is not Grasp:
                raise TypeError('Provided value has wrong type. Expected Grasp or GraspSet of length 1.')
            self._gs_array[key] = value.internal_array

        elif (type(key) == slice) or (type(key) == list) or (type(key) == np.ndarray):
            if type(value) is not GraspSet:
                raise TypeError('Provided value has wrong type. Expected GraspSet.')
            self._gs_array[key] = value.internal_array
        else:
            raise TypeError('unknown index type calling GraspSet.__setitem__')

    @property
    def internal_array(self):
        """
        :return: gives the internal numpy array representation of shape (N, Grasp.ARRAY_LEN)
        """
        return self._gs_array

    @internal_array.setter
    def internal_array(self, np_array):
        """
        :param np_array: sets the new internal np array, must be of dim (n, Grasp.ARRAY_LEN). use with caution.
        """
        assert(np_array.shape[1] == Grasp.ARRAY_LEN), 'provided np_array has wrong shape.'
        self._gs_array = np_array.astype(np.float32)

    @property
    def translations(self):
        """
        :return: (n, 3) np array with translations
        """
        return self._gs_array[:, 0:3]

    @translations.setter
    def translations(self, translations):
        """
        :param translations: an (n, 3) np array
        :return:
        """
        assert(translations.shape == (len(self), 3)), "provided translations have wrong shape"
        self._gs_array[:, 0:3] = translations

    @property
    def rotation_matrices(self):
        """
        :return: (n, 3, 3) np array with rotation matrices
        """
        return self._gs_array[:, 3:12].reshape((-1, 3, 3))

    @rotation_matrices.setter
    def rotation_matrices(self, rotation_matrices):
        """
        :param rotation_matrices:  (n, 3, 3) np array with rotation matrices
        """
        assert(rotation_matrices.shape == (len(self), 3, 3)), "provided rotation matrices have wrong shape"
        self._gs_array[:, 3:12] = rotation_matrices.reshape((-1, 9))

    @property
    def poses(self):
        """
        :return: (n, 4, 4) np array with poses (homogenous tf matrices)
        """
        poses = np.zeros((len(self), 4, 4), dtype=np.float32)
        poses[:, 3, 3] = 1
        poses[:, 0:3, 3] = self.translations
        poses[:, 0:3, 0:3] = self.rotation_matrices
        return poses

    @poses.setter
    def poses(self, poses):
        """
        :param poses: (n, 4, 4) np array with poses (homogenous tf matrices)
        """
        assert(poses.shape == (len(self), 4, 4)), "provided poses have wrong shape"
        self._gs_array[:, 0:3] = poses[:, 0:3, 3]
        self._gs_array[:, 3:12] = poses[:, 0:3, 0:3].reshape((-1, 9))

    @property
    def scores(self):
        """
        :return: (n,) np array with scores as float - as of yet there is no definition of what score means
        """
        return self._gs_array[:, 12]

    @scores.setter
    def scores(self, scores):
        """
        :param scores: (n,) np array with scores
        """
        assert(len(scores) == len(self)), "provided scores have wrong array length"
        self._gs_array[:, 12] = scores[:]

    def add(self, grasp_set):
        """
        Add the given grasp set to this one. Can also be a single grasp.

        :param grasp_set: Grasp or grasp set.

        :return: Itself.
        """
        self._gs_array = np.concatenate([self._gs_array, grasp_set.internal_array])
        return self

    def transform(self, tf):
        """
        Applies the given transform to the grasp (in-place).

        :param tf: (4, 4) homogenous transformation matrix
        """
        p = self.poses
        self.poses = np.matmul(tf, p)
