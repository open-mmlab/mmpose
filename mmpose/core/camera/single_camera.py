import numpy as np

from .camera_base import CAMERAS, SingleCameraBase


@CAMERAS.register_module()
class SimpleCamera(SingleCameraBase):
    """Camera model to calculate coordinate transformation with given
    intrinsic/extrinsic camera parameters.

    Notes:
        The keypoint coordiante should be an np.ndarray with a shape of
    [...,Kj, C] where Kj is the keypoint number of an instance, and ndims
    is the coordinate dimension. For example:

        [Kj, C]: shape of joint coordinates of a person with Kj joints.
        [N, Kj, C]: shape of a batch of person joint coordinates.
        [N, T, Kj, C]: shape of a batch of pose sequences.

    Args:
        param (dict): camera parameters including:
            - R: 3x3, camera rotation matrix (camera-to-world)
            - T: 3x1, camera translation (camera-to-world)
            - K: (optional) 2x3, camera intrinsic matrix
            - k: (optional) nx1, camera radial distortion coefficients
            - p: (optional) mx1, camera tangential distortion coefficients
            - f: (optional) 2x1, camera focal length
            - c: (optional) 2x1, camera center
        if K is not provided, it will be calculated from f and c.

    Methods:
        world_to_camera: Project points from world coordinates to camera
            coordinates
        camera_to_pixel: Project points from camera coordinates to pixel
            coordinates
        world_to_pixel: Project points from world coordinates to pixel
            coordinates
    """

    def __init__(self, param):

        self.param = {}
        # extrinsic param
        R = np.array(param['R'], dtype=np.float32)
        T = np.array(param['T'], dtype=np.float32)
        assert R.shape == (3, 3)
        assert T.shape == (3, 1)
        self.param['R_c2w'] = R.T
        self.param['T_c2w'] = T.T
        self.param['R_w2c'] = np.linalg.inv(self.param['R_c2w'])
        self.param['T_w2c'] = -self.param['T_c2w'] @ self.param['R_w2c']

        # intrinsic param
        if 'K' in param:
            K = np.array(param['K'], dtype=np.float32)
            assert K.shape == (2, 3)
            self.param['K'] = K.T
        elif 'f' in param and 'c' in param:
            f = np.array(param['f'], dtype=np.float32)
            c = np.array(param['c'], dtype=np.float32)
            assert f.shape == (2, 1)
            assert c.shape == (2, 1)
            self.param['K'] = np.concatenate((np.diagflat(f), c), axis=-1).T
        else:
            raise ValueError('Camera intrinsic parameters are missing. '
                             'Either "K" or "f"&"c" should be provided.')

        # distortion param
        if 'k' in param and 'p' in param:
            self.undistortion = True
            self.param['k'] = np.array(param['k'], dtype=np.float32).flatten()
            self.param['p'] = np.array(param['p'], dtype=np.float32).flatten()
            assert self.param['k'].size in {3, 6}
            assert self.param['p'].size == 2
        else:
            self.undistortion = False

    def world_to_camera(self, X):
        assert isinstance(X, np.ndarray)
        assert X.ndim >= 2 and X.shape[-1] == 3
        return X @ self.param['R_w2c'] + self.param['T_w2c']

    def camera_to_world(self, X):
        assert isinstance(X, np.ndarray)
        assert X.ndim >= 2 and X.shape[-1] == 3
        return X @ self.param['R_c2w'] + self.param['T_c2w']

    def camera_to_pixel(self, X):
        assert isinstance(X, np.ndarray)
        assert X.ndim >= 2 and X.shape[-1] == 3

        _X = X / X[..., 2:]

        if self.undistortion:
            k = self.param['k']
            p = self.param['p']
            _XX = _X[..., :2]
            r2 = (_XX**2).sum(-1)
            radinal = 1 + sum((ki * r2**(i + 1) for i, ki in enumerate(k[:3])))
            if k.size == 6:
                radinal /= 1 + sum(
                    (ki * r2**(i + 1) for i, ki in enumerate(k[3:])))

            tan = 2 * (p[1] * _X[..., 0] + p[0] * _X[..., 1])

            _X[..., :2] = _XX * (radinal + tan)[..., None] + np.outer(
                r2, p[::-1]).reshape(_XX.shape)
        return _X @ self.param['K']
