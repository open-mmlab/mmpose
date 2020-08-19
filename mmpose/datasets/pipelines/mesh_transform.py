import cv2
import mmcv
import numpy as np
from torchvision.transforms import functional as F

from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform)
from mmpose.datasets.registry import PIPELINES


def flip_smpl_pose(pose):
    """Flip SMPL pose parameters horizontally.

    Args:
        pose (np.ndarray([72])): SMPL pose parameters

    Returns:
        pose_flipped
    """

    flippedParts = [
        0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19,
        20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37,
        38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58,
        59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68
    ]
    pose_flipped = pose[flippedParts]
    # we also negate the second and the third dimension of the axis-angle
    pose_flipped[1::3] = -pose_flipped[1::3]
    pose_flipped[2::3] = -pose_flipped[2::3]
    return pose_flipped


def flip_iuv(iuv, uv_type='BF'):
    """Flip IUV image horizontally.

    Notes:
        IUV image height: H
        IUV image width: W

    Args:
        iuv np.ndarray([H, W, 3]): IUV image
        uv_type (str): The type of the UV map.

    Returns:
        iuv_flipped
    """

    if uv_type == 'BF':
        iuv_flipped = iuv[:, ::-1, :]
        iuv_flipped[:, :, 1] = 255 - iuv_flipped[:, :, 1]
    else:
        # The flip of other UV map is complex, not finished yet.
        pass
    return iuv_flipped


def rotate_joints_3d(joints_3d, rot):
    """Rotate the 3D joints in the local coordinates.

    Notes:
        Joints number: K

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        rot (float): Rotation angle.

    Returns:
        joints_3d_rotated
    """
    # in-plane rotation
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = -rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]

    joints_3d_rotated = np.einsum('ij,kj->ki', rot_mat, joints_3d)
    joints_3d_rotated = joints_3d_rotated.astype('float32')
    return joints_3d_rotated


def rot_smpl_pose(pose, rot):
    """Rotate SMPL parameters.

    Args:
        pose (np.ndarray([72])): SMPL pose parameters
        rot (float): Rotation angle.

    Returns:
        pose_rotated
    """
    pose_rotated = pose
    if not rot == 0:
        R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                      [np.sin(np.deg2rad(-rot)),
                       np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]])

        orient = pose[:3]
        # find the rotation of the body in camera frame
        per_rdg, _ = cv2.Rodrigues(orient)
        # apply the global rotation to the global orientation
        resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
        pose_rotated[:3] = (resrot.T)[0]

    return pose_rotated


def flip_joints_3d(joints_3d, joints_3d_visible, flip_pairs):
    """Flip human joints in 3D space horizontally.

    Note:
        num_keypoints: K

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        joints_3d_visible (np.ndarray([K, 1])): Visibility of keypoints.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).

    Returns:
        joints_3d_flipped, joints_3d_visible_flipped
    """

    assert len(joints_3d) == len(joints_3d_visible)

    joints_3d_flipped = joints_3d.copy()
    joints_3d_visible_flipped = joints_3d_visible.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        joints_3d_flipped[left, :] = joints_3d[right, :]
        joints_3d_flipped[right, :] = joints_3d[left, :]

        joints_3d_visible_flipped[left, :] = joints_3d_visible[right, :]
        joints_3d_visible_flipped[right, :] = joints_3d_visible[left, :]

    # Flip horizontally
    joints_3d_flipped[:, 0] = -joints_3d_flipped[:, 0]
    joints_3d_flipped = joints_3d_flipped * joints_3d_visible_flipped

    return joints_3d_flipped, joints_3d_visible_flipped


@PIPELINES.register_module()
class LoadIUVFromFile(object):
    """Loading IUV image from file."""

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32
        self.color_type = 'color'
        # channel relations: iuv->bgr
        channel_order = 'bgr'
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        has_iuv = results['has_iuv']
        use_iuv = results['ann_info']['use_IUV']
        if has_iuv and use_iuv:
            iuv_file = results['iuv_file']
            iuv = mmcv.imread(iuv_file, self.color_type, self.channel_order)
            if iuv is None:
                raise ValueError('Fail to read {}'.format(iuv_file))
        else:
            has_iuv = 0
            iuv = None

        results['has_iuv'] = has_iuv
        results['iuv'] = iuv
        return results


@PIPELINES.register_module()
class IUVToTensor():
    """Transform IUV image to Tensor.

    Required key: 'iuv'. Modifies key: 'iuv'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        iuv = F.to_tensor(results['iuv'])
        iuv[0, :, :] = iuv[0, :, :] * 255
        results['iuv'] = iuv
        return results


@PIPELINES.register_module()
class MeshRandomFlip():
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_2d','joints_2d_visible', 'joints_3d',
    'joints_3d_visible', 'center', 'pose', 'iuv' and 'ann_info'.
    Modifies key: 'img', 'joints_2d','joints_2d_visible', 'joints_3d',
    'joints_3d_visible', 'center', 'pose', 'iuv'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_2d = results['joints_2d']
        joints_2d_visible = results['joints_2d_visible']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        pose = results['pose']
        iuv = results['iuv']

        if np.random.rand() <= self.flip_prob:
            img = img[:, ::-1, :]
            pose = flip_smpl_pose(pose)

            joints_2d, joints_2d_visible = fliplr_joints(
                joints_2d, joints_2d_visible, img.shape[1],
                results['ann_info']['flip_pairs'])

            joints_3d, joints_3d_visible = flip_joints_3d(
                joints_3d, joints_3d_visible,
                results['ann_info']['flip_pairs'])
            if iuv is not None:
                iuv = flip_iuv(iuv, results['ann_info']['uv_type'])

        results['img'] = img
        results['joints_2d'] = joints_2d
        results['joints_2d_visible'] = joints_2d_visible
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['pose'] = pose
        results['iuv'] = iuv
        return results


@PIPELINES.register_module()
class MeshGetRandomScaleRotation():
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['scale'] = s
        results['rotation'] = r

        return results


@PIPELINES.register_module()
class MeshAffine():
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'. Modified keys:'img', 'joints_3d', and
    'joints_3d_visible'.

    Required keys: 'img', 'joints_2d','joints_2d_visible', 'joints_3d',
     'joints_3d_visible', 'pose', 'iuv', 'ann_info','scale',
     'rotation' and 'center'.
    Modifies key: 'img', 'joints_2d','joints_2d_visible', 'joints_3d',
     'pose', 'iuv'.
    """

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_2d = results['joints_2d']
        joints_2d_visible = results['joints_2d_visible']
        joints_3d = results['joints_3d']
        pose = results['pose']
        iuv = results['iuv']

        c = results['center']
        s = results['scale']
        r = results['rotation']
        trans = get_affine_transform(c, s, r, image_size)

        img = cv2.warpAffine(
            img,
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        if iuv is not None:
            iuv = cv2.warpAffine(
                iuv,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_NEAREST)

        for i in range(results['ann_info']['num_joints']):
            if joints_2d_visible[i, 0] > 0.0:
                joints_2d[i] = affine_transform(joints_2d[i], trans)

        joints_3d = rotate_joints_3d(joints_3d, r)
        pose = rot_smpl_pose(pose, r)

        results['img'] = img
        results['joints_2d'] = joints_2d
        results['joints_2d_visible'] = joints_2d_visible
        results['joints_3d'] = joints_3d
        results['pose'] = pose
        results['iuv'] = iuv
        return results
