# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.data import InstanceData, PixelData
from torchvision.transforms import functional as F

from mmpose.registry import TRANSFORMS
from mmpose.structures import PoseDataSample
from mmpose.structures.bbox import bbox_xyxy2cs


@TRANSFORMS.register_module()
class LoadImageFromFileV0(BaseTransform):
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _read_image(self, path):
        img_bytes = self.file_client.get(path)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        if img is None:
            raise ValueError(f'Fail to read {path}')
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    @staticmethod
    def _bgr2rgb(img):
        if img.ndim == 3:
            return mmcv.bgr2rgb(img)
        elif img.ndim == 4:
            return np.concatenate([mmcv.bgr2rgb(img_) for img_ in img], axis=0)
        else:
            raise ValueError('results["img"] has invalid shape '
                             f'{img.shape}')

    def transform(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results.get('img_path', None)

        if isinstance(image_file, (list, tuple)):
            # Load images from a list of paths
            results['img'] = [self._read_image(path) for path in image_file]
        elif image_file is not None:
            # Load single image from path
            results['img'] = self._read_image(image_file)
        else:
            if 'img' not in results:
                # If `image_file`` is not in results, check the `img` exists
                # and format the image. This for compatibility when the image
                # is manually set outside the pipeline.
                raise KeyError('Either `image_file` or `img` should exist in '
                               'results.')
            if isinstance(results['img'], (list, tuple)):
                assert isinstance(results['img'][0], np.ndarray)
            else:
                assert isinstance(results['img'], np.ndarray)
            if self.color_type == 'color' and self.channel_order == 'rgb':
                # The original results['img'] is assumed to be image(s) in BGR
                # order, so we convert the color according to the arguments.
                if isinstance(results['img'], (list, tuple)):
                    results['img'] = [
                        self._bgr2rgb(img) for img in results['img']
                    ]
                else:
                    results['img'] = self._bgr2rgb(results['img'])
            results['image_file'] = None

        img = results['img']
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@TRANSFORMS.register_module()
class TopDownGetBboxCenterScaleV0(BaseTransform):
    """Convert bbox from [x, y, w, h] to center and scale.

    The center is the coordinates of the bbox center, and the scale is the
    bbox width and height normalized by a scale factor.

    Required key: 'bbox', 'ann_info'

    Modifies key: 'center', 'scale'

    Args:
        padding (float): bbox padding scale that will be multilied to scale.
            Default: 1.25
    """

    def __init__(self, padding: float = 1.25):
        self.padding = padding

    def transform(self, results):
        """The transform function of :class:`GetBBoxCenterScale`.

        See ``transform()`` method of :class:`BaseTransform` for details.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """
        if 'bbox_center' in results and 'bbox_scale' in results:
            warnings.warn('Use the existing "bbox_center" and "bbox_scale". '
                          'The padding will still be applied.')
            results['bbox_scale'] *= self.padding

        else:
            bbox = results['bbox']
            center, scale = bbox_xyxy2cs(bbox, padding=self.padding)

            aspect_ratio = 192.0 / 256.0  # hard code here
            w, h = scale[0][0], scale[0][1]
            if w > aspect_ratio * h:
                h = w * 1.0 / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio

            scale = np.array([w, h], dtype=np.float32).reshape(1, 2)

            results['bbox_center'] = center
            results['bbox_scale'] = scale

        return results


@TRANSFORMS.register_module()
class TopDownRandomShiftBboxCenterV0(BaseTransform):
    """Random shift the bbox center.

    Required key: 'center', 'scale'

    Modifies key: 'center'

    Args:
        shift_factor (float): The factor to control the shift range, which is
            scale*pixel_std*scale_factor. Default: 0.16
        prob (float): Probability of applying random shift. Default: 0.3
    """

    def __init__(self, shift_factor: float = 0.16, prob: float = 0.3):
        self.shift_factor = shift_factor
        self.prob = prob

    def transform(self, results):

        center = results['bbox_center']
        scale = results['bbox_scale']
        if np.random.rand() < self.prob:
            center += np.random.uniform(-1, 1, 2) * self.shift_factor * scale

        results['bbox_center'] = center
        return results


@TRANSFORMS.register_module()
class TopDownRandomFlipV0(BaseTransform):
    """Data augmentation with random image flip.

    Required key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    @staticmethod
    def fliplr_joints(joints_3d, joints_3d_visible, img_width, flip_pairs):
        """Flip human joints horizontally.

        Note:
            - num_keypoints: K

        Args:
            joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
            joints_3d_visible (np.ndarray([K, 1])): Visibility of keypoints.
            img_width (int): Image width.
            flip_pairs (list[tuple]): Pairs of keypoints which are mirrored
                (for example, left ear and right ear).

        Returns:
            tuple: Flipped human joints.

            - joints_3d_flipped (np.ndarray([K, 3])): Flipped joints.
            - joints_3d_visible_flipped (np.ndarray([K, 1])): Joint visibility.
        """
        # joints_3d: [1, K, 2]
        # joints_3d_visible: [1, K]
        assert joints_3d.shape[:-1] == joints_3d_visible.shape
        assert img_width > 0

        joints_3d_flipped = joints_3d.copy()
        joints_3d_visible_flipped = joints_3d_visible.copy()
        # hard-code coco flip_pairs
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]

        # Swap left-right parts
        for left, right in flip_pairs:
            joints_3d_flipped[0, left] = joints_3d[0, right]
            joints_3d_flipped[0, right] = joints_3d[0, left]

            joints_3d_visible_flipped[0, left] = joints_3d_visible[0, right]
            joints_3d_visible_flipped[0, right] = joints_3d_visible[0, left]

        # Flip horizontally
        joints_3d_flipped[..., 0] = img_width - 1 - joints_3d_flipped[..., 0]
        joints_3d_flipped = joints_3d_flipped * joints_3d_visible_flipped[...,
                                                                          None]

        # joints_3d_flipped = joints_3d_flipped.reshape(1, -1, 2)
        # joints_3d_visible_flipped = joints_3d_visible_flipped.reshape(1, -1)

        return joints_3d_flipped, joints_3d_visible_flipped

    def transform(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_3d = results['keypoints']
        joints_3d_visible = results['keypoints_visible']
        center = results['bbox_center']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        flipped = False
        if np.random.rand() <= self.flip_prob:
            flipped = True
            img = img[:, ::-1, :]
            joints_3d, joints_3d_visible = self.fliplr_joints(
                joints_3d, joints_3d_visible, img.shape[1],
                results['flip_indices'])
            center[0][0] = img.shape[1] - center[0][0] - 1

        results['img'] = img
        results['keypoints'] = joints_3d
        results['keypoints_visible'] = joints_3d_visible
        results['bbox_center'] = center
        results['flip'] = flipped

        if flipped:
            results['flip_direction'] = 'horizontal'
        else:
            results['flip_direction'] = None

        return results


@TRANSFORMS.register_module()
class TopDownHalfBodyTransformV0(BaseTransform):
    """Data augmentation with half-body transform. Keep only the upper body or
    the lower body at random.

    Required key: 'joints_3d', 'joints_3d_visible', and 'ann_info'.

    Modifies key: 'scale' and 'center'.

    Args:
        num_joints_half_body (int): Threshold of performing
            half-body transform. If the body has fewer number
            of joints (< num_joints_half_body), ignore this step.
        prob_half_body (float): Probability of half-body transform.
    """

    def __init__(self, num_joints_half_body=8, prob_half_body=0.3):
        self.num_joints_half_body = num_joints_half_body
        self.prob_half_body = prob_half_body

    @staticmethod
    def half_body_transform(cfg, joints_3d, joints_3d_visible):
        """Get center&scale for half-body transform."""
        upper_joints = []
        lower_joints = []
        # hard-code 17 here
        for joint_id in range(17):
            if joints_3d_visible[0][joint_id] > 0:
                if joint_id in cfg['upper_body_ids']:
                    upper_joints.append(joints_3d[0][joint_id])
                else:
                    lower_joints.append(joints_3d[0][joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(
            selected_joints, dtype=np.float32)  # [num_joints, 2]
        center = selected_joints.mean(axis=0)[:2]
        center = center.reshape(1, 2)

        left_top = np.amin(selected_joints, axis=0)

        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = 192.0 / 256.0

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w, h], dtype=np.float32).reshape(1, 2)
        scale = scale * 1.5
        return center, scale

    def transform(self, results):
        """Perform data augmentation with half-body transform."""
        joints_3d = results['keypoints']  # [1, K, 2]
        joints_3d_visible = results['keypoints_visible']  # [1, K]

        if (np.sum(joints_3d_visible) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
            c_half_body, s_half_body = self.half_body_transform(
                results, joints_3d, joints_3d_visible)

            if c_half_body is not None and s_half_body is not None:
                results['bbox_center'] = c_half_body
                results['bbox_scale'] = s_half_body

        return results


@TRANSFORMS.register_module()
class TopDownGetRandomScaleRotationV0(BaseTransform):
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'.

    Modifies key: 'scale' and 'rotation'.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def transform(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['bbox_scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['bbox_scale'] = s
        results['bbox_rotation'] = r

        return results


@TRANSFORMS.register_module()
class TopDownAffineV0(BaseTransform):
    """Affine transform the image to make input.

    Required key:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified key:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    @staticmethod
    def _get_3rd_point(a, b):
        """To calculate the affine matrix, three pairs of points are required.
        This function is used to get the 3rd point, given 2D points a & b.

        The 3rd point is defined by rotating vector `a - b` by 90 degrees
        anticlockwise, using b as the rotation center.

        Args:
            a (np.ndarray): point(x,y)
            b (np.ndarray): point(x,y)

        Returns:
            np.ndarray: The 3rd point.
        """
        assert len(a) == 2
        assert len(b) == 2
        direction = a - b
        third_pt = b + np.array([-direction[1], direction[0]],
                                dtype=np.float32)

        return third_pt

    @staticmethod
    def rotate_point(pt, angle_rad):
        """Rotate a point by an angle.

        Args:
            pt (list[float]): 2 dimensional point to be rotated
            angle_rad (float): rotation angle by radian

        Returns:
            list[float]: Rotated point.
        """
        assert len(pt) == 2
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        new_x = pt[0] * cs - pt[1] * sn
        new_y = pt[0] * sn + pt[1] * cs
        rotated_pt = [new_x, new_y]

        return rotated_pt

    def get_affine_transform(self,
                             center,
                             scale,
                             rot,
                             output_size,
                             shift=(0., 0.),
                             inv=False):
        """Get the affine transform matrix, given the
        center/scale/rot/output_size.

        Args:
            center (np.ndarray[2, ]): Center of the bounding box (x, y).
            scale (np.ndarray[2, ]): Scale of the bounding box
                wrt [width, height].
            rot (float): Rotation angle (degree).
            output_size (np.ndarray[2, ] | list(2,)): Size of the
                destination heatmaps.
            shift (0-100%): Shift translation ratio wrt the width/height.
                Default (0., 0.).
            inv (bool): Option to inverse the affine transform direction.
                (inv=False: src->dst or inv=True: dst->src)

        Returns:
            np.ndarray: The transform matrix.
        """
        assert len(center) == 2
        assert len(scale) == 2
        assert len(output_size) == 2
        assert len(shift) == 2

        # # pixel_std is 200.
        # scale_tmp = scale * 200.0
        scale_tmp = scale

        shift = np.array(shift)
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.rotate_point([0., src_w * -0.5], rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    @staticmethod
    def affine_transform(pt, trans_mat):
        """Apply an affine transformation to the points.

        Args:
            pt (np.ndarray): a 2 dimensional point to be transformed
            trans_mat (np.ndarray): 2x3 matrix of an affine transform

        Returns:
            np.ndarray: Transformed points.
        """
        assert len(pt) == 2
        new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

        return new_pt

    def transform(self, results):
        # hard-code here
        image_size = [192, 256]

        img = results['img']
        joints_3d = results['keypoints']  # [1, K, 2]
        joints_3d_visible = results['keypoints_visible']  # [1, K]
        c = results['bbox_center'][0]
        s = results['bbox_scale'][0]
        if 'bbox_rotation' in results:
            r = results['bbox_rotation']
        else:
            r = 0

        trans = self.get_affine_transform(c, s, r, image_size)
        img = cv2.warpAffine(
            img,
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        # hard-code 17 here
        for i in range(17):
            if joints_3d_visible[0, i] > 0.0:
                joints_3d[0][i] = self.affine_transform(joints_3d[0][i], trans)

        results['img'] = img
        results['keypoints'] = joints_3d
        results['keypoints_visible'] = joints_3d_visible
        results['input_size'] = (192, 256)

        return results


@TRANSFORMS.register_module()
class TopDownGenerateTargetV0(BaseTransform):

    def __init__(self,
                 sigma=2,
                 kernel=(11, 11),
                 valid_radius_factor=0.0546875,
                 target_type='GaussianHeatmap',
                 encoding='MSRA',
                 unbiased_encoding=False):
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):

        num_joints = 17
        image_size = np.array([192, 256])
        W, H = 48, 64
        joint_weights = cfg['dataset_keypoint_weights']
        use_different_joint_weights = False

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        for joint_id in range(num_joints):
            target_weight[joint_id] = joints_3d_visible[0, joint_id]

            feat_stride = image_size / [W, H]
            mu_x = int(joints_3d[0][joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints_3d[0][joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0

            if target_weight[joint_id] > 0.5:
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, None]
                x0 = y0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], W)
                img_y = max(0, ul[1]), min(br[1], H)

                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def transform(self, results):
        """Generate the target heatmap."""
        joints_3d = results['keypoints']
        joints_3d_visible = results['keypoints_visible']

        assert self.encoding in ['MSRA']

        if self.encoding == 'MSRA':
            target, target_weight = self._msra_generate_target(
                results, joints_3d, joints_3d_visible, self.sigma)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['heatmaps'] = target
        results['keypoint_weights'] = target_weight.reshape(1, -1)  # [1, K]

        return results


@TRANSFORMS.register_module()
class ToTensorV0(BaseTransform):
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __init__(self, device='cpu'):
        self.device = device

    def _to_tensor(self, x):
        return torch.from_numpy(x.astype('float32')).permute(2, 0, 1).to(
            self.device).div_(255.0)

    def transform(self, results):

        results['img'] = self._to_tensor(results['img'])

        return results


@TRANSFORMS.register_module()
class NormalizeTensorV0(BaseTransform):
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, results):

        results['img'] = F.normalize(
            results['img'], mean=self.mean, std=self.std, inplace=True)

        return results


@TRANSFORMS.register_module()
class PackPoseInputsV0(BaseTransform):
    """Pack the inputs data for pose estimation.

    The ``img_meta`` item is always populated. The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default it includes:

        - ``id``: id of the data sample

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``input_size``: the input size to the network

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys which will be stored in
            :obj: `PoseDataSample` as meta info. Defaults to ``('id',
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'input_size',
            'flip', 'flip_direction', 'flip_indices)``
    """

    # items in `instance_mapping_table` will be directly packed into
    # PoseDataSample without converting to Tensor
    instance_mapping_table = {
        'bbox': 'bboxes',
        'head_size': 'head_size',
        'bbox_center': 'bbox_centers',
        'bbox_scale': 'bbox_scales',
        'bbox_score': 'bbox_scores',
        'keypoints': 'keypoints',
        'keypoints_visible': 'keypoints_visible'
    }

    label_mapping_table = {
        'keypoint_labels': 'keypoint_labels',
        'keypoint_x_labels': 'keypoint_x_labels',
        'keypoint_y_labels': 'keypoint_y_labels',
        'keypoint_weights': 'keypoint_weights'
    }

    field_mapping_table = {
        'heatmaps': 'heatmaps',
    }

    def __init__(self,
                 meta_keys=('id', 'img_id', 'img_path', 'ori_shape',
                            'img_shape', 'input_size', 'flip',
                            'flip_direction', 'flip_indices')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`PoseDataSample`): The annotation info of the
                sample.
        """
        # Pack image(s)
        img_tensor = results['img']
        data_sample = PoseDataSample()

        # pack instance data
        gt_instances = InstanceData()
        for key, packed_key in self.instance_mapping_table.items():
            if key in results:
                gt_instances.set_field(results[key], packed_key)
        data_sample.gt_instances = gt_instances

        # pack instance labels
        gt_instance_labels = InstanceData()
        for key, packed_key in self.label_mapping_table.items():
            if key in results:
                gt_instance_labels.set_field(results[key], packed_key)
        data_sample.gt_instance_labels = gt_instance_labels.to_tensor()

        # pack fields
        gt_fields = PixelData()
        for key, packed_key in self.field_mapping_table.items():
            if key in results:
                gt_fields.set_field(results[key], packed_key)
        data_sample.gt_fields = gt_fields.to_tensor()

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)

        packed_results = dict()
        packed_results['inputs'] = img_tensor
        packed_results['data_sample'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
