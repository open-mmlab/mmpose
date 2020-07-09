import cv2
import numpy as np
from mmcv.parallel import DataContainer as DC
from torchvision.transforms import functional as F

from mmpose.core.post_processing import (affine_transform, fliplr_joints,
                                         get_affine_transform)
from ...registry import PIPELINES


@PIPELINES.register_module()
class ToTensor():
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        results['img'] = F.to_tensor(results['img'])
        return results


@PIPELINES.register_module()
class NormalizeTensor():
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['img'] = F.normalize(
            results['img'], mean=self.mean, std=self.std)
        return results


@PIPELINES.register_module()
class RandomFlip():
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'. Modifies key: 'img', 'joints_3d', 'joints_3d_visible' and
    'center'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']

        if np.random.rand() <= self.flip_prob:
            img = img[:, ::-1, :]

            joints_3d, joints_3d_visible = fliplr_joints(
                joints_3d, joints_3d_visible, img.shape[1],
                results['ann_info']['flip_pairs'])
            center[0] = img.shape[1] - center[0] - 1

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center

        return results


@PIPELINES.register_module()
class HalfBodyTransform():
    """Data augmentation with half-body transform. Keep only the upper body or
    the lower body at random.

    Required keys: 'joints_3d', 'joints_3d_visible', and 'ann_info'.
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
        upper_joints = []
        lower_joints = []
        for joint_id in range(cfg['num_joints']):
            if joints_3d_visible[joint_id][0] > 0:
                if joint_id in cfg['upper_body_ids']:
                    upper_joints.append(joints_3d[joint_id])
                else:
                    lower_joints.append(joints_3d[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        elif len(lower_joints) > 2:
            selected_joints = lower_joints
        else:
            selected_joints = upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)

        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        aspect_ratio = cfg['image_size'][0] / cfg['image_size'][1]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        scale = scale * 1.5
        return center, scale

    def __call__(self, results):
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        if (np.sum(joints_3d_visible[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):

            c_half_body, s_half_body = self.half_body_transform(
                results['ann_info'], joints_3d, joints_3d_visible)

            if c_half_body is not None and s_half_body is not None:
                results['center'] = c_half_body
                results['scale'] = s_half_body

        return results


@PIPELINES.register_module()
class RandomScaleRotation():
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
class AffineTransform():
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'. Modified keys:'img', 'joints_3d', and
    'joints_3d_visible'.
    """

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']
        trans = get_affine_transform(c, s, r, image_size)

        img = cv2.warpAffine(
            img,
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(results['ann_info']['num_joints']):
            if joints_3d_visible[i, 0] > 0.0:
                joints_3d[i, 0:2] = affine_transform(joints_3d[i, 0:2], trans)

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible

        return results


@PIPELINES.register_module()
class GenerateTarget():
    """Generate the target heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.
    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian.
        unbiased_encoding (bool): Option to use unbiased
            encoding methods.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, sigma=2, unbiased_encoding=False):
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding

    def _generate_target(self, cfg, joints_3d, joints_3d_visible):
        """Generate the target heatmap.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray([num_joints, 3])
            joints_3d_visible: np.ndarray([num_joints, 3])
        Returns:
             target, target_weight(1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joints_weight = cfg['joints_weight']
        use_different_joints_weight = cfg['use_different_joints_weight']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                          dtype=np.float32)

        tmp_size = self.sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                heatmap_vis = joints_3d_visible[joint_id, 0]
                target_weight[joint_id] = heatmap_vis

                feat_stride = image_size / heatmap_size
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[
                        0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, heatmap_size[0], 1, np.float32)
                y = np.arange(0, heatmap_size[1], 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(
                        -((x - mu_x)**2 + (y - mu_y)**2) / (2 * self.sigma**2))
        else:
            for joint_id in range(num_joints):
                heatmap_vis = joints_3d_visible[joint_id, 0]
                target_weight[joint_id] = heatmap_vis

                feat_stride = image_size / heatmap_size
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[
                        0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                               (2 * self.sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if use_different_joints_weight:
            target_weight = np.multiply(target_weight, joints_weight)

        return target, target_weight

    def __call__(self, results):
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        target, target_weight = self._generate_target(results['ann_info'],
                                                      joints_3d,
                                                      joints_3d_visible)

        results['target'] = target
        results['target_weight'] = target_weight

        return results


@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
            This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
            The contents of the `meta_name` dictionary depends on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DC(meta, cpu_only=True)

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')
