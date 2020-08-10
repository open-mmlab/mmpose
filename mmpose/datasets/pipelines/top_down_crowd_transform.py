import cv2
import numpy as np

from mmpose.core.post_processing import (affine_transform,
                                         fliplr_interference_joints,
                                         fliplr_joints, get_affine_transform)
from mmpose.datasets.registry import PIPELINES


@PIPELINES.register_module
class TopDownCrowdRandomFlip(object):

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        interference_joints = results['interference_joints']
        center = results['center']

        if np.random.rand() <= self.flip_prob:
            img = img[:, ::-1, :]

            joints_3d, joints_3d_visible = fliplr_joints(
                joints_3d, joints_3d_visible, img.shape[1],
                results['ann_info']['flip_pairs'])

            interference_joints = \
                fliplr_interference_joints(
                    interference_joints,
                    img.shape[1], results['ann_info']['joint_to_joint'])

            center[0] = img.shape[1] - center[0] - 1

        results['data_numpy'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_vis'] = joints_3d_visible
        results['interference_joints'] = interference_joints
        results['center'] = center

        return results


@PIPELINES.register_module()
class TopDownCrowdAffine():
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible',
    'interference_joints', 'ann_info', 'scale', 'rotation' and 'center'.
    Modified keys:'img', 'joints_3d', 'joints_3d_visible', and
    'interference_joints'.
    """

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        interference_joints = results['interference_joints']
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

        for i in range(len(interference_joints)):
            interference_joints[i, 0:2] = affine_transform(
                interference_joints[i, 0:2], trans)

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['interference_joints'] = interference_joints

        return results


@PIPELINES.register_module()
class TopDownCrowdGenerateTarget():
    """Generate the target heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible',
    'interference_joints', 'ann_info'.
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

    def _crowd_generate_target(self, cfg, joints_3d, joints_3d_visible,
                               interference_joints):
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

            for i in range(len(interference_joints)):
                joint_id = int(interference_joints[i][2])
                feat_stride = image_size / heatmap_size
                mu_x = interference_joints[i][0] / feat_stride[0]
                mu_y = interference_joints[i][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[
                        0] < 0 or br[1] < 0:
                    break
                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, heatmap_size[0], 1, np.float32)
                y = np.arange(0, heatmap_size[1], 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    g = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) /
                               (2 * self.sigma**2))
                    g = g * 0.5
                    target[joint_id] = np.maximum(target[joint_id], g)

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

            for i in range(len(interference_joints)):
                joint_id = int(interference_joints[i][2])
                feat_stride = image_size / heatmap_size
                mu_x = int(interference_joints[i][0] / feat_stride[0] + 0.5)
                mu_y = int(interference_joints[i][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] or br[
                        0] < 0 or br[1] < 0:
                    break
                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized, \
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) /
                               (2 * self.sigma**2))
                    g *= 0.5

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        np.maximum(target[joint_id][img_y[0]:img_y[1],
                                   img_x[0]:img_x[1]],
                                   g[g_y[0]:g_y[1], g_x[0]:g_x[1]])

        if use_different_joints_weight:
            target_weight = np.multiply(target_weight, joints_weight)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        interference_joints = results['interference_joints']

        target, target_weight = self._crowd_generate_target(
            results['ann_info'], joints_3d, joints_3d_visible,
            interference_joints)

        results['target'] = target
        results['target_weight'] = target_weight

        return results
