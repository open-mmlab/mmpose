# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from torchvision.transforms import functional as F

from mmpose.datasets.builder import PIPELINES
from .top_down_transform import TopDownRandomFlip


@PIPELINES.register_module()
class HandRandomFlip(TopDownRandomFlip):
    """Data augmentation with random image flip. A child class of
    TopDownRandomFlip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth' and 'ann_info'.

    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth'.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        # base flip augmentation
        super().__call__(results)

        # flip hand type and root depth
        hand_type = results['hand_type']
        rel_root_depth = results['rel_root_depth']
        flipped = results['flipped']
        if flipped:
            hand_type[0], hand_type[1] = hand_type[1], hand_type[0]
            rel_root_depth = -rel_root_depth
        results['hand_type'] = hand_type
        results['rel_root_depth'] = rel_root_depth
        return results


@PIPELINES.register_module()
class HandGenerateRelDepthTarget:
    """Generate the target relative root depth.

    Required keys: 'rel_root_depth', 'rel_root_valid', 'ann_info'.

    Modified keys: 'target', 'target_weight'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Generate the target heatmap."""
        rel_root_depth = results['rel_root_depth']
        rel_root_valid = results['rel_root_valid']
        cfg = results['ann_info']
        D = cfg['heatmap_size_root']
        root_depth_bound = cfg['root_depth_bound']
        target = (rel_root_depth / root_depth_bound + 0.5) * D
        target_weight = rel_root_valid * (target >= 0) * (target <= D)
        results['target'] = target * np.ones(1, dtype=np.float32)
        results['target_weight'] = target_weight * np.ones(1, dtype=np.float32)
        return results


@PIPELINES.register_module()
class DepthToTensor:
    """Transform depth image to Tensor.
    TODO: add reference from AWR github

    Required key: 'img', 'cube_size', 'center_depth'.

    Modifies key: 'img'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        if isinstance(results['img'], (list, tuple)):
            results['img'] = [
                F.to_tensor(self._process_depth(img, results))
                for img in results['img']
            ]
        else:
            depth = self._process_depth(results['img'], results)
            results['img'] = F.to_tensor(depth)
        return results

    @staticmethod
    def _process_depth(img, results):
        depth = np.asarray(img[:, :, 0] + img[:, :, 1] * 256, dtype=np.float32)
        img_max = np.max(depth)
        depth_max = results['center_depth'] + (results['cube_size'][2] / 2.)
        depth_min = results['center_depth'] - (results['cube_size'][2] / 2.)
        depth[depth == img_max] = depth_max
        depth[depth == 0] = depth_max
        depth = np.clip(depth, depth_min, depth_max)
        depth = (depth - results['center_depth']) / (
            results['cube_size'][2] / 2.)
        return depth


@PIPELINES.register_module()
class HandGenerateJointToOffset:
    """"""

    def __init__(self, heatmap_kernel_size):
        self.heatmap_kernel_size = heatmap_kernel_size

    def __call__(self, results):
        cfg = results['ann_info']
        feature_size = cfg['heatmap_size']
        joint_uvd = results['target']  # UV -1,1
        num_joints = joint_uvd.shape[0]

        img = results['img']
        depth = img.numpy()[0]  # it is a hack

        coord_x = (2.0 * (np.arange(feature_size[0]) + 0.5) / feature_size[0] -
                   1.0).astype(np.float32)
        coord_y = (2.0 * (np.arange(feature_size[1]) + 0.5) / feature_size[1] -
                   1.0).astype(np.float32)
        xv, yv = np.meshgrid(coord_x, coord_y)
        coord = np.stack((xv, yv), 0)
        depth_resize = mmcv.imresize(
            depth, (feature_size[0], feature_size[1]), interpolation='nearest')
        depth_resize = np.expand_dims(depth_resize, 0)
        coord_with_depth = np.expand_dims(
            np.concatenate((coord, depth_resize), 0), 0)
        jt_ft = np.broadcast_to(joint_uvd[:, :, np.newaxis, np.newaxis],
                                (joint_uvd.shape[0], joint_uvd.shape[1],
                                 feature_size[0], feature_size[1]))
        offset = jt_ft - coord_with_depth  # [jt_num, 3, F, F]
        dis = np.linalg.norm(offset + 1e-8, axis=1)  # [jt_num, F, F]
        offset_norm = offset / dis[:, np.newaxis, ...]  # value in [-1, 1]
        heatmap = (self.heatmap_kernel_size -
                   dis) / self.heatmap_kernel_size  # [jt_num, F, F]
        mask = (heatmap > 0).astype(np.float32) * (depth_resize < 0.99).astype(
            np.float32)  # [jt_num, F, F]
        offset_norm_mask = (offset_norm * mask[:, None, ...]).reshape(
            -1, feature_size[0], feature_size[1])
        heatmap_mask = heatmap * mask
        offset_field = np.concatenate((offset_norm_mask, heatmap_mask),
                                      axis=0)  # [jt_num*4, F, F]
        results['target'] = offset_field
        results['target_weight'] = np.ones(num_joints)
        return results
