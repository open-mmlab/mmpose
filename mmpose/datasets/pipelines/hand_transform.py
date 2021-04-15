import numpy as np

from mmpose.core.post_processing import fliplr_joints
from mmpose.datasets.pipelines import Compose
from mmpose.datasets.registry import PIPELINES


@PIPELINES.register_module()
class HandRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth' and 'ann_info'.
    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth'.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']
        hand_type = results['hand_type']
        rel_root_depth = results['rel_root_depth']

        if np.random.rand() <= self.flip_prob:
            img = img[:, ::-1, :]

            joints_3d, joints_3d_visible = fliplr_joints(
                joints_3d, joints_3d_visible, img.shape[1],
                results['ann_info']['flip_pairs'])
            center[0] = img.shape[1] - center[0] - 1
            hand_type[0], hand_type[1] = hand_type[1].copy(
            ), hand_type[0].copy()
            rel_root_depth = -rel_root_depth

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center
        results['hand_type'] = hand_type
        results['rel_root_depth'] = rel_root_depth

        return results


@PIPELINES.register_module()
class HandGetRandomTranslation:
    """Data augmentation with random translation.

    Required key: 'scale' and 'center'. Modifies key: 'center'.

    Args:
        trans_factor (float): Translating center to
        ``[-trans_factor, -trans_factor] * scale + center``.
    """

    def __init__(self, trans_factor=0.15):
        self.trans_factor = trans_factor

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        center = results['center']
        scale = results['scale']
        center += self.trans_factor * (2 * np.random.rand(2) - 1) * scale * 200
        results['center'] = center
        return results


@PIPELINES.register_module()
class HandGenerate3DHeatmapTarget:
    """Generate the target 3d heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.
    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian.
    """

    def __init__(self, sigma=2):
        self.sigma = sigma

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        cfg = results['ann_info']
        image_size = cfg['image_size']
        W, H, D = cfg['heatmap_size']
        bbox_depth_size = cfg['bbox_depth_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        mu_x = joints_3d[:, 0] * W / image_size[0]
        mu_y = joints_3d[:, 1] * H / image_size[1]
        mu_z = (joints_3d[:, 2] / bbox_depth_size + 0.5) * D

        target_weight = joints_3d_visible[:, 0]
        target_weight = target_weight * (mu_z >= 0) * (mu_z < D)
        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)
        target_weight = target_weight[:, None]

        x, y, z = np.arange(W), np.arange(H), np.arange(D)
        zz, yy, xx = np.meshgrid(z, y, x)
        xx = xx[None, ...].astype(np.float32)
        yy = yy[None, ...].astype(np.float32)
        zz = zz[None, ...].astype(np.float32)

        mu_x = mu_x[..., None, None, None]
        mu_y = mu_y[..., None, None, None]
        mu_z = mu_z[..., None, None, None]

        target = np.exp(-((xx - mu_x)**2 + (yy - mu_y)**2 + (zz - mu_z)**2) /
                        (2 * self.sigma**2))

        results['target'] = target
        results['target_weight'] = target_weight
        return results


@PIPELINES.register_module()
class HandGenerateDepthTarget:
    """Generate the target relative root depth.

    Required keys: 'rel_root_depth', 'rel_root_valid', 'ann_info'. Modified
    keys: 'target', 'target_weight'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Generate the target heatmap."""
        rel_root_depth = results['rel_root_depth']
        rel_root_valid = results['rel_root_valid']
        cfg = results['ann_info']
        D = cfg['heatmap_size_root']
        depth_size = cfg['bbox_depth_size_root']
        target = (rel_root_depth / depth_size + 0.5) * D
        target_weight = rel_root_valid * (target >= 0) * (target <= D)
        results['target'] = target * np.ones(1, dtype=np.float32)
        results['target_weight'] = target_weight * np.ones(1, dtype=np.float32)
        return results


@PIPELINES.register_module()
class HandGenerateLabelTarget:
    """Generate the target hand type label.

    Required keys: 'hand_type', 'hand_type_valid'. Modified keys: 'target',
    'target_weight'.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Generate the target hand type label."""
        target = results['hand_type']
        target_weight = np.ones(
            target.shape, dtype=np.float32) * results['hand_type_valid']
        results['target'] = target
        results['target_weight'] = target_weight
        return results


@PIPELINES.register_module()
class MultitaskGatherTarget:
    """Gather the targets for multitask heads.

    Args:
        pipeline_list (list[list]): List of pipelines for each head.
    """

    def __init__(self, pipeline_list):
        self.pipelines = []
        for pipeline in pipeline_list:
            self.pipelines.append(Compose(pipeline))

    def __call__(self, results):
        target = []
        target_weight = []
        for pipeline in self.pipelines:
            results_head = pipeline(results)
            target.append(results_head['target'])
            target_weight.append(results_head['target_weight'])
        results['target'] = target
        results['target_weight'] = target_weight
        return results
