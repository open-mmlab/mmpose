import numpy as np

from mmpose.datasets.registry import PIPELINES
from .top_down_transform import TopDownRandomFlip


@PIPELINES.register_module()
class HandRandomFlip(TopDownRandomFlip):
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth' and 'ann_info'.
    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center',
    'hand_type', 'rel_root_depth'.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        super().__init__(flip_prob=flip_prob)

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        # base flip augment
        super().__call__(results)

        # flip hand type and root depth
        hand_type = results['hand_type']
        rel_root_depth = results['rel_root_depth']
        if self.flip_flag:
            hand_type[0], hand_type[1] = hand_type[1], hand_type[0]
            rel_root_depth = -rel_root_depth
        results['hand_type'] = hand_type
        results['rel_root_depth'] = rel_root_depth
        return results


@PIPELINES.register_module()
class HandGenerateRelDepthTarget:
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
