import numpy as np
import torch.nn as nn
from mmcv.cnn import build_conv_layer, constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmpose.core import (WeightNormClipRegister, compute_similarity_transform,
                         fliplr_regression)
from mmpose.models.builder import build_loss
from mmpose.models.registry import HEADS


@HEADS.register_module()
class TemporalRegressionHead(nn.Module):
    """Regression head of VideoPose3D.

    Paper ref: Dario Pavllo.
    ``3D human pose estimation in video with temporal convolutions and
     semi-supervised training``

     Args:
         in_channels (int): Number of input channels
         num_joints (int): Number of joints
         loss_keypoint (dict): Config for keypoint loss. Default: None.
         max_norm (float|None): if not None, the weight of convolution layers
            will be clipped to have a maximum norm of max_norm.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 max_norm=None,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.max_norm = max_norm
        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.conv = build_conv_layer(
            dict(type='Conv1d'), in_channels, num_joints * 3, 1)

        if self.max_norm is not None:
            # Apply weight norm clip to conv layers
            weight_clip = WeightNormClipRegister(self.max_norm)
            for module in self.modules():
                if isinstance(module, nn.modules.conv._ConvNd):
                    weight_clip.register(module)

    def _transform_inputs(self, x):
        """Transform inputs for decoder.

        Args:
            inputs (tuple/list of Tensor | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(x, (list, tuple)):
            return x

        assert len(x) > 0
        return x[-1]

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)

        assert x.ndim == 3 and x.shape[2] == 1, f'Invalid shape {x.shape}'
        output = self.conv(x)
        N = output.shape[0]
        return output.reshape(N, self.num_joints, 3)

    def get_loss(self, output, target, target_weight):
        """Calculate keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        if target_weight is None:
            target_weight = target.new_ones(target.shape)
        assert target.dim() == 3 and target_weight.dim() == 3
        losses['reg_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight, metas):
        """Calculate accuracy for keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
            metas (list(dict)): Information about data augmentation.
                By default this includes:
        """

        accuracy = dict()

        N = output.shape[0]
        output_ = output.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        if target_weight is None:
            target_weight_ = np.ones_like(target_)
        else:
            target_weight_ = target_weight.detach().cpu().numpy()

        # Denormalize the predicted pose
        if 'target_mean' in metas[0] and 'target_std' in metas[0]:
            target_mean = np.stack(_['target_mean'] for _ in metas)
            target_std = np.stack(_['target_std'] for _ in metas)
            output_ = self._denormalize_joints(output_, target_mean,
                                               target_std)

        mpjpe = np.mean(
            np.linalg.norm((output_ - target_) * target_weight_, axis=-1))

        transformed_output = np.zeros_like(output_)
        for i in range(N):
            transformed_output[i, :, :] = compute_similarity_transform(
                output_[i, :, :], target_[i, :, :])
        p_mpjpe = np.mean(
            np.linalg.norm(
                (transformed_output - target_) * target_weight_, axis=-1))

        accuracy['mpjpe'] = output.new_tensor(mpjpe)
        accuracy['p_mpjpe'] = output.new_tensor(p_mpjpe)

        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_regression (np.ndarray): Output regression.

        Args:
            x (torch.Tensor[N, K, 2]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if flip_pairs is not None:
            output_regression = fliplr_regression(
                output.detach().cpu().numpy(),
                flip_pairs,
                center_mode='static',
                center_x=0)
        else:
            output_regression = output.detach().cpu().numpy()
        return output_regression

    def decode(self, metas, output):
        """Decode the keypoints from output regression.

        Args:
            metas (list(dict)): Information about data augmentation.
                By default this includes:
                - "target_image_path": path to the image file
            output (np.ndarray[N, K, 3]): predicted regression vector.
        """

        # Denormalize the predicted pose
        if 'target_mean' in metas[0] and 'target_std' in metas[0]:
            target_mean = np.stack(_['target_mean'] for _ in metas)
            target_std = np.stack(_['target_std'] for _ in metas)
            output = self._denormalize_joints(output, target_mean, target_std)

        # Restore global position
        if 'global_position' in metas[0]:
            global_position = np.stack(_['global_position'] for _ in metas)
            output += global_position
            if 'global_position_index' in metas[0]:
                idx = metas[0]['global_position_index']
                assert all(_['global_position_index'] == idx for _ in metas)
                output = np.insert(
                    output, idx, global_position.squeeze(-2), axis=-2)

        target_image_paths = [
            _m.get('target_image_path', None) for _m in metas
        ]
        result = {'preds': output, 'target_image_paths': target_image_paths}

        return result

    @staticmethod
    def _denormalize_joints(x, mean, std):
        """Denormalize joint coordinates with given statistics mean and std.

        Args:
            x (np.ndarray[N, K, 3]): normalized joint coordinates
            mean (np.ndarray[K, 3]): mean value
            std (np.ndarray[K, 3]): std value
        """
        assert x.ndim == 3
        assert x.shape == mean.shape == std.shape

        return x * std + mean

    def init_weights(self):
        """Initialize the weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
