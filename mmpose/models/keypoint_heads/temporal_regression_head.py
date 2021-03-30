import numpy as np
import torch.nn as nn
from mmcv.cnn import build_conv_layer, normal_init

from mmpose.core.evaluation import compute_similarity_transform
from mmpose.core.post_processing import fliplr_regression
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
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints

        self.loss = build_loss(loss_keypoint)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.conv = build_conv_layer(
            dict(type='Conv1d'), in_channels, num_joints * 3, 1)

    def forward(self, x):
        """Forward function."""
        assert x.ndim == 3 and x.shape[2] == 1, 'Invalid shape {x.shape}'
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
        assert target.dim() == 3 and target_weight.dim() == 3
        losses['reg_loss'] = self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, 3]): Output keypoints.
            target (torch.Tensor[N, K, 3]): Target keypoints.
            target_weight (torch.Tensor[N, K, 3]):
                Weights across different joint types.
        """

        accuracy = dict()

        N = output.shape[0]
        output_ = output.detach().cpu().numpy()
        target_ = target.detach().cpu().numpy()
        target_weight_ = target_weight.detach().cpu().numpy()

        mpjpe = np.mean(
            np.linalg.norm((output_ - target_) * target_weight_, axis=-1))

        transformed_output = np.zeros_like(output_)
        for i in range(N):
            transformed_output[i, :, :] = compute_similarity_transform(
                output_[i, :, :], target_[i, :, :])
        p_mpjpe = np.mean(
            np.linalg.norm(
                (transformed_output - target_) * target_weight_, axis=-1))

        accuracy['mpjpe'] = mpjpe
        accuracy['p_mpjpe'] = p_mpjpe

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
        target_image_paths = []
        for i in range(len(metas)):
            target_image_paths.append(metas[i]['target_image_path'])

        result = {'preds': output, 'target_image_paths': target_image_paths}

        return result

    def init_weights(self):
        """Initialize model weights."""
        normal_init(self.conv, mean=0, std=0.001, bias=0)
