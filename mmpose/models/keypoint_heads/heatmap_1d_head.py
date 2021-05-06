import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmpose.models.builder import build_loss
from ..registry import HEADS


@HEADS.register_module()
class Heatmap1DHead(nn.Module):
    """Root depth head of paper ref: Gyeongsik Moon. "InterHand2.6M: A Dataset
    and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB
    Image".

    Args:
        in_channels (int): Number of input channels
        heatmap_size (int): Heatmap size
        hidden_dims (list|tuple): Number of feature dimension of FC layers.
        loss_value (dict): Config for heatmap 1d loss. Default: None.
    """

    def __init__(self,
                 in_channels=2048,
                 heatmap_size=64,
                 hidden_dims=(512, ),
                 loss_value=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.loss = build_loss(loss_value)
        self.in_channels = in_channels
        self.heatmap_size = heatmap_size

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        feature_dims = [in_channels] + \
                       [dim for dim in hidden_dims] + \
                       [heatmap_size]
        self.fc = self._make_linear_layers(feature_dims)

    def soft_argmax_1d(self, heatmap1d):
        heatmap1d = F.softmax(heatmap1d, 1)
        accu = heatmap1d * torch.arange(
            self.heatmap_size, dtype=heatmap1d.dtype,
            device=heatmap1d.device)[None, :]
        coord = accu.sum(dim=1)
        return coord

    def _make_linear_layers(self, feat_dims, relu_final=True):
        """Make linear layers."""
        layers = []
        for i in range(len(feat_dims) - 1):
            layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
            if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2
                                          and relu_final):
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        heatmap1d = self.fc(x)
        value = self.soft_argmax_1d(heatmap1d).view(-1, 1)
        return value

    def get_loss(self, output, target, target_weight):
        """Calculate regression loss of heatmap.

        Note:
            batch size: N

        Args:
            output (torch.Tensor[N, 1]): Output depth.
            target (torch.Tensor[N, 1]): Target depth.
            target_weight (torch.Tensor[N, 1]):
                Weights across different data.
        """

        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 2 and target_weight.dim() == 2
        losses['value_loss'] = self.loss(output, target, target_weight)
        return losses

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_labels (np.ndarray): Output labels.

        Args:
            x (torch.Tensor[NxC]): Input features vector.
            flip_pairs (None | list[tuple()):
                Pairs of labels which are mirrored.
        """
        value = self.forward(x).detach().cpu().numpy()

        if flip_pairs is not None:
            value_flipped_back = value.copy()
            for left, right in flip_pairs:
                value_flipped_back[:, left, ...] = value[:, right, ...]
                value_flipped_back[:, right, ...] = value[:, left, ...]
            return value_flipped_back
        return value

    def decode(self, img_metas, output, **kwargs):
        """Decode heatmap 1d values.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
            output (np.ndarray[N, 1]): model predicted values.
        """
        return dict(values=output)

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
