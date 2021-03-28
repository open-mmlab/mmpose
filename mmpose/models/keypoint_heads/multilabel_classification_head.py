import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmpose.models.builder import build_loss
from ..registry import HEADS


@HEADS.register_module()
class MultilabelClassificationHead(nn.Module):
    """Multi-label classification head. Paper ref: Gyeongsik Moon.
    "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image".

    Args:
        in_channels (int): Number of input channels
        num_labels (int): Number of labels
        hidden_dims (list|tuple): Number of hidden dimension of FC layers.
        loss_classification (dict): Config for classification loss.
          Default: None.
    """

    def __init__(self,
                 in_channels=2048,
                 num_labels=2,
                 hidden_dims=(512, ),
                 loss_classification=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.loss = build_loss(loss_classification)
        self.in_channels = in_channels
        self.num_labesl = num_labels

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        feature_dims = [in_channels] + \
                       [dim for dim in hidden_dims] + \
                       [num_labels]
        self.fc = self._make_linear_layers(feature_dims)

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
        labels = torch.sigmoid(self.fc(x))
        return labels

    def get_loss(self, output, target, target_weight):
        """Calculate regression loss of root depth.

        Note:
            batch_size: N

        Args:
            output (torch.Tensor[N, 1]): Output depth.
            target (torch.Tensor[N, 1]): Target depth.
            target_weight (torch.Tensor[N, 1]):
                Weights across different data.
        """

        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 2 and target_weight.dim() == 2
        losses['classification_loss'] = self.loss(output, target,
                                                  target_weight)
        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for classification.

        Note:
            batch size: N
            number labels: L

        Args:
            output (torch.Tensor[N, L]): Output hand visibility.
            target (torch.Tensor[N, L]): Target hand visibility.
            target_weight (torch.Tensor[N, L]):
                Weights across different labels.
        """

        accuracy = dict()
        # only calculate accuracy on the samples with ground truth labels
        valid = (target_weight > 0).min(dim=1)[0]

        output, target = output[valid], target[valid]

        if output.shape[0] == 0:
            # when no samples with gt labels, set acc to 0.
            acc = output.new_zeros(1)
        else:
            acc = (((output - 0.5) *
                    (target - 0.5)).min(dim=1)[0] > 0).float().mean()
        accuracy['acc_classification'] = acc
        return accuracy

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_labels (np.ndarray): Output labels.

        Args:
            x (torch.Tensor[NxC]): Input features vector.
            flip_pairs (None | list[tuple()]):
                Pairs of labels which are mirrored.
        """
        labels = self.forward(x).detach().cpu().numpy()

        if flip_pairs is not None:
            labels_flipped_back = labels.copy()
            for left, right in flip_pairs:
                labels_flipped_back[:, left, ...] = labels[:, right, ...]
                labels_flipped_back[:, right, ...] = labels[:, left, ...]
            return labels_flipped_back

        return labels

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file": path to the image file
            output (np.ndarray[N, L]): model predicted labels.
        """
        return dict(labels=output)

    def init_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, mean=0, std=0.01, bias=0)
