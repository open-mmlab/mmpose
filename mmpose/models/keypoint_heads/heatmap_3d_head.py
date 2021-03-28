import numpy as np
import torch.nn as nn

from mmpose.core.post_processing import flip_back, transform_preds
from ..registry import HEADS
from .top_down_simple_head import TopDownSimpleHead


def _get_max_preds_3d(heatmaps):
    """Get keypoint predictions from 3D score maps.

    Note:
        batch size: N
        num keypoints: K
        heatmap depth size: D
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, D, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.
        - preds (np.ndarray[N, K, 3]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps, np.ndarray), \
        ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 5, 'heatmaps should be 5-ndim'

    N, K, D, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.zeros((N, K, 3), dtype=np.float32)
    _idx = idx[..., 0]
    preds[:, :, 2], _idx = _idx // (H * W), _idx % (H * W)
    preds[:, :, 1], _idx = _idx // W, _idx % W
    preds[:, :, 0] = _idx

    preds = np.where(np.tile(maxvals, (1, 1, 3)) > 0.0, preds, -1)
    return preds, maxvals


@HEADS.register_module()
class HeatMap3DHead(TopDownSimpleHead):
    """3D heatmap head of paper ref: Gyeongsik Moon. "InterHand2.6M: A Dataset
    and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB
    Image" HeatMap3DHead is a variant of TopDownSimpleHead, and is composed of
    (>=0) number of deconv layers and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        depth_size (int): Number of depth discretization size
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth_size=64,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None):

        super().__init__(in_channels, out_channels, num_deconv_layers,
                         num_deconv_filters, num_deconv_kernels, extra,
                         in_index, input_transform, align_corners,
                         loss_keypoint, train_cfg, test_cfg)
        assert out_channels % depth_size == 0
        self.depth_size = depth_size

    def get_loss(self, output, target, target_weight):
        """Calculate 3D heatmap loss.

        Note:
            batch size: N
            num keypoints: K
            heatmaps depth size: D
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxDxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxDxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        losses = dict()
        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 5 and target_weight.dim() == 3
        losses['heatmap_loss'] = self.loss(output, target, target_weight)
        return losses

    # The accuracy of interhand keypoints is not well defined.
    def get_accuracy(self, output, target, target_weight):
        return {}

    def forward(self, x):
        """Forward function."""
        x = super().forward(x)
        N, C, H, W = x.shape
        # reshape the 2D heatmap to 3D heatmap
        x = x.reshape(N, C // self.depth_size, self.depth_size, H, W)
        return x

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, D, H, W]): model predicted 3D heatmaps.
        """
        batch_size = len(img_metas)
        N, K, D, H, W = output.shape

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        center = np.zeros((batch_size, 2), dtype=np.float32)
        scale = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size, dtype=np.float32)
        for i in range(batch_size):
            center[i, :] = img_metas[i]['center']
            scale[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = _get_max_preds_3d(output)
        # Transform back to the image
        for i in range(N):
            preds[i, :, :2] = transform_preds(preds[i, :, :2], center[i],
                                              scale[i], [W, H])

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = center[:, 0:2]
        all_boxes[:, 2:4] = scale[:, 0:2]
        # scale is defined as: bbox_size / 200.0,
        # so we need multiply 200.0 to get  bbox size
        all_boxes[:, 4] = np.prod(scale * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}
        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids
        return result

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[NxKxHxW]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """

        output = self.forward(x)

        if flip_pairs is not None:
            N, K, D, H, W = output.shape
            # reshape 3D heatmap to 2D heatmap
            output = output.reshape(N, K * D, H, W)
            # 2D heatmap flip
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # reshape back to 3D heatmap
            output_heatmap = output_heatmap.reshape(N, K, D, H, W)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[..., 1:] = output_heatmap[..., :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap
