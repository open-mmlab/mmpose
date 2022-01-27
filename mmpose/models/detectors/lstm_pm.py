# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import copy

from ..builder import POSENETS
from .top_down import TopDown

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class LSTMPoseMachine(TopDown):
    """Top-down pose detectors for LSTM Pose Machine.
    Paper ref: Luo, Yue, et al. "Lstm pose machines." Proceedings of the IEEE
    conference on computer vision and pattern recognition (2018).

    <``https://arxiv.org/abs/1712.06316``>

    A child class of TopDown detector.

    Args:
        backbone (dict): Backbone modules to extract features.
        neck (dict): intermediate modules to transform features.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
        concat_tensors (bool): Whether to concat the tensors on the batch dim,
            which can speed up, Default: True
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            keypoint_head=keypoint_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            loss_pose=loss_pose)

    @auto_fp16(apply_to=('img', ))
    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            number of frames: F
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            imgs (list[Fxtorch.Tensor[NxCximgHximgW]]): multiple input frames
            target (torch.Tensor[NxKxHxW]): Target heatmaps for one frame.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: paths to multiple video frames
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
                Otherwise, return predicted poses, boxes, image paths
                and heatmaps.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, imgs, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        # imgs (list[Fxtorch.Tensor[NxCximgHximgW]]): multiple input frames
        assert imgs[0].size(0) == len(img_metas)
        output = self.backbone(imgs)
        if self.with_neck:
            output = self.neck(output)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            target = torch.stack(target, dim=1)
            target_weight = torch.stack(target_weight, dim=1)
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)
            losses.update(keypoint_losses)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, imgs, img_metas, return_heatmap=False, **kwargs):
        """Defines the computation performed at every call when testing."""
        # imgs (list[Fxtorch.Tensor[NxCximgHximgW]]): multiple input frames
        assert imgs[0].size(0) == len(img_metas)
        batch_size, _, img_height, img_width = imgs[0].shape
        if batch_size > 1:
            assert 'bbox_id' in img_metas[0]

        result = {}

        features = self.backbone(imgs)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output_heatmap = self.keypoint_head.inference_model(
                features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            imgs_flipped = [img.flip(3) for img in imgs]
            features_flipped = self.backbone(imgs_flipped)
            if self.with_neck:
                features_flipped = self.neck(features_flipped)
            if self.with_keypoint:
                output_flipped_heatmap = self.keypoint_head.inference_model(
                    features_flipped, img_metas[0]['flip_pairs'])
                for i in range(len(output_heatmap)):
                    output_heatmap[i] = (output_heatmap[i] +
                                         output_flipped_heatmap[i]) * 0.5

        if self.with_keypoint:
            meta_keys = ['image_file', 'center', 'scale', 'bbox_score', 'bbox_id']
            batch_size = len(img_metas)
            num_frame = len(img_metas[0]['image_file'])
            for f in range(num_frame):
                test_metas = copy.deepcopy(img_metas)
                for i in range(batch_size):
                    for key in meta_keys:
                        test_metas[i][key] = img_metas[i][key][f]
                keypoint_result = self.keypoint_head.decode(
                    test_metas, output_heatmap[f], img_size=[img_width, img_height])

                if result == {}:
                    result.update(keypoint_result)
                else:
                    for key in result.keys():
                        result[key] = np.concatenate((result[key],
                                                      keypoint_result[key]), axis=0)

            if not return_heatmap:
                output_heatmap = None
            else:
                output_heatmap = np.concatenate(output_heatmap, axis=0)
            result['output_heatmap'] = output_heatmap

        return result
