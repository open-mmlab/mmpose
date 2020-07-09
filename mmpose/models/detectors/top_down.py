import numpy as np
import torch

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.evaluation.acc import keypoints_from_heatmaps
from mmpose.core.post_processing import flip_back
from .. import builder
from ..registry import POSENETS
from .base import BasePose


@POSENETS.register_module()
class TopDown(BasePose):
    """Top-down pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 backbone,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_pose=None):
        super(TopDown, self).__init__()

        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:
            self.keypoint_head = builder.build_head(keypoint_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss = builder.build_loss(loss_pose)
        self.init_weights(pretrained=pretrained)

    @property
    def with_keypoint(self):
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):

        self.backbone.init_weights(pretrained)
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    def forward(self,
                img,
                target=None,
                target_weight=None,
                img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img weight: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (torch.Tensor[NxCximgHximgW]): Input images.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]): Weights across
                different joint types.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.

        Returns:
            if `return loss` is true, then return losses. Otherwise, return
            predicted poses, boxes and image paths.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if isinstance(output, list):
            # multi-stage models
            for i in range(len(output)):
                if 'mse_loss' not in losses:
                    losses['mse_loss'] = self.loss(output[i], target,
                                                   target_weight)
                else:
                    losses['mse_loss'] += self.loss(output[i], target,
                                                    target_weight)
        else:
            losses['mse_loss'] = self.loss(output, target, target_weight)

        if isinstance(output, list):
            _, avg_acc, cnt = pose_pck_accuracy(
                output[-1][target_weight.squeeze(-1) > 0].unsqueeze(
                    0).detach().cpu().numpy(),
                target[target_weight.squeeze(-1) > 0].unsqueeze(
                    0).detach().cpu().numpy())
        else:
            _, avg_acc, cnt = pose_pck_accuracy(
                output[target_weight.squeeze(-1) > 0].unsqueeze(
                    0).detach().cpu().numpy(),
                target[target_weight.squeeze(-1) > 0].unsqueeze(
                    0).detach().cpu().numpy())

        losses['acc_pose'] = float(avg_acc)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        assert img.size(0) == 1
        assert len(img_metas) == 1

        img_metas = img_metas[0]

        flip_pairs = img_metas['flip_pairs']
        # compute output
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        if isinstance(output, list):
            output = output[-1]

        if self.test_cfg['flip_test']:
            img_flipped = img.flip(3)

            output_flipped = self.backbone(img_flipped)
            if self.with_keypoint:
                output_flipped = self.keypoint_head(output_flipped)
            if isinstance(output_flipped, list):
                output_flipped = output_flipped[-1]
            output_flipped = flip_back(output_flipped.cpu().numpy(),
                                       flip_pairs)

            output_flipped = torch.from_numpy(output_flipped.copy()).to(
                output.device)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg['shift_heatmap']:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]
            output = (output + output_flipped) * 0.5

        c = img_metas['center'].reshape(1, -1)
        s = img_metas['scale'].reshape(1, -1)
        score = np.array(img_metas['bbox_score']).reshape(-1)

        preds, maxvals = keypoints_from_heatmaps(
            output.clone().cpu().numpy(),
            c,
            s,
            post_process=self.test_cfg['post_process'],
            unbiased=self.test_cfg['unbiased_decoding'],
            kernel=self.test_cfg['modulate_kernel'])

        all_preds = np.zeros((1, output.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((1, 6))
        image_path = []

        all_preds[0, :, 0:2] = preds[:, :, 0:2]
        all_preds[0, :, 2:3] = maxvals
        all_boxes[0, 0:2] = c[:, 0:2]
        all_boxes[0, 2:4] = s[:, 0:2]
        all_boxes[0, 4] = np.prod(s * 200.0, 1)
        all_boxes[0, 5] = score
        image_path.extend(img_metas['image_file'])

        return all_preds, all_boxes, image_path
