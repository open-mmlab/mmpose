import numpy as np
import torch

from mmpose.core.evaluation.top_down_eval import crowd_keypoints_from_heatmaps
from mmpose.core.post_processing import flip_back
from ..registry import POSENETS
from .top_down import TopDown


@POSENETS.register_module()
class TopDownCrowd(TopDown):
    """Top-down crowd pose detectors.

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
        super().__init__(backbone, keypoint_head, train_cfg, test_cfg,
                         pretrained, loss_pose)

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at every call when testing."""
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

        score = 1.0
        if 'bbox_score' in img_metas:
            score = np.array(img_metas['bbox_score']).reshape(-1)

        preds, maxvals = crowd_keypoints_from_heatmaps(
            output.clone(),
            c,
            s,
            post_process=self.test_cfg['post_process'],
            unbiased=self.test_cfg['unbiased_decoding'],
            kernel=self.test_cfg['modulate_kernel'])

        all_preds = np.zeros((1, output.shape[1], 5, 3), dtype=np.float32)
        all_boxes = np.zeros((1, 6))
        image_path = []

        all_preds[0, :, :, 0:2] = preds[:, :, :, 0:2]
        all_preds[0, :, :, 2:3] = maxvals
        all_boxes[0, 0:2] = c[:, 0:2]
        all_boxes[0, 2:4] = s[:, 0:2]
        all_boxes[0, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[0, 5] = score
        image_path.extend(img_metas['image_file'])

        return all_preds, all_boxes, image_path
