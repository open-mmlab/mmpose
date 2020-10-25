import math

import cv2
import mmcv
import numpy as np
import torch.nn as nn
from mmcv.image import imwrite
from mmcv.visualization.image import imshow

from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.evaluation.top_down_eval import keypoints_from_heatmaps
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
        super().__init__()

        self.backbone = builder.build_backbone(backbone)

        if keypoint_head is not None:
            self.keypoint_head = builder.build_head(keypoint_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss = builder.build_loss(loss_pose)
        self.init_weights(pretrained=pretrained)

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
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
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes and image paths.
        """
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        """Defines the computation performed at every call when training."""
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if isinstance(output, list):
            if target.dim() == 5 and target_weight.dim() == 4:
                # target: [batch_size, num_outputs, num_joints, h, w]
                # target_weight: [batch_size, num_outputs, num_joints, 1]
                assert target.size(1) == len(output)
            if isinstance(self.loss, nn.Sequential):
                assert len(self.loss) == len(output)
            if 'loss_weights' in self.train_cfg and self.train_cfg[
                    'loss_weights'] is not None:
                assert len(self.train_cfg['loss_weights']) == len(output)
            for i in range(len(output)):
                if target.dim() == 5 and target_weight.dim() == 4:
                    target_i = target[:, i, :, :, :]
                    target_weight_i = target_weight[:, i, :, :]
                else:
                    target_i = target
                    target_weight_i = target_weight
                if isinstance(self.loss, nn.Sequential):
                    loss_func = self.loss[i]
                else:
                    loss_func = self.loss

                loss_i = loss_func(output[i], target_i, target_weight_i)
                if 'loss_weights' in self.train_cfg and self.train_cfg[
                        'loss_weights']:
                    loss_i = loss_i * self.train_cfg['loss_weights'][i]
                if 'mse_loss' not in losses:
                    losses['mse_loss'] = loss_i
                else:
                    losses['mse_loss'] += loss_i
        else:
            assert not isinstance(self.loss, nn.Sequential)
            assert target.dim() == 4 and target_weight.dim() == 3
            # target: [batch_size, num_joints, h, w]
            # target_weight: [batch_size, num_joints, 1]
            losses['mse_loss'] = self.loss(output, target, target_weight)

        if isinstance(output, list):
            if target.dim() == 5 and target_weight.dim() == 4:
                _, avg_acc, _ = pose_pck_accuracy(
                    output[-1][target_weight[:, -1, :, :].squeeze(-1) > 0].
                    unsqueeze(0).detach().cpu().numpy(),
                    target[:, -1, :, :, :][target_weight[:, -1, :, :].squeeze(
                        -1) > 0].unsqueeze(0).detach().cpu().numpy())
                # Only use the last output for prediction
            else:
                _, avg_acc, _ = pose_pck_accuracy(
                    output[-1][target_weight.squeeze(-1) > 0].unsqueeze(
                        0).detach().cpu().numpy(),
                    target[target_weight.squeeze(-1) > 0].unsqueeze(
                        0).detach().cpu().numpy())
        else:
            _, avg_acc, _ = pose_pck_accuracy(
                output[target_weight.squeeze(-1) > 0].unsqueeze(
                    0).detach().cpu().numpy(),
                target[target_weight.squeeze(-1) > 0].unsqueeze(
                    0).detach().cpu().numpy())
        losses['acc_pose'] = float(avg_acc)

        return losses

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

        output_heatmap = output.detach().cpu().numpy()
        if self.test_cfg['flip_test']:
            img_flipped = img.flip(3)

            output_flipped = self.backbone(img_flipped)
            if self.with_keypoint:
                output_flipped = self.keypoint_head(output_flipped)
            if isinstance(output_flipped, list):
                output_flipped = output_flipped[-1]
            output_flipped = flip_back(output_flipped.detach().cpu().numpy(),
                                       flip_pairs)

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg['shift_heatmap']:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]
            output_heatmap = (output_heatmap + output_flipped) * 0.5

        c = img_metas['center'].reshape(1, -1)
        s = img_metas['scale'].reshape(1, -1)

        score = 1.0
        if 'bbox_score' in img_metas:
            score = np.array(img_metas['bbox_score']).reshape(-1)

        preds, maxvals = keypoints_from_heatmaps(
            output_heatmap,
            c,
            s,
            post_process=self.test_cfg['post_process'],
            unbiased=self.test_cfg['unbiased_decoding'],
            kernel=self.test_cfg['modulate_kernel'])

        all_preds = np.zeros((1, output_heatmap.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((1, 6), dtype=np.float32)
        image_path = []

        all_preds[0, :, 0:2] = preds[:, :, 0:2]
        all_preds[0, :, 2:3] = maxvals
        all_boxes[0, 0:2] = c[:, 0:2]
        all_boxes[0, 2:4] = s[:, 0:2]
        all_boxes[0, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[0, 5] = score
        image_path.extend(img_metas['image_file'])

        return all_preds, all_boxes, image_path, output_heatmap

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    text_color=(255, 0, 0),
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        bbox_result = []
        pose_result = []
        for res in result:
            bbox_result.append(res['bbox'])
            pose_result.append(res['keypoints'])

        if len(bbox_result) > 0:
            bboxes = np.vstack(bbox_result)
            # draw bounding boxes
            mmcv.imshow_bboxes(
                img,
                bboxes,
                colors=bbox_color,
                top_k=-1,
                thickness=thickness,
                show=False,
                win_name=win_name,
                wait_time=wait_time,
                out_file=None)

            for person_id, kpts in enumerate(pose_result):
                # draw each point on image
                if pose_kpt_color is not None:
                    assert len(pose_kpt_color) == len(kpts)
                    for kid, kpt in enumerate(kpts):
                        x_coord, y_coord, kpt_score = int(kpt[0]), int(
                            kpt[1]), kpt[2]
                        if kpt_score > kpt_score_thr:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

                # draw limbs
                if skeleton is not None and pose_limb_color is not None:
                    assert len(pose_limb_color) == len(skeleton)
                    for sk_id, sk in enumerate(skeleton):
                        pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1,
                                                                  1]))
                        pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1,
                                                                  1]))
                        if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                                and pos1[1] < img_h and pos2[0] > 0
                                and pos2[0] < img_w and pos2[1] > 0
                                and pos2[1] < img_h
                                and kpts[sk[0] - 1, 2] > kpt_score_thr
                                and kpts[sk[1] - 1, 2] > kpt_score_thr):
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle),
                                0, 360, 1)

                            r, g, b = pose_limb_color[sk_id]
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0,
                                min(
                                    1, 0.5 *
                                    (kpts[sk[0] - 1, 2] + kpts[sk[1] - 1, 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
