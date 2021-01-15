import math
import warnings

import cv2
import mmcv
import numpy as np
import torch
from mmcv.image import imwrite
from mmcv.visualization.image import imshow

from mmpose.core.evaluation import (aggregate_results, get_group_preds,
                                    get_multi_stage_outputs)
from mmpose.core.post_processing.group import HeatmapParser
from .. import builder
from ..registry import POSENETS
from .base import BasePose


@POSENETS.register_module()
class BottomUp(BasePose):
    """Bottom-up pose detectors.

    Args:
        backbone (dict): Backbone modules to extract feature.
        keypoint_head (dict): Keypoint head to process feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained models.
        loss_pose (None): Deprecated arguments. Please use
            `loss_keypoint` for heads instead.
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

            if 'loss_keypoint' not in keypoint_head and loss_pose is not None:
                warnings.warn(
                    '`loss_pose` for BottomUp is deprecated, '
                    'use `loss_keypoint` for heads instead. See '
                    'https://github.com/open-mmlab/mmpose/pull/382'
                    ' for more information.', DeprecationWarning)
                keypoint_head['loss_keypoint'] = loss_pose

            self.keypoint_head = builder.build_head(keypoint_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_udp = test_cfg.get('use_udp', False)
        self.parser = HeatmapParser(self.test_cfg)
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
                img=None,
                targets=None,
                masks=None,
                joints=None,
                img_metas=None,
                return_loss=True,
                return_heatmap=False,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss is True.
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_weight: imgW
            img_height: imgH
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
        Args:
            img(torch.Tensor[NxCximgHximgW]): Input image.
            targets(List(torch.Tensor[NxKxHxW])): Multi-scale target heatmaps.
            masks(List(torch.Tensor[NxHxW])): Masks of multi-scale target
                                              heatmaps
            joints(List(torch.Tensor[NxMxKx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas(dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

            return loss(bool): Option to 'return_loss'. 'return_loss=True' for
                training, 'return_loss=False' for validation & test
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if 'return_loss' is true, then return losses.
              Otherwise, return predicted poses, scores, image
              paths and heatmaps.
        """

        if return_loss:
            return self.forward_train(img, targets, masks, joints, img_metas,
                                      **kwargs)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def forward_train(self, img, targets, masks, joints, img_metas, **kwargs):
        """Forward the bottom-up model and calculate the loss.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_weight: imgW
            img_height: imgH
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M

        Args:
            img(torch.Tensor[NxCximgHximgW]): Input image.
            targets(List(torch.Tensor[NxKxHxW])): Multi-scale target heatmaps.
            masks(List(torch.Tensor[NxHxW])): Masks of multi-scale target
                                              heatmaps
            joints(List(torch.Tensor[NxMxKx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas(dict):Information about val&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

        Returns:
            dict: The total loss for bottom-up
        """

        output = self.backbone(img)

        if self.with_keypoint:
            output = self.keypoint_head(output)

        # if return loss
        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, targets, masks, joints)
            losses.update(keypoint_losses)

        return losses

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            Tensor: Outputs.
        """
        output = self.backbone(img)
        if self.with_keypoint:
            output = self.keypoint_head(output)
        return output

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        """Inference the bottom-up model.

        Note:
            Batchsize = N (currently support batchsize = 1)
            num_img_channel: C
            img_weight: imgW
            img_height: imgH

        Args:
            flip_index (List(int)):
            aug_data (List(Tensor[NxCximgHximgW])): Multi-scale image
            test_scale_factor (List(float)): Multi-scale factor
            base_size (Tuple(int)): Base size of image when scale is 1
            center (np.ndarray): center of image
            scale (np.ndarray): the scale of image
        """
        assert img.size(0) == 1
        assert len(img_metas) == 1

        img_metas = img_metas[0]

        aug_data = img_metas['aug_data']

        test_scale_factor = img_metas['test_scale_factor']
        base_size = img_metas['base_size']
        center = img_metas['center']
        scale = img_metas['scale']

        result = {}

        aggregated_heatmaps = None
        tags_list = []
        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            image_resized = aug_data[idx].to(img.device)

            features = self.backbone(image_resized)
            if self.with_keypoint:
                outputs = self.keypoint_head(features)

            if self.test_cfg['flip_test']:
                # use flip test
                features_flipped = self.backbone(
                    torch.flip(image_resized, [3]))
                if self.with_keypoint:
                    outputs_flipped = self.keypoint_head(features_flipped)
            else:
                outputs_flipped = None

            _, heatmaps, tags = get_multi_stage_outputs(
                outputs,
                outputs_flipped,
                self.test_cfg['num_joints'],
                self.test_cfg['with_heatmaps'],
                self.test_cfg['with_ae'],
                self.test_cfg['tag_per_joint'],
                img_metas['flip_index'],
                self.test_cfg['project2image'],
                base_size,
                align_corners=self.use_udp)

            aggregated_heatmaps, tags_list = aggregate_results(
                s,
                aggregated_heatmaps,
                tags_list,
                heatmaps,
                tags,
                test_scale_factor,
                self.test_cfg['project2image'],
                self.test_cfg['flip_test'],
                align_corners=self.use_udp)

        # average heatmaps of different scales
        aggregated_heatmaps = aggregated_heatmaps / float(
            len(test_scale_factor))
        tags = torch.cat(tags_list, dim=4)

        # perform grouping
        grouped, scores = self.parser.parse(aggregated_heatmaps, tags,
                                            self.test_cfg['adjust'],
                                            self.test_cfg['refine'])

        preds = get_group_preds(
            grouped,
            center,
            scale, [aggregated_heatmaps.size(3),
                    aggregated_heatmaps.size(2)],
            use_udp=self.use_udp)

        image_paths = []
        image_paths.append(img_metas['image_file'])

        if return_heatmap:
            output_heatmap = aggregated_heatmaps.detach().cpu().numpy()
        else:
            output_heatmap = None

        result['preds'] = preds
        result['scores'] = scores
        result['image_paths'] = image_paths
        result['output_heatmap'] = output_heatmap

        return result

    def show_result(self,
                    img,
                    result,
                    skeleton=None,
                    kpt_score_thr=0.3,
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    radius=4,
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    show_keypoint_weight=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized image only if not `show` or `out_file`
        """

        img = mmcv.imread(img)
        img = img.copy()
        img_h, img_w, _ = img.shape

        pose_result = []
        for res in result:
            pose_result.append(res['keypoints'])

        for _, kpts in enumerate(pose_result):
            # draw each point on image
            if pose_kpt_color is not None:
                assert len(pose_kpt_color) == len(kpts)
                for kid, kpt in enumerate(kpts):
                    x_coord, y_coord, kpt_score = int(kpt[0]), int(
                        kpt[1]), kpt[2]
                    if kpt_score > kpt_score_thr:
                        if show_keypoint_weight:
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
                        else:
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)

            # draw limbs
            if skeleton is not None and pose_limb_color is not None:
                assert len(pose_limb_color) == len(skeleton)
                for sk_id, sk in enumerate(skeleton):
                    pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                    pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                    if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                            and pos1[1] < img_h and pos2[0] > 0
                            and pos2[0] < img_w and pos2[1] > 0
                            and pos2[1] < img_h
                            and kpts[sk[0] - 1, 2] > kpt_score_thr
                            and kpts[sk[1] - 1, 2] > kpt_score_thr):
                        r, g, b = pose_limb_color[sk_id]
                        if show_keypoint_weight:
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
                        else:
                            cv2.line(
                                img,
                                pos1,
                                pos2, (int(r), int(g), int(b)),
                                thickness=thickness)

        if show:
            imshow(img, win_name, wait_time)

        if out_file is not None:
            imwrite(img, out_file)

        return img
