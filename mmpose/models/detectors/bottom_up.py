import torch

from mmpose.core.evaluation import (aggregate_results, get_group_preds,
                                    get_multi_stage_outputs)
from mmpose.core.post_processing.group import HeatmapParser
from mmpose.models.builder import build_loss
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

        self.parser = HeatmapParser(self.test_cfg)

        self.loss = build_loss(loss_pose)
        self.init_weights(pretrained=pretrained)

    @property
    def with_keypoint(self):
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):

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
            joints(List(torch.Tensor[NxMxkx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            return loss(bool): Option to 'return_loss'. 'return_loss=True' for
                training, 'return_loss=False' for validation & test
            img_metas(dict):Information about valid&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints
        Returns:
            if 'return_loss' is true, then return losses. Otherwise, return
                predicted poses, scores and image paths.
        """

        if return_loss:
            return self.forward_train(img, targets, masks, joints, img_metas,
                                      **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

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
            joints(List(torch.Tensor[NxMxkx2])): Joints of multi-scale target
                                                 heatmaps for ae loss
            img_metas(dict):Information about valid&test
                By default this includes:
                - "image_file": image path
                - "aug_data": input
                - "test_scale_factor": test scale factor
                - "base_size": base size of input
                - "center": center of image
                - "scale": scale of image
                - "flip_index": flip index of keypoints

        Returns:
            losses (dict): the total loss for bottom-up
        """

        output = self.backbone(img)

        if self.with_keypoint:
            output = self.keypoint_head(output)

        heatmaps_losses, push_losses, pull_losses = self.loss(
            output, targets, masks, joints)

        losses = dict()

        loss = 0
        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                loss = loss + heatmaps_loss
                if push_losses[idx] is not None:
                    push_loss = push_losses[idx].mean(dim=0)
                    loss = loss + push_loss
                if pull_losses[idx] is not None:
                    pull_loss = pull_losses[idx].mean(dim=0)
                    loss = loss + pull_loss

        losses['all_loss'] = loss
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """Inference the bottom-up model.

        Note:
            Batchsize = N (currently support batchsize = 1)
            num_img_channel: C
            img_weight: imgW
            img_height: imgH

        Args:
            flip_index (List(int)):
            aug_data (List(Tensor[NxCximgHximgW])): Multi-scale image
            test_scale_fator (List(float)): Multi-scale fator
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

        aggregated_heatmaps = None
        tags_list = []
        for idx, s in enumerate(sorted(test_scale_factor, reverse=True)):
            image_resized = aug_data[idx].to(img.device)

            outputs = self.backbone(image_resized)
            outputs = self.keypoint_head(outputs)

            if self.test_cfg['flip_test']:
                # use flip test
                outputs_flip = self.backbone(torch.flip(image_resized, [3]))
                outputs_flip = self.keypoint_head(outputs_flip)
            else:
                outputs_flip = None

            _, heatmaps, tags = get_multi_stage_outputs(
                outputs, outputs_flip, self.test_cfg['num_joints'],
                self.test_cfg['with_heatmaps'], self.test_cfg['with_ae'],
                self.test_cfg['tag_per_joint'], img_metas['flip_index'],
                self.test_cfg['project2image'], base_size)

            aggregated_heatmaps, tags_list = aggregate_results(
                s, aggregated_heatmaps, tags_list, heatmaps, tags,
                test_scale_factor, self.test_cfg['project2image'],
                self.test_cfg['flip_test'])

        # average heatmaps of different scales
        aggregated_heatmaps = aggregated_heatmaps / float(
            len(test_scale_factor))
        tags = torch.cat(tags_list, dim=4)

        # perform grouping
        grouped, scores = self.parser.parse(aggregated_heatmaps, tags,
                                            self.test_cfg['adjust'],
                                            self.test_cfg['refine'])

        results = get_group_preds(
            grouped, center, scale,
            [aggregated_heatmaps.size(3),
             aggregated_heatmaps.size(2)])

        image_path = []
        image_path.extend(img_metas['image_file'])

        return results, scores, image_path
