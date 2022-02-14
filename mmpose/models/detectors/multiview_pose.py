# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import load_checkpoint

from .. import builder
from ..builder import POSENETS
from .base import BasePose


@POSENETS.register_module()
class DetectAndRegress(BasePose):
    """VoxelPose Please refer to the `paper <https://arxiv.org/abs/2004.06239>`
    for details.

    Args:
        detector_2d (ConfigDict): Dictionary to construct the 2D pose detector
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained 2D model. Default: None.
        freeze_2d (bool): Whether to freeze the 2D model in training.
            Default: True.
    """

    def __init__(self,
                 detector_2d,
                 human_detector,
                 pose_regressor,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 freeze_2d=True):
        super(DetectAndRegress, self).__init__()

        self.backbone = builder.build_posenet(detector_2d)
        if self.training and pretrained is not None:
            load_checkpoint(self.backbone, pretrained)

        self.freeze_2d = freeze_2d
        self.human_detector = builder.MODELS.build(human_detector)
        self.pose_regressor = builder.MODELS.build(pose_regressor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @staticmethod
    def _freeze(model):
        """Freeze parameters."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Sets the module in training mode.
        Args:
            mode (bool): whether to set training mode (``True``)
                or evaluation mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode and self.freeze_2d:
            self._freeze(self.backbone)

        return self

    def forward(self,
                img,
                img_metas,
                return_loss=True,
                targets=None,
                masks=None,
                targets_3d=None,
                input_heatmaps=None,
                **kwargs):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            return_loss: Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            targets (list(torch.Tensor[NxKxHxW])):
                Multi-camera target feature_maps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.
            **kwargs:

        Returns:
            dict: if 'return_loss' is true, then return losses.
              Otherwise, return predicted poses, human centers and sample_id

        """
        if return_loss:
            return self.forward_train(img, img_metas, targets, masks,
                                      targets_3d, input_heatmaps)
        else:
            return self.forward_test(img, img_metas, input_heatmaps)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self.forward(**data_batch)

        loss, log_vars = self._parse_losses(losses)
        batch_size = data_batch['img'][0].shape[0]
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=batch_size)

        return outputs

    def forward_train(self,
                      img,
                      img_metas,
                      targets=None,
                      masks=None,
                      targets_3d=None,
                      input_heatmaps=None):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            targets (list(torch.Tensor[NxKxHxW])):
                Multi-camera target feature_maps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: losses.

        """
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.backbone.forward_dummy(img_)[0])

        losses = dict()
        human_candidates, human_loss = self.human_detector(feature_maps, img_metas,
                                                           targets, return_loss=True)
        losses.update(human_loss)

        _, pose_loss = self.pose_regressor(feature_maps, img_metas, human_candidates,
                                           return_loss=True)
        losses.update(pose_loss)

        if not self.freeze_2d:
            losses_2d = {}
            heatmaps_tensor = torch.cat(feature_maps, dim=0)
            targets_tensor = torch.cat(targets, dim=0)
            masks_tensor = torch.cat(masks, dim=0)
            losses_2d_ = self.backbone.get_loss(heatmaps_tensor,
                                                targets_tensor, masks_tensor)
            for k, v in losses_2d_.items():
                losses_2d[k + '_2d'] = v
            losses.update(losses_2d)

        return losses

    def forward_test(
        self,
        img,
        img_metas,
        input_heatmaps=None,
    ):
        """
        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C
            img_width: imgW
            img_height: imgH
            feature_maps width: W
            feature_maps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera feature_maps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: predicted poses, human centers and sample_id

        """
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.backbone.forward_dummy(img_)[0])

        human_candidates = self.human_detector(feature_maps, img_metas,
                                               return_loss=False)

        human_poses = self.pose_regressor(feature_maps, img_metas, human_candidates,
                                          return_loss=False)

        result = {}
        result['human_pose_3d'] = human_poses.cpu().numpy()
        result['human_detection_3d'] = human_candidates.cpu().numpy()
        result['sample_id'] = [img_meta['sample_id'] for img_meta in img_metas]

        return result

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError

    def forward_dummy(self, img, input_heatmaps=None, num_candidates=5):
        """Used for computing network FLOPs."""
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.backbone.forward_dummy(img_)[0])

        _ = self.human_detector.forward_dummy(feature_maps)

        _ = self.pose_regressor.forward_dummy(feature_maps,
                                              num_candidates)
