# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_upsample_layer, constant_init, normal_init

from mmpose.core.camera import SimpleCamera
from mmpose.core.evaluation.top_down_eval import keypoints_from_joint_uvd
from mmpose.models.builder import build_loss
from ..builder import HEADS


class OffsetHead(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels_vector,
                 out_channels_scalar,
                 heatmap_kernel_size,
                 dummy_args=None):

        super().__init__()

        self.heatmap_kernel_size = heatmap_kernel_size
        assert out_channels_vector == out_channels_scalar * 3
        self.vector_offset = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels_vector,
            kernel_size=1,
            stride=1,
            padding=0)
        self.scalar_offset = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels_scalar,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, x):
        """Forward function."""
        vec = self.vector_offset(x)
        ht = self.scalar_offset(x)
        # N, C, H, W = x.shape
        offset_field = torch.cat((vec, ht), dim=1)
        return offset_field

    def init_weights(self):
        """Initialize model weights."""
        for m in self.vector_offset.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        for m in self.scalar_offset.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


class UpsampleHead(nn.Module):
    """UpsampleHead is a sub-module of AWR Head, and outputs 3D heatmaps.
    UpsampleHead is composed of (>=0) number of deconv layers.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        depth_size (int): Number of depth discretization size
        num_deconv_layers (int): Number of deconv layers.
        num_deconv_layers should >= 0. Note that 0 means no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
        num_deconv_kernels (list|tuple): Kernel sizes.
        extra (dict): Configs for extra conv layers. Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 depth_size=64,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None):

        super().__init__()

        assert out_channels % depth_size == 0
        self.depth_size = depth_size
        self.in_channels = in_channels

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0]
        identity_final_layer = True

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            # TODO: do not support this type of layer configuration
            raise NotImplementedError

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def forward(self, x):
        """Forward function."""
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        # N, C, H, W = x.shape
        # # reshape the 2D heatmap to 3D heatmap
        # x = x.reshape(N, C // self.depth_size, self.depth_size, H, W)
        return x

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)


@HEADS.register_module()
class AdaptiveWeightingRegression3DHead(nn.Module):
    """

    Args:
        deconv_head_cfg (dict): Configs of UpsampleHead for hand
            keypoint estimation.
        offset_head_cfg (dict): Configs of OffsetHead for hand
            keypoint offset field estimation.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
        loss_offset (dict): Config for offset field loss. Default: None.
    """

    def __init__(self,
                 deconv_head_cfg,
                 offset_head_cfg,
                 loss_keypoint=None,
                 loss_offset=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.deconv_head_cfg = deconv_head_cfg
        self.offset_head_cfg = offset_head_cfg

        # build sub-module heads
        # dense head
        self.offset_head = OffsetHead(**offset_head_cfg)
        # regression head
        self.upsample_feature_head = UpsampleHead(**deconv_head_cfg)

        # build losses
        self.keypoint_loss = build_loss(loss_keypoint)
        self.offset_loss = build_loss(loss_offset)
        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

    def init_weights(self):
        self.upsample_feature_head.init_weights()
        self.offset_head.init_weights()

    @staticmethod
    def offset2joint_softmax(offset, img, kernel_size):
        batch_size, feature_num, feature_size, _ = offset.size()
        jt_num = int(feature_num / 4)
        img = F.interpolate(
            img, size=[feature_size, feature_size])  # (B, 1, F, F)
        # unit directional vector
        offset_vec = offset[:, :jt_num * 3].contiguous()  # (B, jt_num*3, F, F)
        # closeness heatmap
        offset_ht = offset[:, jt_num * 3:].contiguous()  # (B, jt_num, F, F)

        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(
            feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(
            feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_x, mesh_y), dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1,
                                            1).to(offset.device)
        coords = torch.cat((coords, img),
                           dim=1).repeat(1, jt_num, 1,
                                         1)  # (B, jt_num*3, F, F)
        coords = coords.view(batch_size, jt_num, 3, -1)  # (B, jt_num, 3, F*F)

        mask = img.lt(0.99).float()  # (B, 1, F, F)
        offset_vec_mask = (offset_vec * mask).view(batch_size, jt_num, 3,
                                                   -1)  # (B, jt_num, 3, F*F)
        offset_ht_mask = (offset_ht * mask).view(batch_size, jt_num,
                                                 -1)  # (B, jt_num, F*F)
        offset_ht_norm = F.softmax(
            offset_ht_mask * 30, dim=-1)  # (B, jt_num, F*F)
        dis = kernel_size - offset_ht_mask * kernel_size  # (B, jt_num, F*F)

        jt_uvd = torch.sum(
            (offset_vec_mask * dis.unsqueeze(2) + coords) *
            offset_ht_norm.unsqueeze(2),
            dim=-1)

        return jt_uvd.float()

    @staticmethod
    def joint2offset(jt_uvd, img, kernel_size, feature_size):
        """
        :params joint: hand joint coordinates, shape (B, joint_num, 3)
        :params img: depth image, shape (B, C, H, W)
        :params kernel_size
        :params feature_size: size of generated offsets feature
        """
        batch_size, jt_num, _ = jt_uvd.size()
        img = F.interpolate(img, size=[feature_size, feature_size])
        jt_ft = jt_uvd.view(batch_size, -1, 1,
                            1).repeat(1, 1, feature_size,
                                      feature_size)  # (B, joint_num*3, F, F)

        mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(
            feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(
            feature_size, feature_size).float() + 0.5) / feature_size - 1.0
        coords = torch.stack((mesh_x, mesh_y), dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(
            jt_uvd.device)  # (B, 2, F, F)
        coords = torch.cat((coords, img),
                           dim=1).repeat(1, jt_num, 1,
                                         1)  # (B, jt_num*3, F, F)

        offset = jt_ft - coords  # (B, jt_num*3, F, F)
        offset = offset.view(batch_size, jt_num, 3, feature_size,
                             feature_size)  # (B, jt_num, 3, F, F)
        dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2) +
                         1e-8)  # (B, jt_num, F, F)

        offset_norm = offset / dis.unsqueeze(2)  # (B, jt_num, 3, F, F)
        heatmap = (kernel_size - dis) / kernel_size  # (B, jt_num, F, F)
        mask = heatmap.ge(0).float() * img.lt(
            0.99).float()  # (B, jt_num, F, F)

        offset_norm_mask = (offset_norm *
                            mask.unsqueeze(2)).view(batch_size, -1,
                                                    feature_size, feature_size)
        heatmap_mask = heatmap * mask.float()
        return torch.cat((offset_norm_mask, heatmap_mask), dim=1).float()

    def get_loss(self, output, target, target_weight):
        """Calculate loss for hand keypoint heatmaps, relative root depth and
        hand type.

        Args:
            output (list[Tensor]): a list of outputs from multiple heads.
            target (list[Tensor]): a list of targets for multiple heads.
            target_weight (list[Tensor]): a list of targets weight for
                multiple heads.
        """
        losses = dict()

        # hand keypoint offset field loss, dense loss
        assert not isinstance(self.keypoint_loss, nn.Sequential)
        out, tar, tar_weight = output[0], target[0], target_weight[0]
        assert tar.dim() == 4 and tar_weight.dim() in [1, 2]
        losses['offset_loss'] = self.offset_loss(out, tar)
        # hand keypoint joint loss, regression loss
        assert not isinstance(self.offset_loss, nn.Sequential)
        out, tar, tar_weight = output[1], target[1], target_weight[1]
        assert tar.dim() == 3 and tar_weight.dim() == 3
        losses['joint_loss'] = self.keypoint_loss(out, tar, tar_weight)

        return losses

    def forward(self, x):
        """Forward function."""
        backbone_feature, img = x
        feature = self.upsample_feature_head(backbone_feature)
        offset_field = self.offset_head(feature)
        jt_uvd = self.offset2joint_softmax(
            offset_field, img, self.offset_head_cfg['heatmap_kernel_size'])
        outputs = [offset_field, jt_uvd]
        return outputs

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output (list[np.ndarray]): list of output hand keypoint
            heatmaps, relative root depth and hand type.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """

        output = self.forward(x)

        if flip_pairs is not None:
            raise NotImplementedError
        else:
            output = [out.detach().cpu().numpy() for out in output]

        return output

    def decode(self, img_metas, output, **kwargs):
        """Decode hand keypoint and offset field.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:

                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
                - "heatmap3d_depth_bound": depth bound of hand keypoint
                    3D heatmap
                - "root_depth_bound": depth bound of relative root depth
                    1D heatmap
            output (list[np.ndarray]): model predicted 3D heatmaps, relative
                root depth and hand type.
        """

        batch_size = len(img_metas)
        result = {}

        center = np.zeros((batch_size, 2), dtype=np.float32)
        scale = np.zeros((batch_size, 2), dtype=np.float32)
        image_size = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size, dtype=np.float32)
        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        for i in range(batch_size):
            center[i, :] = img_metas[i]['center']
            scale[i, :] = img_metas[i]['scale']
            image_size[i, :] = img_metas[i]['image_size']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_boxes[:, 0:2] = center[:, 0:2]
        all_boxes[:, 2:4] = scale[:, 0:2]
        # scale is defined as: bbox_size / 200.0, so we
        # need multiply 200.0 to get bbox size
        all_boxes[:, 4] = np.prod(scale * 200.0, axis=1)
        all_boxes[:, 5] = score
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        # transform keypoint depth to camera space
        joint_uvd = output[1]
        preds, maxvals = keypoints_from_joint_uvd(joint_uvd, center, scale,
                                                  image_size)
        keypoints_3d = np.zeros((batch_size, joint_uvd.shape[1], 4),
                                dtype=np.float32)
        keypoints_3d[:, :, 0:3] = preds[:, :, 0:3]
        keypoints_3d[:, :, 3:4] = maxvals

        center_depth = np.array(
            [img_metas[i]['center_depth'] for i in range(len(img_metas))],
            dtype=np.float32)
        cube_size = np.array(
            [img_metas[i]['cube_size'] for i in range(len(img_metas))],
            dtype=np.float32)
        keypoints_3d[:, :, 2] = \
            keypoints_3d[:, :, 2] * cube_size[:, 2:] / 2 \
            + center_depth[:, np.newaxis]

        result['preds'] = keypoints_3d
        # joint uvd to joint xyz
        cam_param = {
            'R': np.eye(3, dtype=np.float32),
            'T': np.zeros((3, 1), dtype=np.float32),
            'f': img_metas[0]['focal'].reshape(2, 1),
            'c': img_metas[0]['princpt'].reshape(2, 1),
        }
        single_view_camera = SimpleCamera(param=cam_param)
        keypoints_xyz_list = []
        for batch_idx in range(batch_size):
            keypoints_xyz_list.append(
                single_view_camera.pixel_to_camera(
                    keypoints_3d[batch_idx, :, :3]))
        result['preds_xyz'] = np.stack(keypoints_xyz_list, 0)

        return result
