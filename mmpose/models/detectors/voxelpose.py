# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from mmpose.core.camera import SimpleCameraTorch
from mmpose.core.post_processing.post_transforms import (
    affine_transform_torch, get_affine_transform)
from .. import builder
from ..builder import POSENETS
from .base import BasePose


class CuboidProposalNet(nn.Module):
    """Cuboid proposal net in the paper "VoxelPose: Towards Multi-Camera 3D
    Human Pose Estimation in Wild Environment".

    Args:
        v2v_net (ConfigDict): dictionary to construct
            the 3D convolutional layers.
    """

    def __init__(self, v2v_net):
        super(CuboidProposalNet, self).__init__()
        self.v2v_net = builder.build_backbone(v2v_net)

    def forward(self, initial_cubes):
        return self.v2v_net(initial_cubes)


class PoseRegressionNet(nn.Module):
    """Pose regression net in the paper "VoxelPose: Towards Multi-Camera 3D
    Human Pose Estimation in Wild Environment".

    Args:
        v2v_net (ConfigDict): dictionary to construct
            the 3D convolutional layers.
    """

    def __init__(self, v2v_net):
        super(PoseRegressionNet, self).__init__()
        self.v2v_net = builder.build_backbone(v2v_net)

    def forward(self, cubes):
        return self.v2v_net(cubes)


class ProjectLayer(nn.Module):

    def __init__(self, cfg):
        """Project layer to get voxel feature. Adapted from
        https://github.com/microsoft/voxelpose-
        pytorch/blob/main/lib/models/project_layer.py.

        Args:
            cfg (dict):
                image_size: input size of the 2D model
                heatmap_size: output size of the 2D model
        """
        super(ProjectLayer, self).__init__()
        self.image_size = cfg['image_size']
        self.heatmap_size = cfg['heatmap_size']
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        if isinstance(self.heatmap_size, int):
            self.heatmap_size = [self.heatmap_size, self.heatmap_size]

    def compute_grid(self, box_size, box_center, num_bins, device=None):
        if isinstance(box_size, int) or isinstance(box_size, float):
            box_size = [box_size, box_size, box_size]
        if isinstance(num_bins, int):
            num_bins = [num_bins, num_bins, num_bins]

        grid_1D_x = torch.linspace(
            -box_size[0] / 2, box_size[0] / 2, num_bins[0], device=device)
        grid_1D_y = torch.linspace(
            -box_size[1] / 2, box_size[1] / 2, num_bins[1], device=device)
        grid_1D_z = torch.linspace(
            -box_size[2] / 2, box_size[2] / 2, num_bins[2], device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(
            grid_1D_x + box_center[0],
            grid_1D_y + box_center[1],
            grid_1D_z + box_center[2],
        )
        grid_x = grid_x.contiguous().view(-1, 1)
        grid_y = grid_y.contiguous().view(-1, 1)
        grid_z = grid_z.contiguous().view(-1, 1)
        grid = torch.cat([grid_x, grid_y, grid_z], dim=1)

        return grid

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size):
        device = heatmaps[0].device
        batch_size = heatmaps[0].shape[0]
        num_joints = heatmaps[0].shape[1]
        num_bins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(heatmaps)
        cubes = torch.zeros(
            batch_size, num_joints, 1, num_bins, n, device=device)
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, num_bins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, num_bins, n, device=device)
        for i in range(batch_size):
            if len(grid_center[0]) == 3 or grid_center[i][3] >= 0:
                if len(grid_center) == 1:
                    grid = self.compute_grid(
                        grid_size, grid_center[0], cube_size, device=device)
                else:
                    grid = self.compute_grid(
                        grid_size, grid_center[i], cube_size, device=device)
                grids[i:i + 1] = grid
                for c in range(n):
                    center = meta[i]['center'][c]
                    scale = meta[i]['scale'][c]

                    width, height = center * 2
                    trans = torch.as_tensor(
                        get_affine_transform(center, scale / 200.0, 0,
                                             self.image_size),
                        dtype=torch.float,
                        device=device)

                    cam_param = meta[i]['camera'][c].copy()

                    single_view_camera = SimpleCameraTorch(
                        param=cam_param, device=device)
                    xy = single_view_camera.world_to_pixel(grid)

                    bounding[i, 0, 0, :, c] = (xy[:, 0] >= 0) & (
                        xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                            xy[:, 1] < height)
                    xy = torch.clamp(xy, -1.0, max(width, height))
                    xy = affine_transform_torch(xy, trans)
                    xy = xy * torch.tensor(
                        [w, h], dtype=torch.float,
                        device=device) / torch.tensor(
                            self.image_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor([w - 1, h - 1],
                                                    dtype=torch.float,
                                                    device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(
                        sample_grid.view(1, 1, num_bins, 2), -1.1, 1.1)

                    cubes[i:i + 1, :, :, :, c] += F.grid_sample(
                        heatmaps[c][i:i + 1, :, :, :],
                        sample_grid,
                        align_corners=True)

        cubes = torch.sum(
            torch.mul(cubes, bounding), dim=-1) / (
                torch.sum(bounding, dim=-1) + 1e-6)
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)

        cubes = cubes.view(batch_size, num_joints, cube_size[0], cube_size[1],
                           cube_size[2])
        return cubes, grids

    def forward(self, heatmaps, meta, grid_size, grid_center, cube_size):
        cubes, grids = self.get_voxel(heatmaps, meta, grid_size, grid_center,
                                      cube_size)
        return cubes, grids


@POSENETS.register_module()
class VoxelPose(BasePose):
    """VoxelPose Please refer to the `paper <https://arxiv.org/abs/2004.06239>`
    for details.

    Args:
        detector_2d (ConfigDict): Dictionary to construct the 2D pose detector
        space_3d (ConfigDict): Dictionary that contains 3D space information
            space_size (list): Size of the 3D space
            cube_size (list): Size of the input volume to the  center net.
            space_center (list): Coordinate of the center of the 3D space
            sub_space_size (list): Size of the cuboid human proposal.
            sub_cube_size (list): Size of the input volume to the pose net.
        project_layer (ConfigDict): Dictionary to construct the project layer.
        center_net (ConfigDict): Dictionary to construct the center net.
        center_head (ConfigDict): Dictionary to construct the center head.
        pose_net (ConfigDict): Dictionary to construct the pose net.
        pose_head (ConfigDict): Dictionary to construct the pose head.
        train_cfg (ConfigDict): Config for training. Default: None.
        test_cfg (ConfigDict): Config for testing. Default: None.
        pretrained (str): Path to the pretrained 2D model. Default: None.
        freeze_2d (bool): Whether to freeze the 2D model in training.
            Default: True.
    """

    def __init__(self,
                 detector_2d,
                 space_3d,
                 project_layer,
                 center_net,
                 center_head,
                 pose_net,
                 pose_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 freeze_2d=True):
        super(VoxelPose, self).__init__()

        self.backbone = builder.build_posenet(detector_2d)
        if self.training and pretrained is not None:
            load_checkpoint(self.backbone, pretrained)

        self.freeze_2d = freeze_2d
        self.project_layer = ProjectLayer(project_layer)
        self.root_net = CuboidProposalNet(center_net)
        self.center_head = builder.build_head(center_head)
        self.pose_net = PoseRegressionNet(pose_net)
        self.pose_head = builder.build_head(pose_head)

        self.space_size = space_3d['space_size']
        self.cube_size = space_3d['cube_size']
        self.space_center = space_3d['space_center']

        self.sub_space_size = space_3d['sub_space_size']
        self.sub_cube_size = space_3d['sub_cube_size']

        self.num_joints = pose_net['output_channels']

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
            heatmaps width: W
            heatmaps height: H
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
                Multi-camera target heatmaps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera heatmaps when the 2D model is not available.
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

    def assign2gt(self, center_candidates, gt_centers, gt_num_persons):
        """"Assign gt id to each valid human center candidate."""
        det_centers = center_candidates[..., :3]
        batch_size = center_candidates.shape[0]
        cand_num = center_candidates.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = det_centers[i].view(cand_num, 1, -1)
            gt = gt_centers[None, i, :gt_num_persons[i]]

            dist = torch.sqrt(torch.sum((cand - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > self.train_cfg['dist_threshold']] = -1.0

        center_candidates[:, :, 3] = cand2gt

        return center_candidates

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
            heatmaps width: W
            heatmaps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            targets (list(torch.Tensor[NxKxHxW])):
                Multi-camera target heatmaps of the 2D model.
            masks (list(torch.Tensor[NxHxW])):
                Multi-camera masks of the input to the 2D model.
            targets_3d (torch.Tensor[NxcubeLxcubeWxcubeH]):
                Ground-truth 3D heatmap of human centers.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera heatmaps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: losses.

        """
        if self.backbone is None:
            assert input_heatmaps is not None
            heatmaps = []
            for input_heatmap in input_heatmaps:
                heatmaps.append(input_heatmap[0])
        else:
            heatmaps = []
            assert isinstance(img, list)
            for img_ in img:
                heatmaps.append(self.backbone.forward_dummy(img_)[0])
        initial_cubes, _ = self.project_layer(heatmaps, img_metas,
                                              self.space_size,
                                              [self.space_center],
                                              self.cube_size)
        center_heatmaps_3d = self.root_net(initial_cubes)
        num_joints = initial_cubes.shape[1]
        center_heatmaps_3d = center_heatmaps_3d.squeeze(1)
        center_candidates = self.center_head(center_heatmaps_3d)

        device = center_candidates.device

        gt_centers = torch.stack([
            torch.tensor(img_meta['roots_3d'], device=device)
            for img_meta in img_metas
        ])
        gt_num_persons = torch.stack([
            torch.tensor(img_meta['num_persons'], device=device)
            for img_meta in img_metas
        ])
        center_candidates = self.assign2gt(center_candidates, gt_centers,
                                           gt_num_persons)
        gt_3d = torch.stack([
            torch.tensor(img_meta['joints_3d'], device=device)
            for img_meta in img_metas
        ])
        gt_3d_vis = torch.stack([
            torch.tensor(img_meta['joints_3d_visible'], device=device)
            for img_meta in img_metas
        ])
        valid_preds = []
        valid_targets = []
        valid_weights = []

        batch_size, num_candidates, _ = center_candidates.shape
        pred = center_candidates.new_zeros(batch_size, num_candidates,
                                           self.num_joints, 5)
        pred[:, :, :, 3:] = center_candidates[:, :, None, 3:]

        for n in range(num_candidates):
            index = pred[:, n, 0, 3] >= 0
            num_valid = index.sum()
            if num_valid > 0:
                pose_input_cube, coordinates \
                    = self.project_layer(heatmaps,
                                         img_metas,
                                         self.sub_space_size,
                                         center_candidates[:, n, :3],
                                         self.sub_cube_size)
                pose_heatmaps_3d = self.pose_net(pose_input_cube)
                pose_3d = self.pose_head(pose_heatmaps_3d[index],
                                         coordinates[index])

                pred[index, n, :, 0:3] = pose_3d.detach()
                valid_targets.append(gt_3d[index, pred[index, n, 0, 3].long()])
                valid_weights.append(gt_3d_vis[index, pred[index, n, 0,
                                                           3].long(), :,
                                               0:1].float())
                valid_preds.append(pose_3d)

        losses = dict()
        losses.update(
            self.center_head.get_loss(center_heatmaps_3d, targets_3d))
        if len(valid_preds) > 0:
            valid_targets = torch.cat(valid_targets, dim=0)
            valid_weights = torch.cat(valid_weights, dim=0)
            valid_preds = torch.cat(valid_preds, dim=0)
            losses.update(
                self.pose_head.get_loss(valid_preds, valid_targets,
                                        valid_weights))
        else:
            pose_input_cube = initial_cubes.new_zeros(batch_size, num_joints,
                                                      *self.sub_cube_size)
            coordinates = initial_cubes.new_zeros(batch_size,
                                                  *self.sub_cube_size,
                                                  3).view(batch_size, -1, 3)
            pseudo_targets = initial_cubes.new_zeros(batch_size, num_joints, 3)
            pseudo_weights = initial_cubes.new_zeros(batch_size, num_joints, 1)
            pose_heatmaps_3d = self.pose_net(pose_input_cube)
            pose_3d = self.pose_head(pose_heatmaps_3d, coordinates)
            losses.update(
                self.pose_head.get_loss(pose_3d, pseudo_targets,
                                        pseudo_weights))

        if not self.freeze_2d:
            losses_2d = {}
            heatmaps_tensor = torch.cat(heatmaps, dim=0)
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
            heatmaps width: W
            heatmaps height: H
            volume_length: cubeL
            volume_width: cubeW
            volume_height: cubeH

        Args:
            img (list(torch.Tensor[NxCximgHximgW])):
                Multi-camera input images to the 2D model.
            img_metas (list(dict)):
                Information about image, 3D groundtruth and camera parameters.
            input_heatmaps (list(torch.Tensor[NxKxHxW])):
                Multi-camera heatmaps when the 2D model is not available.
                 Default: None.

        Returns:
            dict: predicted poses, human centers and sample_id

        """
        if self.backbone is None:
            assert input_heatmaps is not None
            heatmaps = []
            for input_heatmap in input_heatmaps:
                heatmaps.append(input_heatmap[0])
        else:
            heatmaps = []
            assert isinstance(img, list)
            for img_ in img:
                heatmaps.append(self.backbone.forward_dummy(img_)[0])
        initial_cubes, _ = self.project_layer(heatmaps, img_metas,
                                              self.space_size,
                                              [self.space_center],
                                              self.cube_size)
        center_heatmaps_3d = self.root_net(initial_cubes)
        center_heatmaps_3d = center_heatmaps_3d.squeeze(1)
        center_candidates = self.center_head(center_heatmaps_3d)

        center_candidates[..., 3] = \
            (center_candidates[..., 4] >
             self.test_cfg['center_threshold']).float() - 1.0

        batch_size, num_candidates, _ = center_candidates.shape
        pred = center_candidates.new_zeros(batch_size, num_candidates,
                                           self.num_joints, 5)
        pred[:, :, :, 3:] = center_candidates[:, :, None, 3:]

        for n in range(num_candidates):
            index = pred[:, n, 0, 3] >= 0
            num_valid = index.sum()
            if num_valid > 0:
                pose_input_cube, coordinates \
                    = self.project_layer(heatmaps,
                                         img_metas,
                                         self.sub_space_size,
                                         center_candidates[:, n, :3],
                                         self.sub_cube_size)
                pose_heatmaps_3d = self.pose_net(pose_input_cube)
                pose_3d = self.pose_head(pose_heatmaps_3d[index],
                                         coordinates[index])

                pred[index, n, :, 0:3] = pose_3d.detach()

        result = {}
        result['pose_3d'] = pred.cpu().numpy()
        result['center_3d'] = center_candidates.cpu().numpy()
        result['sample_id'] = [img_meta['sample_id'] for img_meta in img_metas]

        return result

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError

    def forward_dummy(self, img, input_heatmaps=None, num_candidates=5):
        """Used for computing network FLOPs."""
        if self.backbone is None:
            assert input_heatmaps is not None
            heatmaps = []
            for input_heatmap in input_heatmaps:
                heatmaps.append(input_heatmap[0])
        else:
            heatmaps = []
            assert isinstance(img, list)
            for img_ in img:
                heatmaps.append(self.backbone.forward_dummy(img_)[0])
        batch_size, num_channels, _, _ = heatmaps[0].shape
        initial_cubes = heatmaps[0].new_zeros(batch_size, num_channels,
                                              *self.cube_size)
        _ = self.root_net(initial_cubes)
        pose_input_cube = heatmaps[0].new_zeros(batch_size, num_channels,
                                                *self.sub_cube_size)
        for n in range(num_candidates):
            _ = self.pose_net(pose_input_cube)
