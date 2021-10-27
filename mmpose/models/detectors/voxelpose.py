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

    def __init__(self, v2v_net):
        super(CuboidProposalNet, self).__init__()
        self.v2v_net = builder.build_backbone(v2v_net)

    def forward(self, initial_cubes):
        return self.v2v_net(initial_cubes)


class PoseRegressionNet(nn.Module):

    def __init__(self, v2v_net):
        super(PoseRegressionNet, self).__init__()
        self.v2v_net = builder.build_backbone(v2v_net)

    def forward(self, cubes):
        return self.v2v_net(cubes)


class ProjectLayer(nn.Module):

    def __init__(self, cfg):
        super(ProjectLayer, self).__init__()
        self.img_size = cfg['image_size']
        self.heatmap_size = cfg['heatmap_size']
        if isinstance(self.img_size, int):
            self.img_size = [self.img_size, self.img_size]
        if isinstance(self.heatmap_size, int):
            self.heatmap_size = [self.heatmap_size, self.heatmap_size]

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(
            -boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(
            -boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(
            -boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid

    def get_voxel(self, heatmaps, meta, grid_size, grid_center, cube_size):
        device = heatmaps[0].device
        batch_size = heatmaps[0].shape[0]
        num_joints = heatmaps[0].shape[1]
        nbins = cube_size[0] * cube_size[1] * cube_size[2]
        n = len(heatmaps)
        cubes = torch.zeros(batch_size, num_joints, 1, nbins, n, device=device)
        # h, w = heatmaps[0].shape[2], heatmaps[0].shape[3]
        w, h = self.heatmap_size
        grids = torch.zeros(batch_size, nbins, 3, device=device)
        bounding = torch.zeros(batch_size, 1, 1, nbins, n, device=device)
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
                                             self.img_size),
                        dtype=torch.float,
                        device=device)

                    # for k, v in meta[i]['camera'][c].items():
                    #     cam_param[k] = v
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
                            self.img_size, dtype=torch.float, device=device)
                    sample_grid = xy / torch.tensor([w - 1, h - 1],
                                                    dtype=torch.float,
                                                    device=device) * 2.0 - 1.0
                    sample_grid = torch.clamp(
                        sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)

                    # if pytorch version < 1.3.0,
                    # align_corners=True should be omitted.
                    cubes[i:i + 1, :, :, :, c] += F.grid_sample(
                        heatmaps[c][i:i + 1, :, :, :],
                        sample_grid,
                        align_corners=True)

        # cubes = cubes.mean(dim=-1)
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
        if self.training and freeze_2d:
            self._freeze(self.backbone)

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

        Args:
            img: list(torch.Tensor[NxCximgHximgW])
            target: list(torch.Tensor[NxKxHxW])
            targets_3d:
            img_metas:
            return_loss:
            input_heatmaps:
            **kwargs:

        Returns:

        """
        if self.backbone is None:
            assert input_heatmaps is not None
            heatmaps = input_heatmaps
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

        device = center_candidates.device
        if return_loss:
            gt_centers = torch.stack([
                torch.tensor(img_meta['roots_3d'], device=device)
                for img_meta in img_metas
            ])
            # gt_centers = img_metas['roots_3d'].float()
            gt_num_persons = torch.stack([
                torch.tensor(img_meta['num_persons'], device=device)
                for img_meta in img_metas
            ])
            # gt_num_persons = img_metas['num_persons']
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
            # gt_3d = img_metas['joints_3d'].float()
            # gt_3d_vis = img_metas['joints_3d_visible']
            valid_preds = []
            valid_targets = []
            valid_weights = []

        else:
            center_candidates[..., 3] = \
                (center_candidates[..., 4] >
                 self.test_cfg['center_threshold']).float() - 1.0

        batch_size, num_candidates, _ = center_candidates.shape
        pred = center_candidates.new_zeros(batch_size, num_candidates,
                                           self.num_joints, 5)
        pred[:, :, :, 3:] = center_candidates[:, :, None, 3:]  # matched gt

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

                if return_loss:
                    valid_targets.append(gt_3d[index, pred[index, n, 0,
                                                           3].long()])
                    valid_weights.append(gt_3d_vis[index, pred[index, n, 0,
                                                               3].long(), :,
                                                   0:1].float())
                    valid_preds.append(pose_3d)

        if return_loss:
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

            if not self.freeze_2d:
                losses_2d = {}
                heatmaps_tensor = torch.cat(heatmaps, dim=0)
                targets_tensor = torch.cat(targets, dim=0)
                masks_tensor = torch.cat(masks, dim=0)
                losses_2d_ = self.backbone.get_loss(heatmaps_tensor,
                                                    targets_tensor,
                                                    masks_tensor)
                for k, v in losses_2d_.items():
                    losses_2d[k + '_2d'] = v
                losses.update(losses_2d)

            return losses

        result = {}
        result['pose_3d'] = pred.cpu().numpy()
        result['center_3d'] = center_candidates.cpu().numpy()
        result['sample_id'] = [img_meta['sample_id'] for img_meta in img_metas]

        return result

    def assign2gt(self, center_candidates, gt_centers, gt_num_persons):
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

    def forward_train(self, img, img_metas, **kwargs):
        """Defines the computation performed at training."""
        raise NotImplementedError

    def forward_test(self, img, img_metas, **kwargs):
        """Defines the computation performed at testing."""
        raise NotImplementedError

    def show_result(self, **kwargs):
        """Visualize the results."""
        raise NotImplementedError
