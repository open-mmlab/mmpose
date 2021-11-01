# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS


def get_index(indices, shape):
    batch_size = indices.shape[0]
    num_people = indices.shape[1]
    indices_x = (indices //
                 (shape[1] * shape[2])).reshape(batch_size, num_people, -1)
    indices_y = ((indices % (shape[1] * shape[2])) //
                 shape[2]).reshape(batch_size, num_people, -1)
    indices_z = (indices % shape[2]).reshape(batch_size, num_people, -1)
    indices = torch.cat([indices_x, indices_y, indices_z], dim=2)
    return indices


def max_pool(inputs, kernel=3):
    padding = (kernel - 1) // 2
    max = F.max_pool3d(inputs, kernel_size=kernel, stride=1, padding=padding)
    keep = (inputs == max).float()
    return keep * inputs


def nms(root_cubes, max_num):
    batch_size = root_cubes.shape[0]
    root_cubes_nms = max_pool(root_cubes)
    root_cubes_nms_reshape = root_cubes_nms.reshape(batch_size, -1)
    topk_values, topk_index = root_cubes_nms_reshape.topk(max_num)
    topk_unravel_index = get_index(topk_index, root_cubes[0].shape)

    return topk_values, topk_unravel_index


@HEADS.register_module()
class CuboidCenterHead(nn.Module):

    def __init__(self, cfg):
        super(CuboidCenterHead, self).__init__()
        self.grid_size = torch.tensor(cfg['space_size'])
        self.cube_size = torch.tensor(cfg['cube_size'])
        self.grid_center = torch.tensor(cfg['space_center'])
        self.num_cand = cfg['max_num']
        self.loss = nn.MSELoss()
        # self.threshold = cfg['threshold']

    def get_real_loc(self, index):
        device = index.device
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = index.float() / (cube_size -
                               1) * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, root_cubes):
        batch_size = root_cubes.shape[0]

        topk_values, topk_unravel_index = nms(root_cubes.detach(),
                                              self.num_cand)
        topk_unravel_index = self.get_real_loc(topk_unravel_index)

        grid_centers = torch.zeros(
            batch_size, self.num_cand, 5, device=root_cubes.device)
        grid_centers[:, :, 0:3] = topk_unravel_index
        grid_centers[:, :, 4] = topk_values
        # grid_centers[:, :, 3] = (topk_values > self.threshold).float() - 1.0

        return grid_centers

    def get_loss(self, pred_cubes, gt):

        return dict(loss_center=self.loss(pred_cubes, gt))


@HEADS.register_module()
class CuboidPoseHead(nn.Module):

    def __init__(self, beta):
        super(CuboidPoseHead, self).__init__()
        self.beta = beta
        self.loss = nn.L1Loss()

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x

    def get_loss(self, preds, targets, weights):

        return dict(loss_pose=self.loss(preds * weights, targets * weights))
