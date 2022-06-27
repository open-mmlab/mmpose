# Copyright (c) OpenMMLab. All rights reserved.

import torch


class CoordNormLinear(torch.nn.Linear):

    def __init__(self, num_joints: int, coord_dim: int = 2, *args, **kwargs):
        super(CoordNormLinear, self).__init__(*args, **kwargs)
        self.num_joints = num_joints
        self.coord_dim = coord_dim

    def forward(self, x):
        y = x.matmul(self.weight.t())
        y = y.view(x.size(0), self.num_joints, -1)

        x_norm = torch.norm(x, dim=1, keepdim=True).unsqueeze(-1)
        y = torch.cat(
            (y[..., :self.coord_dim] / x_norm, y[..., self.coord_dim:]),
            dim=-1)
        y = y.view(x.size(0), -1)

        if self.bias is not None:
            y = y + self.bias

        return y
