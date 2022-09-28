from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import numpy as np
import torch
import torch.nn as nn

class PredictOKSNet(nn.Module):
    def __init__(self, cfg, input_channel, **kwargs):
        super(PredictOKSNet, self).__init__()
        hidden = cfg.RESCORE.HIDDEN_LAYER
        self.l1 = torch.nn.Linear(input_channel, hidden, bias=True)
        self.l2 = torch.nn.Linear(hidden, hidden, bias=True)
        self.l3 = torch.nn.Linear(hidden, 1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        y_pred = self.l3(x2)
        return y_pred

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)


def get_pose_net(cfg, input_channel, is_train, **kwargs):
    model = PredictOKSNet(cfg, input_channel, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights()

    return model