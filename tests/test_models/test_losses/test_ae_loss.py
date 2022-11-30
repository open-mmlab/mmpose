# Copyright (c) OpenMMLab. All rights reserved.
from itertools import product
from typing import Tuple
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmpose.codecs.associative_embedding import AssociativeEmbedding
from mmpose.models.losses.ae_loss import AssociativeEmbeddingLoss
from mmpose.testing._utils import get_coco_sample


class AELoss(nn.Module):
    """Associative Embedding loss in MMPose v0.x."""

    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    @staticmethod
    def _make_input(t, requires_grad=False, device=torch.device('cpu')):
        """Make zero inputs for AE loss.

        Args:
            t (torch.Tensor): input
            requires_grad (bool): Option to use requires_grad.
            device: torch device

        Returns:
            torch.Tensor: zero input.
        """
        inp = torch.autograd.Variable(t, requires_grad=requires_grad)
        inp = inp.sum()
        inp = inp.to(device)
        return inp

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            - heatmaps weight: W
            - heatmaps height: H
            - max_num_people: M
            - num_keypoints: K

        Args:
            pred_tag (torch.Tensor[KxHxW,1]): tag of output for one image.
            joints (torch.Tensor[M,K,2]): joints information for one image.
        """
        tags = []
        pull = 0
        pred_tag = pred_tag.view(17, -1, 1)
        for joints_per_person in joints:
            tmp = []
            for k, joint in enumerate(joints_per_person):
                if joint[1] > 0:
                    tmp.append(pred_tag[k, joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return (self._make_input(
                torch.zeros(1).float(), device=pred_tag.device),
                    self._make_input(
                        torch.zeros(1).float(), device=pred_tag.device))
        elif num_tags == 1:
            return (self._make_input(
                torch.zeros(1).float(), device=pred_tag.device), pull)

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == 'exp':
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push)
        elif self.loss_type == 'max':
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError('Unknown ae loss type')

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, keypoint_indices):
        assert tags.shape[0] == len(keypoint_indices)

        pull_loss = 0.
        push_loss = 0.

        for i in range(tags.shape[0]):
            _push, _pull = self.singleTagLoss(tags[i].view(-1, 1),
                                              keypoint_indices[i])
            pull_loss += _pull
            push_loss += _push

        return pull_loss, push_loss


class TestAssociativeEmbeddingLoss(TestCase):

    def _make_input(self, num_instance: int) -> Tuple[Tensor, Tensor]:

        encoder = AssociativeEmbedding(
            input_size=(256, 256), heatmap_size=(64, 64))

        data = get_coco_sample(
            img_shape=(256, 256), num_instances=num_instance)
        encoded = encoder.encode(data['keypoints'], data['keypoints_visible'])
        heatmaps = encoded['heatmaps']
        keypoint_indices = encoded['keypoint_indices']

        tags = self._get_tags(
            heatmaps, keypoint_indices, tag_per_keypoint=True)

        batch_tags = torch.from_numpy(tags[None])
        batch_keypoint_indices = [torch.from_numpy(keypoint_indices)]

        return batch_tags, batch_keypoint_indices

    def _get_tags(self,
                  heatmaps,
                  keypoint_indices,
                  tag_per_keypoint: bool,
                  with_randomness: bool = True):

        K, H, W = heatmaps.shape
        N = keypoint_indices.shape[0]

        if tag_per_keypoint:
            tags = np.zeros((K, H, W), dtype=np.float32)
        else:
            tags = np.zeros((1, H, W), dtype=np.float32)

        for n, k in product(range(N), range(K)):
            y, x = np.unravel_index(keypoint_indices[n, k, 0], (H, W))

            randomness = np.random.rand() if with_randomness else 0

            if tag_per_keypoint:
                tags[k, y, x] = n + randomness
            else:
                tags[0, y, x] = n + randomness

        return tags

    def test_loss(self):

        tags, keypoint_indices = self._make_input(num_instance=2)

        # test loss calculation
        loss_module = AssociativeEmbeddingLoss()
        pull_loss, push_loss = loss_module(tags, keypoint_indices)
        _pull_loss, _push_loss = AELoss('exp')(tags, keypoint_indices)

        self.assertTrue(torch.allclose(pull_loss, _pull_loss))
        self.assertTrue(torch.allclose(push_loss, _push_loss))

        # test loss weight
        loss_module = AssociativeEmbeddingLoss(loss_weight=0.)
        pull_loss, push_loss = loss_module(tags, keypoint_indices)

        self.assertTrue(torch.allclose(pull_loss, torch.zeros(1)))
        self.assertTrue(torch.allclose(push_loss, torch.zeros(1)))

        # test push loss factor
        loss_module = AssociativeEmbeddingLoss(push_loss_factor=0.)
        pull_loss, push_loss = loss_module(tags, keypoint_indices)

        self.assertFalse(torch.allclose(pull_loss, torch.zeros(1)))
        self.assertTrue(torch.allclose(push_loss, torch.zeros(1)))
