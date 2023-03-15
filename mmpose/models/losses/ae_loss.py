# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS


@MODELS.register_module()
class AssociativeEmbeddingLoss(nn.Module):
    """Associative Embedding loss.

    Details can be found in
    `Associative Embedding <https://arxiv.org/abs/1611.05424>`_

    Note:

        - batch size: B
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - embedding tag dimension: L
        - heatmap size: [W, H]

    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0
        push_loss_factor (float): A factor that controls the weight between
            the push loss and the pull loss. Defaults to 0.5
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 push_loss_factor: float = 0.5) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.push_loss_factor = push_loss_factor

    def _ae_loss_per_image(self, tags: Tensor, keypoint_indices: Tensor):
        """Compute associative embedding loss for one image.

        Args:
            tags (Tensor): Tagging heatmaps in shape (K*L, H, W)
            keypoint_indices (Tensor): Ground-truth keypint position indices
                in shape (N, K, 2)
        """
        K = keypoint_indices.shape[1]
        C, H, W = tags.shape
        L = C // K

        tags = tags.view(L, K, H * W)
        instance_tags = []
        instance_kpt_tags = []

        for keypoint_indices_n in keypoint_indices:
            _kpt_tags = []
            for k in range(K):
                if keypoint_indices_n[k, 1]:
                    _kpt_tags.append(tags[:, k, keypoint_indices_n[k, 0]])

            if _kpt_tags:
                kpt_tags = torch.stack(_kpt_tags)
                instance_kpt_tags.append(kpt_tags)
                instance_tags.append(kpt_tags.mean(dim=0))

        N = len(instance_kpt_tags)  # number of instances with valid keypoints

        if N == 0:
            pull_loss = tags.new_zeros(size=(), requires_grad=True)
            push_loss = tags.new_zeros(size=(), requires_grad=True)
        else:
            pull_loss = sum(
                F.mse_loss(_kpt_tags, _tag.expand_as(_kpt_tags))
                for (_kpt_tags, _tag) in zip(instance_kpt_tags, instance_tags))

            if N == 1:
                push_loss = tags.new_zeros(size=(), requires_grad=True)
            else:
                tag_mat = torch.stack(instance_tags)  # (N, L)
                diff = tag_mat[None] - tag_mat[:, None]  # (N, N, L)
                push_loss = torch.sum(torch.exp(-diff.pow(2)))

            # normalization
            eps = 1e-6
            pull_loss = pull_loss / (N + eps)
            push_loss = push_loss / ((N - 1) * N + eps)

        return pull_loss, push_loss

    def forward(self, tags: Tensor, keypoint_indices: Union[List[Tensor],
                                                            Tensor]):
        """Compute associative embedding loss on a batch of data.

        Args:
            tags (Tensor): Tagging heatmaps in shape (B, L*K, H, W)
            keypoint_indices (Tensor|List[Tensor]): Ground-truth keypint
                position indices represented by a Tensor in shape
                (B, N, K, 2), or a list of B Tensors in shape (N_i, K, 2)
                Each keypoint's index is represented as [i, v], where i is the
                position index in the heatmap (:math:`i=y*w+x`) and v is the
                visibility

        Returns:
            tuple:
            - pull_loss (Tensor)
            - push_loss (Tensor)
        """

        assert tags.shape[0] == len(keypoint_indices)

        pull_loss = 0.
        push_loss = 0.

        for i in range(tags.shape[0]):
            _pull, _push = self._ae_loss_per_image(tags[i],
                                                   keypoint_indices[i])
            pull_loss += _pull * self.loss_weight
            push_loss += _push * self.loss_weight * self.push_loss_factor

        return pull_loss, push_loss
