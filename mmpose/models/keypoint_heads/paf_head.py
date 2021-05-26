import torch.nn as nn

from mmpose.models.builder import build_head
from ..registry import HEADS


@HEADS.register_module()
class PAFHead(nn.Module):
    """Bottom-up PAF head.

    Paper ref: Cao, Zhe, et al. "OpenPose: realtime multi-person
    2D pose estimation using Part Affinity Fields." (TPAMI'2019)

    Args:
        heatmap_heads_cfg (list(dict)): Configs of heatmap heads.
        paf_heads_cfg (list(dict)): Configs of paf heads.
        heatmap_index (list(int)): The correspondence between heatmap heads
            and input features.
        paf_index (list(int)): The correspondence between paf heads
            and input features.
    """

    def __init__(self, heatmap_heads_cfg, paf_heads_cfg, heatmap_index,
                 paf_index):
        super().__init__()

        assert len(heatmap_heads_cfg) == len(heatmap_index)
        assert len(paf_heads_cfg) == len(paf_index)

        # build heatmap heads
        self.heatmap_heads_list = []
        for head_cfg in heatmap_heads_cfg:
            self.heatmap_heads_list.append(build_head(head_cfg))

        # build paf heads
        self.paf_heads_list = []
        for head_cfg in paf_heads_cfg:
            self.paf_heads_list.append(build_head(head_cfg))

        self.heatmap_index = heatmap_index
        self.paf_index = paf_index

    def get_loss(self, outputs, targets, masks):
        """Calculate heatmap and paf loss.

        Note:
            batch_size: N
            num_channels: C
            heatmaps height: H
            heatmaps weight: W

        Args:
            outputs (dict): Outputs of network, including heatmaps and pafs.
            targets (list(list)): List of heatmaps and pafs, each of which
                multi-scale targets.
            masks (list(torch.Tensor[NxHxW])): Masks of multi-scale target
                heatmaps and pafs.
        """

        losses = dict()

        heatmap_outputs = outputs['heatmaps']
        heatmap_targets = targets[:len(self.heatmap_heads_list)]
        for idx, head in enumerate(self.heatmap_heads_list):
            heatmap_losses = head.get_loss(heatmap_outputs[idx],
                                           heatmap_targets[idx], masks)
            if 'heatmap_loss' not in losses:
                losses['heatmap_loss'] = heatmap_losses['loss']
            else:
                losses['heatmap_loss'] += heatmap_losses['loss']

        paf_outputs = outputs['pafs']
        paf_targets = targets[len(self.heatmap_heads_list):]
        for idx, head in enumerate(self.paf_heads_list):
            paf_losses = head.get_loss(paf_outputs[idx], paf_targets[idx],
                                       masks)
            if 'paf_loss' not in losses:
                losses['paf_loss'] = paf_losses['loss']
            else:
                losses['paf_loss'] += paf_losses['loss']

        return losses

    def forward(self, x):
        """Forward function."""
        if not isinstance(x, list):
            x = [x]

        assert max(self.heatmap_index) < len(x)
        assert max(self.paf_index) < len(x)

        final_outputs = {'heatmaps': [], 'pafs': []}

        for idx, head in enumerate(self.heatmap_heads_list):
            features = x[self.heatmap_index[idx]]
            output = head(features)
            final_outputs['heatmaps'].append(output)

        for idx, head in enumerate(self.paf_heads_list):
            features = x[self.paf_index[idx]]
            output = head(features)
            final_outputs['pafs'].append(output)

        return final_outputs

    def init_weights(self):
        for head in self.heatmap_heads_list:
            head.init_weights()

        for head in self.paf_heads_list:
            head.init_weights()
