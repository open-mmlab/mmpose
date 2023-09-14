# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmengine.dist import all_reduce_dict, get_dist_info
from mmengine.hooks import Hook
from torch import nn

from mmpose.registry import HOOKS


def get_norm_states(module: nn.Module) -> OrderedDict:
    """Get the state_dict of batch norms in the module."""
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states['.'.join([name, k])] = v
    return async_norm_states


@HOOKS.register_module()
class SyncNormHook(Hook):
    """Synchronize Norm states before validation."""

    def before_val_epoch(self, runner):
        """Synchronize normalization statistics."""
        module = runner.model
        rank, world_size = get_dist_info()

        if world_size == 1:
            return

        norm_states = get_norm_states(module)
        if len(norm_states) == 0:
            return

        try:
            norm_states = all_reduce_dict(norm_states, op='mean')
            module.load_state_dict(norm_states, strict=True)
        except Exception as e:
            runner.logger.warn(f'SyncNormHook failed: {str(e)}')
