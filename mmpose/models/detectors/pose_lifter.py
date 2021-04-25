import warnings

from .. import builder
from ..registry import POSENETS
from .base import BasePose

try:
    from mmcv.runner import auto_fp16
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import auto_fp16


@POSENETS.register_module()
class PoseLifter(BasePose):
    """Pose lifter that lifts 2D pose to 3D pose."""

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.fp16_enabled = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg
            self.keypoint_head = builder.build_head(keypoint_head)

        self.init_weights(pretrained=pretrained)

    @property
    def with_neck(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    @auto_fp16(apply_to=('input', ))
    def forward(self,
                input,
                target=None,
                target_weight=None,
                metas=None,
                return_loss=True,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note:
            Note:
            batch_size: N
            num_input_keypoints: Ki
            input_keypoint_dim: Ci
            input_sequence_len: Ti
            num_output_keypoints: Ko
            output_keypoint_dim: Co
            input_sequence_len: To



        Args:
            input (torch.Tensor[NxKixCixTi]): Input keypoint coordinates.
            target (torch.Tensor[NxKoxCoxTo]): Output keypoint coordinates.
                Defaults to None.
            target_weight (torch.Tensor[NxKox1]): Weights across different
                joint types. Defaults to None.
            metas (list(dict)): Information about data augmentation
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
        Returns:
            dict|Tensor: if `reutrn_loss` is true, return losses. Otherwise
                return predicted poses
        """
        if return_loss:
            return self.forward_train(input, target, target_weight, metas,
                                      **kwargs)
        else:
            return self.forward_test(input, metas, **kwargs)

    def forward_train(self, input, target, target_weight, metas, **kwargs):
        """Defines the computation performed at every call when training."""
        assert input.size(0) == len(metas)

        features = self.backbone(input)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output = self.keypoint_head(features)

        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight, metas)

            losses.update(keypoint_losses)
            losses.update(keypoint_accuracy)

        return losses

    def forward_test(self, input, metas, **kwargs):
        """Defines the computation performed at every call when training."""
        assert input.size(0) == len(metas)

        results = {}

        features = self.backbone(input)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output = self.keypoint_head.inference_model(features)
            keypoint_result = self.keypoint_head.decode(metas, output)
            results.update(keypoint_result)

        return results

    def forward_dummy(self, input):
        """Used for computing network FLOPs.

        See ``tools/get_flops.py``.

        Args:
            input (torch.Tensor): Input pose

        Returns:
            Tensor: Model output
        """
        features = self.backbone(input)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output = self.keypoint_head(features)

        return output

    def show_result(self, **kwargs):
        pass
