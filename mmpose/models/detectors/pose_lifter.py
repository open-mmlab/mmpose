from .. import builder
from ..registry import POSENETS
from .base import BasePose


@POSENETS.register_module()
class PoseLifter(BasePose):
    """Pose lifter that lifts 2D pose to 3D pose."""

    def __init__(self,
                 backbone,
                 neck=None,
                 keypoint_head=None,
                 traj_backbone=None,
                 traj_neck=None,
                 traj_head=None,
                 loss_semi=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # semi-supervised learning
        self.semi = keypoint_head is not None and traj_head is not None and \
            loss_semi is not None

        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if keypoint_head is not None:
            keypoint_head['train_cfg'] = train_cfg
            keypoint_head['test_cfg'] = test_cfg
            self.keypoint_head = builder.build_head(keypoint_head)

        if traj_backbone is not None:
            self.traj_backbone = builder.build_backbone(traj_backbone)
            if traj_neck is not None:
                self.traj_neck = builder.build_neck(traj_neck)
            if traj_head is not None:
                self.traj_head = builder.build_head(traj_head)

        if self.semi:
            self.loss_semi = builder.build_loss(loss_semi)

        self.init_weights(pretrained=pretrained)

    @property
    def with_neck(self):
        """Check if has keypoint_neck."""
        return hasattr(self, 'neck')

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, 'keypoint_head')

    @property
    def with_traj_backbone(self):
        """Check if has trajectory_backbone."""
        return hasattr(self, 'traj_backbone')

    @property
    def with_traj_neck(self):
        """Check if has trajectory_neck."""
        return hasattr(self, 'traj_neck')

    @property
    def with_traj(self):
        """Check if has trajectory_head."""
        return hasattr(self, 'traj_head')

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.backbone.init_weights(pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()
        if self.with_traj_backbone:
            self.traj_backbone.init_weights(pretrained)
        if self.with_traj_neck:
            self.traj_neck.init_weights()
        if self.with_traj:
            self.traj_head.init_weights()

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

        # supervised learning
        # pose model
        features = self.backbone(input)
        if self.with_neck:
            features = self.neck(features)
        if self.with_keypoint:
            output = self.keypoint_head(features)
        # trajectory model
        if self.with_traj:
            traj_features = self.traj_backbone(input)
            if self.with_traj_neck:
                traj_features = self.traj_neck(traj_features)
            traj_output = self.traj_head(traj_features)

        losses = dict()
        if self.with_keypoint:
            keypoint_losses = self.keypoint_head.get_loss(
                output, target, target_weight)
            keypoint_accuracy = self.keypoint_head.get_accuracy(
                output, target, target_weight, metas)
            losses.update(keypoint_losses)
            losses.update(keypoint_accuracy)
        if self.with_traj:
            traj_losses = self.traj_head.get_loss(traj_output,
                                                  kwargs['traj_target'], None)
            losses.update(traj_losses)

        # semi-supervised learning
        if self.semi:
            ul_input = kwargs['unlabeled_input']
            ul_features = self.backbone(ul_input)
            if self.with_neck:
                ul_features = self.neck(ul_features)
            ul_output = self.keypoint_head(ul_features)

            ul_traj_features = self.traj_backbone(ul_input)
            if self.with_traj_neck:
                ul_traj_features = self.traj_neck(ul_traj_features)
            ul_traj_output = self.traj_head(ul_traj_features)

            output_semi = dict(
                labeled_pose=output,
                unlabeled_pose=ul_output,
                unlabeled_traj=ul_traj_output)
            target_semi = dict(
                unlabeled_target_2d=kwargs['unlabeled_target_2d'],
                intrinsics=kwargs['intrinsics'])

            semi_losses = self.loss_semi(output_semi, target_semi)
            losses.update(semi_losses)

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

        if self.with_traj:
            traj_features = self.traj_backbone(input)
            if self.with_traj_neck:
                traj_features = self.traj_neck(traj_features)
            traj_output = self.traj_head.inference_model(traj_features)
            results['traj_preds'] = traj_output

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

        if self.with_traj:
            traj_features = self.traj_backbone(input)
            if self.with_neck:
                traj_features = self.traj_neck(traj_features)
            traj_output = self.traj_head(traj_features)
            output = output + traj_output

        return output

    def show_result(self, **kwargs):
        pass
