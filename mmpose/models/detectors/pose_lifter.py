import warnings

import mmcv
import numpy as np

from mmpose.core import imshow_keypoints, imshow_keypoints_3d
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

    def show_result(self,
                    result,
                    img=None,
                    skeleton=None,
                    pose_kpt_color=None,
                    pose_limb_color=None,
                    viz_hight=400,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Visualize 3D pose estimation results.

        Args:
            result (list[dict]): The pose estimation results containing:
                - "keypoints_3d" ([K,4]): 3D keypoints
                - "keypoints" ([K,3] or [T,K,3]): Optional for visualizing
                    2D inputs. If a sequence is given, only the last frame
                    will be used for visualization
                - "bbox" ([4,] or [T,4]): Optional for visualizing 2D inputs
                - "title" (str): title for the subplot
            img (str or Tensor): Optional. The image to visualize 2D inputs on.
            skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
                limbs, each is a pair of joint indices.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_limb_color (np.array[Mx3]): Color of M limbs.
                If None, do not draw limbs.
            viz_hight (int): The image hight of the visualization. The width
                will be N*viz_hight depending on the number of visualized
                items.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """

        assert len(result) > 0
        result = sorted(result, key=lambda x: x.get('track_id', 0))

        # draw image and input 2d poses
        if img is not None:
            img = mmcv.imread(img)

            bbox_result = []
            pose_input_2d = []
            for res in result:
                if 'bbox' in res:
                    bbox = np.array(res['bbox'])
                    if bbox.ndim != 1:
                        assert bbox.ndim == 2
                        bbox = bbox[-1]  # Get bbox from the last frame
                    bbox_result.append(bbox)
                if 'keypoints' in res:
                    kpts = np.array(res['keypoints'])
                    if kpts.ndim != 2:
                        assert kpts.ndim == 3
                        kpts = kpts[-1]  # Get 2D keypoints from the last frame
                    pose_input_2d.append(kpts)

            if len(bbox_result) > 0:
                bboxes = np.vstack(bbox_result)
                mmcv.imshow_bboxes(
                    img,
                    bboxes,
                    colors='green',
                    top_k=-1,
                    thickness=2,
                    show=False)
            if len(pose_input_2d) > 0:
                imshow_keypoints(
                    img,
                    pose_input_2d,
                    skeleton,
                    kpt_score_thr=0.3,
                    pose_kpt_color=pose_kpt_color,
                    pose_limb_color=pose_limb_color,
                    radius=8,
                    thickness=2)
            img = mmcv.imrescale(img, scale=viz_hight / img.shape[0])

        img_vis = imshow_keypoints_3d(result, img, skeleton, pose_kpt_color,
                                      pose_limb_color, viz_hight)

        if show:
            mmcv.visualization.imshow(img_vis, win_name, wait_time)

        if out_file is not None:
            mmcv.imwrite(img_vis, out_file)

        return img_vis
