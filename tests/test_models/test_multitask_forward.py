# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models.detectors import MultiTask


def test_multitask_forward():
    """Test multitask forward."""

    # build MultiTask detector
    model_cfg = dict(
        backbone=dict(type='ResNet', depth=50),
        heads=[
            dict(
                type='DeepposeRegressionHead',
                in_channels=2048,
                num_joints=17,
                loss_keypoint=dict(
                    type='SmoothL1Loss', use_target_weight=False)),
        ],
        necks=[dict(type='GlobalAveragePooling')],
        head2neck={0: 0},
        pretrained=None,
    )
    model = MultiTask(**model_cfg)
    with pytest.raises(TypeError):
        model.init_weights(pretrained=dict())
    model.pretrained = model_cfg['pretrained']
    # build inputs and target
    mm_inputs = _demo_mm_inputs()
    inputs = mm_inputs['img']
    target = [mm_inputs['target_keypoints']]
    target_weight = [mm_inputs['target_weight']]
    img_metas = mm_inputs['img_metas']

    # Test forward train
    losses = model(inputs, target, target_weight, return_loss=True)
    assert 'reg_loss' in losses and 'acc_pose' in losses

    # Test forward test
    outputs = model(inputs, img_metas=img_metas, return_loss=False)
    assert 'preds' in outputs

    # Test dummy forward
    outputs = model.forward_dummy(inputs)
    assert outputs[0].shape == torch.Size([1, 17, 2])

    # Build multitask detector with no neck
    model_cfg = dict(
        backbone=dict(type='ResNet', depth=50),
        heads=[
            dict(
                type='TopdownHeatmapSimpleHead',
                in_channels=2048,
                out_channels=17,
                num_deconv_layers=3,
                num_deconv_filters=(256, 256, 256),
                num_deconv_kernels=(4, 4, 4),
                loss_keypoint=dict(
                    type='JointsMSELoss', use_target_weight=True))
        ],
        pretrained=None,
    )
    model = MultiTask(**model_cfg)

    # build inputs and target
    target = [mm_inputs['target_heatmap']]

    # Test forward train
    losses = model(inputs, target, target_weight, return_loss=True)
    assert 'heatmap_loss' in losses and 'acc_pose' in losses

    # Test forward test
    outputs = model(inputs, img_metas=img_metas, return_loss=False)
    assert 'preds' in outputs

    # Test dummy forward
    outputs = model.forward_dummy(inputs)
    assert outputs[0].shape == torch.Size([1, 17, 64, 64])


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    target_keypoints = np.zeros([N, 17, 2])
    target_heatmap = np.zeros([N, 17, H // 4, W // 4])
    target_weight = np.ones([N, 17, 1])

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'bbox_id': 0,
        'flip_pairs': [],
        'inference_channel': np.arange(17),
        'image_file': '<demo>.png',
    } for _ in range(N)]

    mm_inputs = {
        'img': torch.FloatTensor(imgs).requires_grad_(True),
        'target_keypoints': torch.FloatTensor(target_keypoints),
        'target_heatmap': torch.FloatTensor(target_heatmap),
        'target_weight': torch.FloatTensor(target_weight),
        'img_metas': img_metas,
    }
    return mm_inputs
