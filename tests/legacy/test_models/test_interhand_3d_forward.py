# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.models import build_posenet


def test_interhand3d_forward():
    # model settings
    model_cfg = dict(
        type='Interhand3D',
        pretrained='torchvision://resnet50',
        backbone=dict(type='ResNet', depth=50),
        keypoint_head=dict(
            type='Interhand3DHead',
            keypoint_head_cfg=dict(
                in_channels=2048,
                out_channels=21 * 64,
                depth_size=64,
                num_deconv_layers=3,
                num_deconv_filters=(256, 256, 256),
                num_deconv_kernels=(4, 4, 4),
            ),
            root_head_cfg=dict(
                in_channels=2048,
                heatmap_size=64,
                hidden_dims=(512, ),
            ),
            hand_type_head_cfg=dict(
                in_channels=2048,
                num_labels=2,
                hidden_dims=(512, ),
            ),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
            loss_root_depth=dict(type='L1Loss'),
            loss_hand_type=dict(type='BCELoss', use_target_weight=True),
        ),
        train_cfg={},
        test_cfg=dict(flip_test=True, shift_heatmap=True))

    detector = build_posenet(model_cfg)
    detector.init_weights()

    input_shape = (2, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    target = mm_inputs.pop('target')
    target_weight = mm_inputs.pop('target_weight')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    losses = detector.forward(
        imgs, target, target_weight, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        _ = detector.forward_dummy(imgs)


def _demo_mm_inputs(input_shape=(1, 3, 256, 256), num_outputs=None):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    imgs = torch.FloatTensor(imgs)

    target = [
        imgs.new_zeros(N, 42, 64, H // 4, W // 4),
        imgs.new_zeros(N, 1),
        imgs.new_zeros(N, 2),
    ]
    target_weight = [
        imgs.new_ones(N, 42, 1),
        imgs.new_ones(N, 1),
        imgs.new_ones(N),
    ]

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'bbox_id': 0,
        'flip_pairs': [],
        'inference_channel': np.arange(42),
        'image_file': '<demo>.png',
        'heatmap3d_depth_bound': 400.0,
        'root_depth_bound': 400.0,
    } for _ in range(N)]

    mm_inputs = {
        'imgs': imgs.requires_grad_(True),
        'target': target,
        'target_weight': target_weight,
        'img_metas': img_metas
    }
    return mm_inputs
