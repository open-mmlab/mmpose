# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.models import build_posenet


def test_interhand3d_forward():
    # model settings
    model_cfg = dict(
        type='Depthhand3D',  # pretrained=None
        backbone=dict(
            type='AWRResNet',
            depth=50,
            frozen_stages=-1,
            zero_init_residual=False,
            in_channels=1),
        keypoint_head=dict(
            type='AdaptiveWeightingRegression3DHead',
            offset_head_cfg=dict(
                in_channels=256,
                out_channels_vector=42,
                out_channels_scalar=14,
                heatmap_kernel_size=0.4,
            ),
            deconv_head_cfg=dict(
                in_channels=2048,
                out_channels=256,
                depth_size=64,
                num_deconv_layers=3,
                num_deconv_filters=(256, 256, 256),
                num_deconv_kernels=(4, 4, 4),
                extra=dict(final_conv_kernel=0, )),
            loss_offset=dict(type='AWRSmoothL1Loss', use_target_weight=False),
            loss_keypoint=dict(type='AWRSmoothL1Loss', use_target_weight=True),
        ),
        train_cfg=dict(use_img_for_head=True),
        test_cfg=dict(use_img_for_head=True, flip_test=False))

    detector = build_posenet(model_cfg)
    detector.init_weights()

    input_shape = (2, 1, 128, 128)
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


def _demo_mm_inputs(input_shape=(1, 1, 128, 128), num_outputs=None):
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
        imgs.new_ones(N, 14 * 4, 64, 64),
        imgs.new_ones(N, 14, 3),
    ]
    target_weight = [
        imgs.new_ones(N, 14),
        imgs.new_ones(N, 14, 3),
    ]

    cameras = {'fx': 588.03, 'fy': 587.07, 'cx': 320.0, 'cy': 240.0}

    img_metas = [{
        'img_shape': (128, 128, 3),
        'center': np.array([112, 112]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'bbox_id': 0,
        'flip_pairs': [],
        'inference_channel': np.arange(14),
        'cube_size': np.array([300, 300, 300]),
        'center_depth': 1.0,
        'focal': np.array([cameras['fx'], cameras['fy']]),
        'princpt': np.array([cameras['cx'], cameras['cy']]),
        'image_file': '<demo>.png',
    } for _ in range(N)]

    mm_inputs = {
        'imgs': imgs.requires_grad_(True),
        'target': target,
        'target_weight': target_weight,
        'img_metas': img_metas
    }
    return mm_inputs
