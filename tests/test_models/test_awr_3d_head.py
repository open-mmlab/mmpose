# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.models import AdaptiveWeightingRegression3DHead


def test_awr_3d_head():
    N = 4
    input_shape = (N, 2048, 8, 8)
    inputs = torch.rand(input_shape, dtype=torch.float32)

    img_input_shape = (N, 1, 128, 128)
    img_inputs = torch.rand(img_input_shape, dtype=torch.float32)

    target = [
        inputs.new_ones(N, 14 * 4, 64, 64),
        inputs.new_ones(N, 14, 3),
    ]
    target_weight = [
        inputs.new_ones(N, 14),
        inputs.new_ones(N, 14, 3),
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

    print('fake input OK')

    head = AdaptiveWeightingRegression3DHead(
        deconv_head_cfg=dict(
            in_channels=2048,
            out_channels=256,
            depth_size=64,
            num_deconv_layers=3,
            num_deconv_filters=(256, 256, 256),
            num_deconv_kernels=(4, 4, 4),
            extra=dict(final_conv_kernel=0, )),
        offset_head_cfg=dict(
            in_channels=256,
            out_channels_vector=42,
            out_channels_scalar=14,
            heatmap_kernel_size=0.4,
        ),
        loss_keypoint=dict(type='AWRSmoothL1Loss', use_target_weight=True),
        loss_offset=dict(type='AWRSmoothL1Loss', use_target_weight=False),
        train_cfg=dict(use_img_for_head=True),
        test_cfg=dict(use_img_for_head=True, flip_test=False))

    print('init OK')

    head.init_weights()

    # test forward
    inputs_with_img = (inputs, img_inputs)
    output = head(inputs_with_img)
    assert isinstance(output, list)
    assert len(output) == 2
    assert output[0].shape == (N, 14 * 4, 64, 64)
    assert output[1].shape == (N, 14, 3)

    # test loss computation
    losses = head.get_loss(output, target, target_weight)
    assert 'joint_loss' in losses
    assert 'offset_loss' in losses

    # test inference model
    output = head.inference_model(inputs_with_img, flip_pairs=None)
    assert isinstance(output, list)
    assert len(output) == 2
    assert output[0].shape == (N, 14 * 4, 64, 64)
    assert output[1].shape == (N, 14, 3)

    # test decode
    result = head.decode(img_metas, output)
    assert 'preds' in result
    assert 'preds_xyz' in result
