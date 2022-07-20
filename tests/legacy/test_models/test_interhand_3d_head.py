# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.models import Interhand3DHead


def test_interhand_3d_head():
    """Test interhand 3d head."""
    N = 4
    input_shape = (N, 2048, 8, 8)
    inputs = torch.rand(input_shape, dtype=torch.float32)
    target = [
        inputs.new_zeros(N, 42, 64, 64, 64),
        inputs.new_zeros(N, 1),
        inputs.new_zeros(N, 2),
    ]
    target_weight = [
        inputs.new_ones(N, 42, 1),
        inputs.new_ones(N, 1),
        inputs.new_ones(N),
    ]

    img_metas = [{
        'img_shape': (256, 256, 3),
        'center': np.array([112, 112]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'bbox_id': 0,
        'flip_pairs': [],
        'inference_channel': np.arange(42),
        'image_file': '<demo>.png',
        'heatmap3d_depth_bound': 400.0,
        'root_depth_bound': 400.0,
    } for _ in range(N)]

    head = Interhand3DHead(
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
        train_cfg={},
        test_cfg={},
    )
    head.init_weights()

    # test forward
    output = head(inputs)
    assert isinstance(output, list)
    assert len(output) == 3
    assert output[0].shape == (N, 42, 64, 64, 64)
    assert output[1].shape == (N, 1)
    assert output[2].shape == (N, 2)

    # test loss computation
    losses = head.get_loss(output, target, target_weight)
    assert 'hand_loss' in losses
    assert 'rel_root_loss' in losses
    assert 'hand_type_loss' in losses

    # test inference model
    flip_pairs = [[i, 21 + i] for i in range(21)]
    output = head.inference_model(inputs, flip_pairs)
    assert isinstance(output, list)
    assert len(output) == 3
    assert output[0].shape == (N, 42, 64, 64, 64)
    assert output[1].shape == (N, 1)
    assert output[2].shape == (N, 2)

    # test decode
    result = head.decode(img_metas, output)
    assert 'preds' in result
    assert 'rel_root_depth' in result
    assert 'hand_type' in result
