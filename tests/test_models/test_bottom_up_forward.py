# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models.backbones.litepose import LitePose
from mmpose.models.detectors import AssociativeEmbedding
from mmpose.models.heads.litepose_head import LitePoseHead


def test_litepose_forward():
    loss_keypoint = dict(
        type='MultiLossFactory',
        num_joints=17,
        num_stages=2,
        ae_loss_type='exp',
        with_ae_loss=[True, False],
        push_loss_factor=[0.001, 0.001],
        pull_loss_factor=[0.001, 0.001],
        with_heatmaps_loss=[True, True],
        heatmaps_loss_factor=[1.0, 1.0])
    backbone = LitePose(
        num_blocks=(6, 8, 10, 10),
        strides=(2, 2, 2, 1),
        channels=(16, 32, 48, 80),
        block_settings=(
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7]],
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7]],
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7],
             [6, 7], [6, 7]],
            [[6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7], [6, 7],
             [6, 7], [6, 7]],
        ),
        input_channel=16)
    head = LitePoseHead(
        deconv_setting=(16, 24, 24),
        num_deconv_layers=3,
        num_deconv_kernels=(4, 4, 4),
        num_joints=17,  # coco
        tag_per_joint=True,
        with_heatmaps_loss=(True, True),
        with_ae_loss=(True, False),
        channels=(16, 16, 32, 48, 80),
        loss_keypoint=loss_keypoint)
    x = torch.randn(4, 3, 256, 256)
    feat = backbone(x)
    outputs = head(feat)
    assert len(outputs) == 2
    assert outputs[0].shape[2] == x.shape[2] // 4
    assert outputs[0].shape[1] == 17 * 2
    assert outputs[1].shape[2] == x.shape[2] // 2
    assert outputs[1].shape[1] == 17


def test_ae_forward():
    model_cfg = dict(
        type='AssociativeEmbedding',
        pretrained=None,
        backbone=dict(type='ResNet', depth=18),
        keypoint_head=dict(
            type='AESimpleHead',
            in_channels=512,
            num_joints=17,
            num_deconv_layers=0,
            tag_per_joint=True,
            with_ae_loss=[True],
            extra=dict(final_conv_kernel=1, ),
            loss_keypoint=dict(
                type='MultiLossFactory',
                num_joints=17,
                num_stages=1,
                ae_loss_type='exp',
                with_ae_loss=[True],
                push_loss_factor=[0.001],
                pull_loss_factor=[0.001],
                with_heatmaps_loss=[True],
                heatmaps_loss_factor=[1.0])),
        train_cfg=dict(),
        test_cfg=dict(
            num_joints=17,
            max_num_people=30,
            scale_factor=[1],
            with_heatmaps=[True],
            with_ae=[True],
            project2image=True,
            nms_kernel=5,
            nms_padding=2,
            tag_per_joint=True,
            detection_threshold=0.1,
            tag_threshold=1,
            use_detection_val=True,
            ignore_too_much=False,
            adjust=True,
            refine=True,
            soft_nms=False,
            flip_test=True,
            post_process=True,
            shift_heatmap=True,
            use_gt_bbox=True,
            flip_pairs=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                        [13, 14], [15, 16]],
        ))

    detector = AssociativeEmbedding(model_cfg['backbone'],
                                    model_cfg['keypoint_head'],
                                    model_cfg['train_cfg'],
                                    model_cfg['test_cfg'],
                                    model_cfg['pretrained'])

    with pytest.raises(TypeError):
        detector.init_weights(pretrained=dict())
    detector.pretrained = model_cfg['pretrained']
    detector.init_weights()

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)

    imgs = mm_inputs.pop('imgs')
    target = mm_inputs.pop('target')
    mask = mm_inputs.pop('mask')
    joints = mm_inputs.pop('joints')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    losses = detector.forward(
        imgs, target, mask, joints, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        _ = detector.forward_dummy(imgs)


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    target = np.zeros([N, 17, H // 32, W // 32], dtype=np.float32)
    mask = np.ones([N, H // 32, W // 32], dtype=np.float32)
    joints = np.zeros([N, 30, 17, 2], dtype=np.float32)

    img_metas = [{
        'image_file':
        'test.jpg',
        'aug_data': [torch.zeros(1, 3, 256, 256)],
        'test_scale_factor': [1],
        'base_size': (256, 256),
        'center':
        np.array([128, 128]),
        'scale':
        np.array([1.28, 1.28]),
        'flip_index':
        [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'target': [torch.FloatTensor(target)],
        'mask': [torch.FloatTensor(mask)],
        'joints': [torch.FloatTensor(joints)],
        'img_metas': img_metas
    }
    return mm_inputs
