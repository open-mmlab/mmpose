# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmpose.models.detectors import DisentangledKeypointRegressor


def test_dekr_forward():
    model_cfg = dict(
        type='DisentangledKeypointRegressor',
        pretrained=None,
        backbone=dict(
            type='HRNet',
            in_channels=3,
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(4, ),
                    num_channels=(64, )),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='BASIC',
                    num_blocks=(4, 4),
                    num_channels=(32, 64)),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='BASIC',
                    num_blocks=(4, 4, 4),
                    num_channels=(32, 64, 128)),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block='BASIC',
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(32, 64, 128, 256),
                    multiscale_output=True)),
        ),
        keypoint_head=dict(
            type='DEKRHead',
            in_channels=(32, 64, 128, 256),
            in_index=(0, 1, 2, 3),
            num_joints=17,
            input_transform='resize_concat',
            heatmap_loss=dict(
                type='JointsMSELoss',
                use_target_weight=True,
            ),
            offset_loss=dict(
                type='SoftWeightSmoothL1Loss',
                use_target_weight=True,
            )),
        train_cfg=dict(),
        test_cfg=dict(
            num_joints=17,
            max_num_people=30,
            project2image=False,
            align_corners=False,
            nms_kernel=5,
            nms_padding=2,
            use_nms=True,
            nms_dist_thr=0.05,
            nms_joints_thr=8,
            keypoint_threshold=0.01,
            rescore_cfg=dict(in_channels=74, norm_indexes=(5, 6)),
            flip_test=True))

    detector = DisentangledKeypointRegressor(model_cfg['backbone'],
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
    heatmaps = mm_inputs.pop('heatmaps')
    masks = mm_inputs.pop('masks')
    offsets = mm_inputs.pop('offsets')
    offset_weights = mm_inputs.pop('offset_weights')
    img_metas = mm_inputs.pop('img_metas')

    # Test forward train
    losses = detector.forward(
        imgs,
        heatmaps,
        masks,
        offsets,
        offset_weights,
        img_metas,
        return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    detector.eval()
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        _ = detector.forward_dummy(imgs)

        # test rescore net
        preds = np.random.rand(2, 17, 3)
        _ = detector.rescore_net(preds, img_metas[0]['skeleton'])

        # test without flip_test
        detector.test_cfg['flip_test'] = False
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    heatmaps = np.zeros([N, 18, H // 4, W // 4], dtype=np.float32)
    masks = np.ones([N, 18, H // 4, W // 4], dtype=np.float32)
    offsets = np.zeros([N, 34, H // 4, W // 4], dtype=np.float32)
    offset_weights = np.ones([N, 34, H // 4, W // 4], dtype=np.float32)

    img_metas = [{
        'image_file':
        'test.jpg',
        'num_joints':
        17,
        'aug_data': [torch.zeros(1, 3, 256, 256),
                     torch.zeros(1, 3, 128, 128)],
        'test_scale_factor': [1, 0.5],
        'base_size': (256, 256),
        'image_size':
        256,
        'heatmap_size': [64],
        'center':
        np.array([128, 128]),
        'scale':
        np.array([1.28, 1.28]),
        'flip_index':
        [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        'skeleton': [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11],
                     [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2],
                     [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'heatmaps': [torch.FloatTensor(heatmaps)],
        'masks': [torch.FloatTensor(masks)],
        'offsets': [torch.FloatTensor(offsets)],
        'offset_weights': [torch.FloatTensor(offset_weights)],
        'img_metas': img_metas
    }
    return mm_inputs
