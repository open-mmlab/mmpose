# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import torch

from mmpose.models.detectors import PoseWarper, TopDown


def test_vipnas_forward():
    # model settings

    channel_cfg = dict(
        num_output_channels=17,
        dataset_joints=17,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ],
        inference_channel=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ])

    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(type='ViPNAS_ResNet', depth=50),
        keypoint_head=dict(
            type='ViPNASHeatmapSimpleHead',
            in_channels=608,
            out_channels=channel_cfg['num_output_channels'],
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11))

    detector = TopDown(model_cfg['backbone'], None, model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'])

    input_shape = (1, 3, 256, 256)
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


def test_topdown_forward():
    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(type='ResNet', depth=18),
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=512,
            out_channels=17,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11))

    detector = TopDown(model_cfg['backbone'], None, model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'])

    detector.init_weights()

    input_shape = (1, 3, 256, 256)
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

    # flip test
    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(
            type='HourglassNet',
            num_stacks=1,
        ),
        keypoint_head=dict(
            type='TopdownHeatmapMultiStageHead',
            in_channels=256,
            out_channels=17,
            num_stages=1,
            num_deconv_layers=0,
            extra=dict(final_conv_kernel=1, ),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=False)),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11))

    detector = TopDown(model_cfg['backbone'], None, model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'])

    # Test forward train
    losses = detector.forward(
        imgs, target, target_weight, img_metas, return_loss=True)
    assert isinstance(losses, dict)

    # Test forward test
    with torch.no_grad():
        _ = detector.forward(imgs, img_metas=img_metas, return_loss=False)

    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(
            type='HourglassNet',
            num_stacks=1,
        ),
        keypoint_head=dict(
            type='TopdownHeatmapMultiStageHead',
            in_channels=256,
            out_channels=17,
            num_stages=1,
            num_deconv_layers=0,
            extra=dict(final_conv_kernel=1, ),
            loss_keypoint=[
                dict(
                    type='JointsMSELoss',
                    use_target_weight=True,
                    loss_weight=1.)
            ]),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=True,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11))

    detector = TopDown(model_cfg['backbone'], None, model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'])

    detector.init_weights()

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_outputs=None)

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

    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(
            type='RSN',
            unit_channels=256,
            num_stages=1,
            num_units=4,
            num_blocks=[2, 2, 2, 2],
            num_steps=4,
            norm_cfg=dict(type='BN')),
        keypoint_head=dict(
            type='TopdownHeatmapMSMUHead',
            out_shape=(64, 48),
            unit_channels=256,
            out_channels=17,
            num_stages=1,
            num_units=4,
            use_prm=False,
            norm_cfg=dict(type='BN'),
            loss_keypoint=[dict(type='JointsMSELoss', use_target_weight=True)]
            * 3 + [dict(type='JointsOHKMMSELoss', use_target_weight=True)]),
        train_cfg=dict(num_units=4),
        test_cfg=dict(
            flip_test=True,
            post_process='default',
            shift_heatmap=False,
            unbiased_decoding=False,
            modulate_kernel=5))

    detector = TopDown(model_cfg['backbone'], None, model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'])

    detector.init_weights()

    input_shape = (1, 3, 256, 192)
    mm_inputs = _demo_mm_inputs(input_shape, num_outputs=4)

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


def test_posewarper_forward():
    # test PoseWarper
    model_cfg = dict(
        type='PoseWarper',
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
                    num_channels=(48, 96)),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='BASIC',
                    num_blocks=(4, 4, 4),
                    num_channels=(48, 96, 192)),
                stage4=dict(
                    num_modules=3,
                    num_branches=4,
                    block='BASIC',
                    num_blocks=(4, 4, 4, 4),
                    num_channels=(48, 96, 192, 384))),
            frozen_stages=4,
        ),
        concat_tensors=True,
        neck=dict(
            type='PoseWarperNeck',
            in_channels=48,
            freeze_trans_layer=True,
            out_channels=17,
            inner_channels=128,
            deform_groups=17,
            dilations=(3, 6, 12, 18, 24),
            trans_conv_kernel=1,
            res_blocks_cfg=dict(block='BASIC', num_blocks=20),
            offsets_kernel=3,
            deform_conv_kernel=3),
        keypoint_head=dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=17,
            out_channels=17,
            num_deconv_layers=0,
            extra=dict(final_conv_kernel=0, ),
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=False,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11))

    detector = PoseWarper(model_cfg['backbone'], model_cfg['neck'],
                          model_cfg['keypoint_head'], model_cfg['train_cfg'],
                          model_cfg['test_cfg'], model_cfg['pretrained'], None,
                          model_cfg['concat_tensors'])
    assert detector.concat_tensors

    detector.init_weights()

    input_shape = (2, 3, 64, 64)
    num_frames = 2
    mm_inputs = _demo_mm_inputs(input_shape, None, num_frames)

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

    # test argument 'concat_tensors'
    model_cfg_copy = copy.deepcopy(model_cfg)
    model_cfg_copy['concat_tensors'] = False

    detector = PoseWarper(model_cfg_copy['backbone'], model_cfg_copy['neck'],
                          model_cfg_copy['keypoint_head'],
                          model_cfg_copy['train_cfg'],
                          model_cfg_copy['test_cfg'],
                          model_cfg_copy['pretrained'], None,
                          model_cfg_copy['concat_tensors'])
    assert not detector.concat_tensors

    detector.init_weights()

    input_shape = (2, 3, 64, 64)
    num_frames = 2
    mm_inputs = _demo_mm_inputs(input_shape, None, num_frames)

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

    # flip test
    model_cfg_copy = copy.deepcopy(model_cfg)
    model_cfg_copy['test_cfg']['flip_test'] = True

    detector = PoseWarper(model_cfg_copy['backbone'], model_cfg_copy['neck'],
                          model_cfg_copy['keypoint_head'],
                          model_cfg_copy['train_cfg'],
                          model_cfg_copy['test_cfg'],
                          model_cfg_copy['pretrained'], None,
                          model_cfg_copy['concat_tensors'])

    detector.init_weights()

    input_shape = (1, 3, 64, 64)
    num_frames = 2
    mm_inputs = _demo_mm_inputs(input_shape, None, num_frames)

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

    # test different number of dilations
    model_cfg_copy = copy.deepcopy(model_cfg)
    model_cfg_copy['neck']['dilations'] = (3, 6, 12)

    detector = PoseWarper(model_cfg_copy['backbone'], model_cfg_copy['neck'],
                          model_cfg_copy['keypoint_head'],
                          model_cfg_copy['train_cfg'],
                          model_cfg_copy['test_cfg'],
                          model_cfg_copy['pretrained'], None,
                          model_cfg_copy['concat_tensors'])

    detector.init_weights()

    input_shape = (2, 3, 64, 64)
    num_frames = 2
    mm_inputs = _demo_mm_inputs(input_shape, None, num_frames)

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

    # test different backbone, change head accordingly
    model_cfg_copy = copy.deepcopy(model_cfg)
    model_cfg_copy['backbone'] = dict(type='ResNet', depth=18)
    model_cfg_copy['neck']['in_channels'] = 512
    model_cfg_copy['keypoint_head'] = dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=17,
        out_channels=17,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True))

    detector = PoseWarper(model_cfg_copy['backbone'], model_cfg_copy['neck'],
                          model_cfg_copy['keypoint_head'],
                          model_cfg_copy['train_cfg'],
                          model_cfg_copy['test_cfg'],
                          model_cfg_copy['pretrained'], None,
                          model_cfg_copy['concat_tensors'])

    detector.init_weights()

    input_shape = (1, 3, 64, 64)
    num_frames = 2
    mm_inputs = _demo_mm_inputs(input_shape, None, num_frames)

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


def _demo_mm_inputs(
        input_shape=(1, 3, 256, 256), num_outputs=None, num_frames=1):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_frames (int):
            number of frames for each sample, default: 1,
            if larger than 1, return a list of tensors
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    if num_outputs is not None:
        target = np.zeros([N, num_outputs, 17, H // 4, W // 4],
                          dtype=np.float32)
        target_weight = np.ones([N, num_outputs, 17, 1], dtype=np.float32)
    else:
        target = np.zeros([N, 17, H // 4, W // 4], dtype=np.float32)
        target_weight = np.ones([N, 17, 1], dtype=np.float32)

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'bbox_id': 0,
        'flip_pairs': [],
        'inference_channel': np.arange(17),
        'image_file': '<demo>.png',
        'frame_weight': np.random.uniform(0, 1, num_frames),
    } for _ in range(N)]

    mm_inputs = {
        'target': torch.FloatTensor(target),
        'target_weight': torch.FloatTensor(target_weight),
        'img_metas': img_metas
    }

    if num_frames == 1:
        imgs = torch.FloatTensor(rng.rand(*input_shape)).requires_grad_(True)
    else:

        imgs = [
            torch.FloatTensor(rng.rand(*input_shape)).requires_grad_(True)
            for _ in range(num_frames)
        ]

    mm_inputs['imgs'] = imgs
    return mm_inputs
