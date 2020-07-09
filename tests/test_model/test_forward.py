import numpy as np
import torch

from mmpose.models.detectors import TopDown


def test_topdown_forward():
    model_cfg = dict(
        type='TopDown',
        pretrained=None,
        backbone=dict(type='ResNet', depth=18),
        keypoint_head=dict(
            type='SimpleHead',
            in_channels=512,
            out_channels=17,
        ),
        train_cfg=dict(),
        test_cfg=dict(
            flip_test=False,
            post_process=True,
            shift_heatmap=True,
            unbiased_decoding=False,
            modulate_kernel=11),
        loss_pose=dict(type='JointsMSELoss', use_target_weight=False))

    detector = TopDown(model_cfg['backbone'], model_cfg['keypoint_head'],
                       model_cfg['train_cfg'], model_cfg['test_cfg'],
                       model_cfg['pretrained'], model_cfg['loss_pose'])

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
        detector = detector.cuda()
        _ = detector.forward(
            imgs.cuda(), img_metas=img_metas, return_loss=False)


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    target = np.zeros([N, 17, H // 4, W // 4])
    target_weight = np.ones([N, 17])

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'bbox_score': 1.0,
        'flip_pairs': [],
        'inference_channel': np.arange(17),
        'image_file': '<demo>.png',
    } for _ in range(N)]

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'target': torch.FloatTensor(target),
        'target_weight': torch.FloatTensor(target_weight),
        'img_metas': img_metas
    }
    return mm_inputs
