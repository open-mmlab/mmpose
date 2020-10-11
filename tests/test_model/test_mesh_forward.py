import os
import pickle

import numpy as np
import torch
from scipy.sparse import csc_matrix

from mmpose.core.optimizer import build_optimizers
from mmpose.models.detectors.mesh import ParametricMesh


def _generate_smpl_weight_file(output_dir):
    """Generate a SMPL model weight file to initialize SMPL model, and generate
    a 3D joints regressor file."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joint_regressor_file = os.path.join(output_dir, 'test_joint_regressor.npy')
    np.save(joint_regressor_file, np.zeros([24, 6890]))

    test_model_file = os.path.join(output_dir, 'SMPL_NEUTRAL.pkl')
    test_data = {}
    test_data['f'] = np.zeros([1, 3], dtype=np.int32)
    test_data['J_regressor'] = csc_matrix(np.zeros([24, 6890]))
    test_data['kintree_table'] = np.zeros([2, 24], dtype=np.uint32)
    test_data['J'] = np.zeros([24, 3])
    test_data['weights'] = np.zeros([6890, 24])
    test_data['posedirs'] = np.zeros([6890, 3, 207])
    test_data['v_template'] = np.zeros([6890, 3])
    test_data['shapedirs'] = np.zeros([6890, 3, 10])
    with open(test_model_file, 'wb') as out_file:
        pickle.dump(test_data, out_file)
    return


def test_parametric_mesh_forward():
    """Test parametric mesh forward."""

    # generate weight file for SMPL model.
    _generate_smpl_weight_file('tests/data/smpl')

    # Test ParametricMesh without discriminator
    model_cfg = dict(
        pretrained=None,
        backbone=dict(type='ResNet', depth=50),
        mesh_head=dict(
            type='MeshHMRHead',
            in_channels=2048,
            smpl_mean_params='tests/data/smpl/smpl_mean_params.npz',
        ),
        disc=None,
        smpl=dict(
            smpl_path='tests/data/smpl',
            joints_regressor='tests/data/smpl/test_joint_regressor.npy'),
        train_cfg=dict(disc_step=1),
        test_cfg=dict(
            flip_test=False,
            post_process=True,
            shift_heatmap=True,
            unbiased_decoding=False,
            modulate_kernel=11),
        loss_mesh=dict(
            type='MeshLoss',
            joints_2d_loss_weight=1,
            joints_3d_loss_weight=1,
            vertex_loss_weight=1,
            smpl_pose_loss_weight=1,
            smpl_beta_loss_weight=1,
            focal_length=5000,
            img_res=256),
        loss_gan=None)
    optimizers_config = dict(generator=dict(type='Adam', lr=0.0001))

    detector = ParametricMesh(**model_cfg)
    detector.init_weights()
    optims = build_optimizers(detector, optimizers_config)

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)
    # Test forward train
    output = detector.train_step(mm_inputs, optims)
    assert isinstance(output, dict)

    # Test forward test
    with torch.no_grad():
        output = detector.val_step(data_batch=mm_inputs)
        assert isinstance(output, tuple)

        imgs = mm_inputs.pop('img')
        img_metas = mm_inputs.pop('img_metas')
        output = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        assert isinstance(output, tuple)

    # Test ParametricMesh with discriminator
    model_cfg['disc'] = dict()
    model_cfg['loss_gan'] = dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1)

    optimizers_config['discriminator'] = dict(type='Adam', lr=0.0001)

    detector = ParametricMesh(**model_cfg)
    detector.init_weights()
    optims = build_optimizers(detector, optimizers_config)

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape)
    # Test forward train
    output = detector.train_step(mm_inputs, optims)
    assert isinstance(output, dict)

    # Test forward test
    with torch.no_grad():
        output = detector.val_step(data_batch=mm_inputs)
        assert isinstance(output, tuple)

        imgs = mm_inputs.pop('img')
        img_metas = mm_inputs.pop('img_metas')
        output = detector.forward(imgs, img_metas=img_metas, return_loss=False)
        assert isinstance(output, tuple)


def _demo_mm_inputs(input_shape=(1, 3, 256, 256)):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
    """
    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)
    joints_2d = np.zeros([N, 24, 2])
    joints_2d_visible = np.ones([N, 24, 1])
    joints_3d = np.zeros([N, 24, 3])
    joints_3d_visible = np.ones([N, 24, 1])
    pose = np.zeros([N, 72])
    beta = np.zeros([N, 10])
    has_smpl = np.ones([N])
    mosh_theta = np.zeros([N, 3 + 72 + 10])

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
        'img': torch.FloatTensor(imgs).requires_grad_(True),
        'joints_2d': torch.FloatTensor(joints_2d),
        'joints_2d_visible': torch.FloatTensor(joints_2d_visible),
        'joints_3d': torch.FloatTensor(joints_3d),
        'joints_3d_visible': torch.FloatTensor(joints_3d_visible),
        'pose': torch.FloatTensor(pose),
        'beta': torch.FloatTensor(beta),
        'has_smpl': torch.FloatTensor(has_smpl),
        'img_metas': img_metas,
        'mosh_theta': torch.FloatTensor(mosh_theta)
    }

    return mm_inputs
