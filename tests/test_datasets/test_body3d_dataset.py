import tempfile

import numpy as np

from mmpose.datasets import DATASETS
from mmpose.datasets.builder import build_dataset


def test_body3d_h36m_dataset():
    # Test Human3.6M dataset
    dataset = 'Body3DH36MDataset'
    dataset_class = DATASETS.get(dataset)

    # test single-frame input
    data_cfg = dict(
        num_joints=17,
        seq_len=1,
        seq_frame_interval=1,
        joint_2d_src='pipeline',
        joint_2d_det_file=None,
        causal=False,
        need_camera_param=True,
        camera_param_file='tests/data/h36m/cameras.pkl')

    _ = dataset_class(
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
    _ = custom_dataset[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = []
        for result in custom_dataset:
            outputs.append({
                'preds': result['target'][None, ...],
                'target_image_paths': [result['target_image_path']],
            })

        metrics = ['mpjpe', 'p-mpjpe', 'n-mpjpe']
        infos = custom_dataset.evaluate(outputs, tmpdir, metrics)

        np.testing.assert_almost_equal(infos['MPJPE'], 0.0)
        np.testing.assert_almost_equal(infos['P-MPJPE'], 0.0)
        np.testing.assert_almost_equal(infos['N-MPJPE'], 0.0)

    # test multi-frame input with joint_2d_src = 'detection'
    data_cfg = dict(
        num_joints=17,
        seq_len=27,
        seq_frame_interval=1,
        causal=True,
        temporal_padding=True,
        joint_2d_src='detection',
        joint_2d_det_file='tests/data/h36m/test_h36m_2d_detection.npy',
        need_camera_param=True,
        camera_param_file='tests/data/h36m/cameras.pkl')

    _ = dataset_class(
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
    _ = custom_dataset[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = []
        for result in custom_dataset:
            outputs.append({
                'preds': result['target'][None, ...],
                'target_image_paths': [result['target_image_path']],
            })

        metrics = ['mpjpe', 'p-mpjpe', 'n-mpjpe']
        infos = custom_dataset.evaluate(outputs, tmpdir, metrics)

        np.testing.assert_almost_equal(infos['MPJPE'], 0.0)
        np.testing.assert_almost_equal(infos['P-MPJPE'], 0.0)
        np.testing.assert_almost_equal(infos['N-MPJPE'], 0.0)


def test_body3d_semi_supervision_dataset():
    # Test Body3d Semi-supervision Dataset

    # load labeled dataset
    labeled_data_cfg = dict(
        num_joints=17,
        seq_len=27,
        seq_frame_interval=1,
        causall=False,
        temporal_padding=True,
        joint_2d_src='gt',
        subset=1,
        subjects=['S1'],
        need_camera_param=True,
        camera_param_file='tests/data/h36m/cameras.pkl')
    labeled_dataset = dict(
        type='Body3DH36MDataset',
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=labeled_data_cfg,
        pipeline=[])

    # load unlabled data
    unlabeled_data_cfg = dict(
        num_joints=17,
        seq_len=27,
        seq_frame_interval=1,
        causal=False,
        temporal_padding=True,
        joint_2d_src='gt',
        subjects=['S5', 'S7', 'S8'],
        need_camera_param=True,
        camera_param_file='tests/data/h36m/cameras.pkl',
        need_2d_label=True)
    unlabeled_dataset = dict(
        type='Body3DH36MDataset',
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=unlabeled_data_cfg,
        pipeline=[
            dict(
                type='Collect',
                keys=[('input_2d', 'unlabeled_input')],
                meta_name='metas',
                meta_keys=[])
        ])

    # combine labeled and unlabeled dataset to form a new dataset
    dataset = 'Body3DSemiSupervisionDataset'
    dataset_class = DATASETS.get(dataset)
    custom_dataset = dataset_class(labeled_dataset, unlabeled_dataset)
    item = custom_dataset[0]
    assert 'unlabeled_input' in item.keys()

    unlabeled_dataset = build_dataset(unlabeled_dataset)
    assert len(unlabeled_dataset) == len(custom_dataset)
