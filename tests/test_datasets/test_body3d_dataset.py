# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import numpy as np
from mmcv import Config

from mmpose.datasets import DATASETS
from mmpose.datasets.builder import build_dataset


def test_body3d_h36m_dataset():
    # Test Human3.6M dataset
    dataset = 'Body3DH36MDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/h36m.py').dataset_info

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
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.dataset_name == 'h36m'
    assert custom_dataset.test_mode is True
    _ = custom_dataset[0]

    results = []
    for result in custom_dataset:
        results.append({
            'preds': result['target'][None, ...],
            'target_image_paths': [result['target_image_path']],
        })

    metrics = ['mpjpe', 'p-mpjpe', 'n-mpjpe']
    infos = custom_dataset.evaluate(results, metric=metrics)

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
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        dataset_info=dataset_info,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
    _ = custom_dataset[0]

    results = []
    for result in custom_dataset:
        results.append({
            'preds': result['target'][None, ...],
            'target_image_paths': [result['target_image_path']],
        })

    metrics = ['mpjpe', 'p-mpjpe', 'n-mpjpe']
    infos = custom_dataset.evaluate(results, metric=metrics)

    np.testing.assert_almost_equal(infos['MPJPE'], 0.0)
    np.testing.assert_almost_equal(infos['P-MPJPE'], 0.0)
    np.testing.assert_almost_equal(infos['N-MPJPE'], 0.0)


def test_body3d_semi_supervision_dataset():
    # Test Body3d Semi-supervision Dataset
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/h36m.py').dataset_info

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
    labeled_dataset_cfg = dict(
        type='Body3DH36MDataset',
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=labeled_data_cfg,
        dataset_info=dataset_info,
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
    unlabeled_dataset_cfg = dict(
        type='Body3DH36MDataset',
        ann_file='tests/data/h36m/test_h36m_body3d.npz',
        img_prefix='tests/data/h36m',
        data_cfg=unlabeled_data_cfg,
        dataset_info=dataset_info,
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
    custom_dataset = dataset_class(labeled_dataset_cfg, unlabeled_dataset_cfg)
    item = custom_dataset[0]
    assert custom_dataset.labeled_dataset.dataset_name == 'h36m'
    assert 'unlabeled_input' in item.keys()

    unlabeled_dataset = build_dataset(unlabeled_dataset_cfg)
    assert len(unlabeled_dataset) == len(custom_dataset)


def test_body3d_mpi_inf_3dhp_dataset():
    # Test MPI-INF-3DHP dataset
    dataset = 'Body3DMpiInf3dhpDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/mpi_inf_3dhp.py').dataset_info

    # Test single-frame input on trainset
    single_frame_train_data_cfg = dict(
        num_joints=17,
        seq_len=1,
        seq_frame_interval=1,
        joint_2d_src='pipeline',
        joint_2d_det_file=None,
        causal=False,
        need_camera_param=True,
        camera_param_file='tests/data/mpi_inf_3dhp/cameras_train.pkl')

    # Test single-frame input on testset
    single_frame_test_data_cfg = dict(
        num_joints=17,
        seq_len=1,
        seq_frame_interval=1,
        joint_2d_src='gt',
        joint_2d_det_file=None,
        causal=False,
        need_camera_param=True,
        camera_param_file='tests/data/mpi_inf_3dhp/cameras_test.pkl')

    # Test multi-frame input on trainset
    multi_frame_train_data_cfg = dict(
        num_joints=17,
        seq_len=27,
        seq_frame_interval=1,
        joint_2d_src='gt',
        joint_2d_det_file=None,
        causal=True,
        temporal_padding=True,
        need_camera_param=True,
        camera_param_file='tests/data/mpi_inf_3dhp/cameras_train.pkl')

    # Test multi-frame input on testset
    multi_frame_test_data_cfg = dict(
        num_joints=17,
        seq_len=27,
        seq_frame_interval=1,
        joint_2d_src='pipeline',
        joint_2d_det_file=None,
        causal=False,
        temporal_padding=True,
        need_camera_param=True,
        camera_param_file='tests/data/mpi_inf_3dhp/cameras_test.pkl')

    ann_files = [
        'tests/data/mpi_inf_3dhp/test_3dhp_train.npz',
        'tests/data/mpi_inf_3dhp/test_3dhp_test.npz'
    ] * 2
    data_cfgs = [
        single_frame_train_data_cfg, single_frame_test_data_cfg,
        multi_frame_train_data_cfg, multi_frame_test_data_cfg
    ]

    for ann_file, data_cfg in zip(ann_files, data_cfgs):
        _ = dataset_class(
            ann_file=ann_file,
            img_prefix='tests/data/mpi_inf_3dhp',
            data_cfg=data_cfg,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=False)

        custom_dataset = dataset_class(
            ann_file=ann_file,
            img_prefix='tests/data/mpi_inf_3dhp',
            data_cfg=data_cfg,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=True)

        assert custom_dataset.test_mode is True
        _ = custom_dataset[0]

        results = []
        for result in custom_dataset:
            results.append({
                'preds': result['target'][None, ...],
                'target_image_paths': [result['target_image_path']],
            })

        metrics = ['mpjpe', 'p-mpjpe', '3dpck', 'p-3dpck', '3dauc', 'p-3dauc']
        infos = custom_dataset.evaluate(results, metric=metrics)

        np.testing.assert_almost_equal(infos['MPJPE'], 0.0)
        np.testing.assert_almost_equal(infos['P-MPJPE'], 0.0)
        np.testing.assert_almost_equal(infos['3DPCK'], 100.)
        np.testing.assert_almost_equal(infos['P-3DPCK'], 100.)
        np.testing.assert_almost_equal(infos['3DAUC'], 30 / 31 * 100)
        np.testing.assert_almost_equal(infos['P-3DAUC'], 30 / 31 * 100)


def test_body3dmview_direct_panoptic_dataset():
    # Test Mview-Panoptic dataset
    dataset = 'Body3DMviewDirectPanopticDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/panoptic_body3d.py').dataset_info
    space_size = [8000, 8000, 2000]
    space_center = [0, -500, 800]
    cube_size = [80, 80, 20]
    train_data_cfg = dict(
        image_size=[960, 512],
        heatmap_size=[[240, 128]],
        space_size=space_size,
        space_center=space_center,
        cube_size=cube_size,
        num_joints=15,
        seq_list=['160906_band1', '160906_band2'],
        cam_list=[(0, 12), (0, 6)],
        num_cameras=2,
        seq_frame_interval=1,
        subset='train',
        need_2d_label=True,
        need_camera_param=True,
        root_id=2)

    test_data_cfg = dict(
        image_size=[960, 512],
        heatmap_size=[[240, 128]],
        num_joints=15,
        space_size=space_size,
        space_center=space_center,
        cube_size=cube_size,
        seq_list=['160906_band1', '160906_band2'],
        cam_list=[(0, 12), (0, 6)],
        num_cameras=2,
        seq_frame_interval=1,
        subset='validation',
        need_2d_label=True,
        need_camera_param=True,
        root_id=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        _ = dataset_class(
            ann_file=tmpdir + '/tmp_train.pkl',
            img_prefix='tests/data/panoptic_body3d/',
            data_cfg=train_data_cfg,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dataset = dataset_class(
            ann_file=tmpdir + '/tmp_validation.pkl',
            img_prefix='tests/data/panoptic_body3d',
            data_cfg=test_data_cfg,
            pipeline=[],
            dataset_info=dataset_info,
            test_mode=False)

    import copy
    gt_num = test_dataset.db_size // test_dataset.num_cameras
    results = []
    for i in range(gt_num):
        index = test_dataset.num_cameras * i
        db_rec = copy.deepcopy(test_dataset.db[index])
        joints_3d = db_rec['joints_3d']
        joints_3d_vis = db_rec['joints_3d_visible']
        num_gts = len(joints_3d)
        gt_pose = -np.ones((1, 10, test_dataset.num_joints, 5))

        if num_gts > 0:
            gt_pose[0, :num_gts, :, :3] = np.array(joints_3d)
            gt_pose[0, :num_gts, :, 3] = np.array(joints_3d_vis)[:, :, 0] - 1.0
            gt_pose[0, :num_gts, :, 4] = 1.0

        results.append(dict(pose_3d=gt_pose, sample_id=[i]))
    _ = test_dataset.evaluate(results, metric=['mAP', 'mpjpe'])
