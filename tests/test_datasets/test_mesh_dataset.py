# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

from mmpose.datasets import DATASETS


def test_mesh_Mosh_dataset():
    # test Mosh dataset
    dataset = 'MoshDataset'
    dataset_class = DATASETS.get(dataset)

    custom_dataset = dataset_class(
        ann_file='tests/data/mosh/test_mosh.npz', pipeline=[])

    _ = custom_dataset[0]


def test_mesh_H36M_dataset():
    # test H36M dataset
    dataset = 'MeshH36MDataset'
    dataset_class = DATASETS.get(dataset)

    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')
    _ = dataset_class(
        ann_file='tests/data/h36m/test_h36m.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=False)

    custom_dataset = dataset_class(
        ann_file='tests/data/h36m/test_h36m.npz',
        img_prefix='tests/data/h36m',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
    _ = custom_dataset[0]

    # test evaluation
    outputs = []
    for item in custom_dataset:
        pred = dict(
            keypoints_3d=item['joints_3d'][None, ...],
            image_path=item['image_file'])
        outputs.append(pred)
    with tempfile.TemporaryDirectory() as tmpdir:
        eval_result = custom_dataset.evaluate(outputs, tmpdir)
    assert 'MPJPE' in eval_result
    assert 'MPJPE-PA' in eval_result


def test_mesh_Mix_dataset():
    # test mesh Mix dataset

    dataset = 'MeshMixDataset'
    dataset_class = DATASETS.get(dataset)

    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')

    custom_dataset = dataset_class(
        configs=[
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
        ],
        partition=[0.6, 0.4])

    _ = custom_dataset[0]


def test_mesh_Adversarial_dataset():
    # test mesh Adversarial dataset

    # load train dataset
    data_cfg = dict(
        image_size=[256, 256],
        iuv_size=[64, 64],
        num_joints=24,
        use_IUV=True,
        uv_type='BF')
    train_dataset = dict(
        type='MeshMixDataset',
        configs=[
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
            dict(
                ann_file='tests/data/h36m/test_h36m.npz',
                img_prefix='tests/data/h36m',
                data_cfg=data_cfg,
                pipeline=[]),
        ],
        partition=[0.6, 0.4])

    # load adversarial dataset
    adversarial_dataset = dict(
        type='MoshDataset',
        ann_file='tests/data/mosh/test_mosh.npz',
        pipeline=[])

    # combine train and adversarial dataset to form a new dataset
    dataset = 'MeshAdversarialDataset'
    dataset_class = DATASETS.get(dataset)
    custom_dataset = dataset_class(train_dataset, adversarial_dataset)
    item = custom_dataset[0]
    assert 'mosh_theta' in item.keys()
