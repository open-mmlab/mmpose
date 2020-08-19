from mmpose.datasets import DATASETS


def test_mesh_H36M_dataset():
    dataset = 'MeshH36MDataset'
    # test H36M datasets
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
