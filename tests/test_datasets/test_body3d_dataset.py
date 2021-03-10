from mmpose.datasets import DATASETS


def test_body3d_h36m_dataset():
    # Test Human3.6M dataset
    dataset = 'Body3DH36MDataset'
    dataset_class = DATASETS.get(dataset)

    data_cfg = dict(
        num_joints=17,
        seq_len=1,
        seq_frame_interval=1,
        joint_2d_src='gt',
        joint_2d_det_file=None,
        causal=False,
        need_camera_param=True,
        camera_param_file='tests/data/h36m/cameras.pkl')

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
