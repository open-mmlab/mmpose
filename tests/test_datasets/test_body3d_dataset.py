import tempfile

import numpy as np

from mmpose.datasets import DATASETS


def test_body3d_h36m_dataset():
    # Test Human3.6M dataset
    dataset = 'Body3DH36MDataset'
    dataset_class = DATASETS.get(dataset)

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
