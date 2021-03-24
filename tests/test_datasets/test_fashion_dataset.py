import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from mmpose.datasets import DATASETS


def convert_db_to_output(db, batch_size=2, keys=None, is_3d=False):
    outputs = []
    len_db = len(db)
    for i in range(0, len_db, batch_size):
        if is_3d:
            keypoints = np.stack([
                db[j]['joints_3d'].reshape((-1, 3))
                for j in range(i, min(i + batch_size, len_db))
            ])
        else:
            keypoints = np.stack([
                np.hstack([
                    db[j]['joints_3d'].reshape((-1, 3))[:, :2],
                    db[j]['joints_3d_visible'].reshape((-1, 3))[:, :1]
                ]) for j in range(i, min(i + batch_size, len_db))
            ])
        image_paths = [
            db[j]['image_file'] for j in range(i, min(i + batch_size, len_db))
        ]
        bbox_ids = [j for j in range(i, min(i + batch_size, len_db))]
        box = np.stack(
            np.array([
                db[j]['center'][0], db[j]['center'][1], db[j]['scale'][0],
                db[j]['scale'][1], db[j]['scale'][0] * db[j]['scale'][1] *
                200 * 200, 1.0
            ],
                     dtype=np.float32)
            for j in range(i, min(i + batch_size, len_db)))

        output = {}
        output['preds'] = keypoints
        output['boxes'] = box
        output['image_paths'] = image_paths
        output['output_heatmap'] = None
        output['bbox_ids'] = bbox_ids

        if keys is not None:
            keys = keys if isinstance(keys, list) else [keys]
            for key in keys:
                output[key] = [
                    db[j][key] for j in range(i, min(i + batch_size, len_db))
                ]

        outputs.append(output)

    return outputs


def test_deepfashion_dataset():
    dataset = 'DeepFashionDataset'
    # test JHMDB datasets
    dataset_class = DATASETS.get(dataset)
    dataset_class.load_annotations = MagicMock()
    dataset_class.coco = MagicMock()

    channel_cfg = dict(
        num_output_channels=8,
        dataset_joints=8,
        dataset_channel=[
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        inference_channel=[0, 1, 2, 3, 4, 5, 6, 7])

    data_cfg = dict(
        image_size=[192, 256],
        heatmap_size=[48, 64],
        num_output_channels=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        dataset_channel=channel_cfg['dataset_channel'],
        inference_channel=channel_cfg['inference_channel'],
        soft_nms=False,
        nms_thr=1.0,
        oks_thr=0.9,
        vis_thr=0.2,
        use_gt_bbox=True,
        det_bbox_thr=0.0,
        image_thr=0.0,
        bbox_file='')

    # Test gt bbox
    custom_dataset = dataset_class(
        ann_file='tests/data/fld/test_fld.json',
        img_prefix='tests/data/fld/',
        subset='full',
        data_cfg=data_cfg,
        pipeline=[],
        test_mode=True)

    assert custom_dataset.test_mode is True
    assert custom_dataset.dataset_name == 'deepfashion_full'

    image_id = 128
    assert image_id in custom_dataset.img_ids
    assert len(custom_dataset.img_ids) == 2
    _ = custom_dataset[0]

    outputs = convert_db_to_output(custom_dataset.db)
    with tempfile.TemporaryDirectory() as tmpdir:
        infos = custom_dataset.evaluate(outputs, tmpdir, ['PCK', 'EPE', 'AUC'])
    assert_almost_equal(infos['PCK'], 1.0)
    assert_almost_equal(infos['AUC'], 0.95)
    assert_almost_equal(infos['EPE'], 0.0)

    with pytest.raises(KeyError):
        infos = custom_dataset.evaluate(outputs, tmpdir, 'mAP')
