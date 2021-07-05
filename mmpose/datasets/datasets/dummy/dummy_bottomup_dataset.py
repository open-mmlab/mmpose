import copy  # noqa: F401

import numpy as np

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.bottom_up.bottom_up_base_dataset import \
    BottomUpBaseDataset


@DATASETS.register_module()
class DummyBottomUpDataset(BottomUpBaseDataset):

    inner_size = 1000

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.ann_info['flip_index'] = []
        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = [
            1.0 for _ in range(self.ann_info['num_joints'])
        ]

        # joint index starts from 1
        self.ann_info['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13],
                                     [12, 13], [6, 12], [7, 13], [6, 7],
                                     [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                                     [1, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                                     [5, 7]]

        self.size = 5000 if test_mode else 100000
        self.db = self._get_db()

    def _get_db(self):
        db = []
        w = h = self.ann_info['image_size']

        # dummy bbox and keypoints
        num_kpts = self.ann_info['num_joints']
        joints = np.random.rand(num_kpts, 2).astype(np.float32) * [w, h]
        joints_visible = np.ones((num_kpts, 1), dtype=np.float32)
        joints = np.concatenate((joints, joints_visible), axis=-1)
        num_classes = 1
        num_joints_per_class = 1

        dummy = {
            'dataset': 'coco',
            'image_file': 'dummy',
            'img': np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8),
            'joints':
            [np.stack([joints] * num_joints_per_class)] * num_classes,
            'mask': [np.ones((h, w), dtype=bool)] * num_classes
        }
        db = [dummy] * self.size

        return db

    def __len__(self):
        return self.size

    def _get_single(self, idx):
        return self.db[idx % self.inner_size]
