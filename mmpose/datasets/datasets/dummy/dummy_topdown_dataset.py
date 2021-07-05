import copy

import numpy as np

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.top_down.topdown_base_dataset import \
    TopDownBaseDataset


@DATASETS.register_module()
class DummyTopDownDataset(TopDownBaseDataset):
    """Dummy top-down dataset for model speed test."""

    inner_size = 1000

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.ann_info['input_size'] = data_cfg['input_size']
        self.ann_info['flip_pairs'] = []
        self.ann_info['upper_body_ids'] = []
        self.ann_info['lower_body_ids'] = []
        self.ann_info['joint_weights'] = [
            1.0 for _ in range(self.ann_info['num_output_channels'])
        ]
        self.ann_info['use_different_joint_weights'] = False

        self.size = 5000 if test_mode else 100000
        self.db = self._get_db()

    def _get_db(self):
        db = []
        w, h = self.ann_info['input_size']

        # dummy bbox and keypoints
        bbox = [int(w / 4), int(h / 4), int(w / 2), int(h / 2)]
        center, scale = self._xywh2cs(*bbox)
        num_kpts = self.ann_info['num_output_channels']
        kpts = np.random.rand(num_kpts, 2).astype(np.float32) * [w, h]
        kpts_visible = np.ones((num_kpts, 1), dtype=np.float32)

        dummy = {
            'img': np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8),
            'center': center,
            'scale': scale,
            'bbox': bbox,
            'rotation': 0,
            'joints_3d': kpts,
            'joints_3d_visible': kpts_visible,
            'bbox_score': 1
        }
        for i in range(self.size):
            sample = dummy.copy()
            sample['image_file'] = 'dummy'
            sample['bbox_id'] = i
            db.append(sample)

        return db

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * 1.25

        return center, scale

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        return dict(AP=1.)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        results = copy.deepcopy(self.db[idx % self.inner_size])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
