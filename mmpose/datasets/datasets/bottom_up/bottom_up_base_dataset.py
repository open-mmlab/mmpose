# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.pipelines import Compose


class BottomUpBaseDataset(Dataset):
    """Base class for bottom-up datasets.

    All datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_single`

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.

        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        self.image_info = {}
        self.ann_info = {}

        self.annotations_path = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        # bottom-up
        self.base_size = data_cfg['base_size']
        self.base_sigma = data_cfg['base_sigma']
        self.int_sigma = False

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']
        self.ann_info['num_scales'] = data_cfg['num_scales']
        self.ann_info['scale_aware_sigma'] = data_cfg['scale_aware_sigma']

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.use_nms = data_cfg.get('use_nms', False)
        self.soft_nms = data_cfg.get('soft_nms', True)
        self.oks_thr = data_cfg.get('oks_thr', 0.9)

        self.img_ids = []
        self.pipeline = Compose(self.pipeline)

    def __len__(self):
        """Get dataset length."""
        return len(self.img_ids)

    def _get_single(self, idx):
        """Get anno for a single image."""
        raise NotImplementedError

    def prepare_train_img(self, idx):
        """Prepare image for training given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Prepare image for testing given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def get_flip_index_from_flip_pairs(self, flip_pairs):
        flip_index = list(range(self.ann_info['num_joints']))
        for pair in flip_pairs:
            flip_index[pair[1]], flip_index[pair[0]] = flip_index[
                pair[0]], flip_index[pair[1]]
        return flip_index

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_img(idx)

        return self.prepare_train_img(idx)
