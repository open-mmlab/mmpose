# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.pipelines import Compose


class GestureBaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for gesture recognition datasets."""

    def __init__(self,
                 ann_file,
                 vid_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        self.video_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.vid_prefix = vid_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['video_size'] = np.array(data_cfg['video_size'])
        self.modalities = data_cfg['modalities']
        if isinstance(self.modalities, (list, tuple)):
            self.modalities = self.modalities
        else:
            self.modalities = (self.modalities, )
        self.dataset_name = dataset_info.dataset_name
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_single(self, idx):
        """Get anno for a single video."""
        raise NotImplementedError

    def prepare_train_vid(self, idx):
        """Prepare video for training given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def prepare_test_vid(self, idx):
        """Prepare video for testing given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def __len__(self):
        """Get dataset length."""
        return len(self.vid_ids)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_vid(idx)

        return self.prepare_train_vid(idx)
