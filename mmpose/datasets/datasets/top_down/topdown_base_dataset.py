import copy
from abc import ABCMeta, abstractmethod

import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.pipelines import Compose


class TopDownBaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for top-down datasets.

    All top-down datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`

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

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['flip_pairs'] = None

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.db = []
        self.pipeline = Compose(self.pipeline)

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
