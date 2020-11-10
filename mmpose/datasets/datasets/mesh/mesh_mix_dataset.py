from abc import ABCMeta

import numpy as np
from torch.utils.data import Dataset

from mmpose.datasets.builder import DATASETS
from .mesh_base_dataset import MeshBaseDataset


@DATASETS.register_module()
class MeshMixDataset(Dataset, metaclass=ABCMeta):
    """Mix Dataset for 3D human mesh estimation.

    The dataset combines data from multiple datasets (MeshBaseDataset) and
    sample the data from different datasets with the provided proportions.
    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        configs (list): List of configs for multiple datasets.
        partition (list): Sample proportion of multiple datasets.
            The the elements of it should be non-negative and the
            sum of it should be 1.
    """

    def __init__(self, configs, partition):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        assert sum(partition) == 1
        self.partition = np.array(partition).cumsum()
        self.datasets = [MeshBaseDataset(**cfg) for cfg in configs]
        self.length = max(len(ds) for ds in self.datasets)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                index_new = (idx + np.random.rand()) * len(
                    self.datasets[i]) / self.length
                index_new = int(np.round(index_new)) % (len(self.datasets[i]))
                return self.datasets[i][index_new]
        return None
