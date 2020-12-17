from collections.abc import Sequence

import numpy as np
from mmcv.parallel import DataContainer as DC
from mmcv.utils import build_from_cfg
from torchvision.transforms import functional as F

from ..registry import PIPELINES


@PIPELINES.register_module()
class ToTensor:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        results['img'] = F.to_tensor(results['img'])
        return results


@PIPELINES.register_module()
class NormalizeTensor:
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        results['img'] = F.normalize(
            results['img'], mean=self.mean, std=self.std)
        return results


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str]): Required keys to be collected.
        meta_name (str): The name of the key that contains meta infomation.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str]): Keys that are collected under meta_name.
          The contents of the `meta_name` dictionary depends on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            data[key] = results[key]

        if len(self.meta_keys) != 0:
            meta = {}
            for key in self.meta_keys:
                meta[key] = results[key]
            data[self.meta_name] = DC(meta, cpu_only=True)

        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')


@PIPELINES.register_module()
class HideAndSeek:
    """Augmentation by informantion dropping in Hide-and-Seek paradigm. Paper
    ref: Huang et al. AID: Pushing the Performance Boundary of Human Pose
    Estimation with Information Dropping Augmentation (arXiv:2008.07139 2020).

    Args:
        prob_has (float): Probability of performing hide-and-seek.
        prob_has_hide (float): Probability of hiding patches.
        grid_sizes (list): List of optional grid sizes.
    """

    def __init__(self,
                 prob_has=1.0,
                 prob_has_hide=0.5,
                 grid_sizes=(0, 16, 32, 44, 56)):
        self.prob_has = prob_has
        self.prob_has_hide = prob_has_hide
        self.grid_sizes = grid_sizes

    def _hide_and_seek(self, img):
        # get width and height of the image
        ht, wd, _ = img.shape

        # randomly choose one grid size
        index = np.random.randint(0, len(self.grid_sizes) - 1)
        grid_size = self.grid_sizes[index]

        # hide the patches
        if grid_size != 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if np.random.rand() <= self.prob_has_hide:
                        img[x:x_end, y:y_end, :] = 0
        return img

    def __call__(self, results):
        img = results['img']
        if np.random.rand() < self.prob_has:
            img = self._hide_and_seek(img)
        results['img'] = img
        return results


@PIPELINES.register_module()
class Cutout:
    """Augmentation by informantion dropping in Cutout paradigm. Paper ref:
    Huang et al. AID: Pushing the Performance Boundary of Human Pose Estimation
    with Information Dropping Augmentation (arXiv:2008.07139 2020).

    Args:
        prob_cutout (float): Probability of performing cutout.
        radius_factor (float): Size factor of cutout area.
        num_patch (float): Number of patches to be cutout.
    """

    def __init__(self, prob_cutout=1.0, radius_factor=0.2, num_patch=1):

        self.prob_cutout = prob_cutout
        self.radius_factor = radius_factor
        self.num_patch = num_patch

    def _cutout(self, img):
        height, width, _ = img.shape
        img = img.reshape(height * width, -1)
        feat_x_int = np.arange(0, width)
        feat_y_int = np.arange(0, height)
        feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
        feat_x_int = feat_x_int.flatten()
        feat_y_int = feat_y_int.flatten()
        for _ in range(self.num_patch):
            center = [np.random.rand() * width, np.random.rand() * height]
            radius = self.radius_factor * (1 + np.random.rand(2)) * width
            x_offset = (center[0] - feat_x_int) / radius[0]
            y_offset = (center[1] - feat_y_int) / radius[1]
            dis = x_offset**2 + y_offset**2
            indexes = np.where(dis <= 1)[0]
            img[indexes, :] = 0
        img = img.reshape(height, width, -1)
        return img

    def __call__(self, results):
        img = results['img']
        if np.random.rand() < self.prob_cutout:
            img = self._cutout(img)
        results['img'] = img
        return results
