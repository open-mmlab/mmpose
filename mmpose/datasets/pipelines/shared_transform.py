import warnings
from collections.abc import Sequence

import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.utils import build_from_cfg
from torchvision.transforms import functional as F

from ..registry import PIPELINES

try:
    import albumentations
except ImportError:
    albumentations = None


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

        meta = {}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                meta[key] = results[key]
        if 'bbox_id' in results:
            meta['bbox_id'] = results['bbox_id']
        data[self.meta_name] = DC(meta, cpu_only=True)

        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')


@PIPELINES.register_module()
class Albumentation(object):
    """Albumentation augmentation (pixel-level transforms only). Adds custom
    pixel-level transformations from Albumentations library. Please visit
    `https://albumentations.readthedocs.io` to get more information.

    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.

    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of Albumentation transformations
        keymap (dict): Contains {'input key':'albumentation-style key'},
            e.g., {'img': 'image'}.
    """

    def __init__(self, transforms, keymap=None):
        self.allowed_transforms = [
            'Blur', 'CLAHE', 'ChannelDropout', 'ChannelShuffle', 'ColorJitter',
            'Downscale', 'Emboss', 'Equalize', 'FDA', 'FancyPCA', 'FromFloat',
            'GaussNoise', 'GaussianBlur', 'GlassBlur', 'HistogramMatching',
            'HueSaturationValue', 'IAAAdditiveGaussianNoise', 'IAAEmboss',
            'IAASharpen', 'IAASuperpixels', 'ISONoise', 'ImageCompression',
            'InvertImg', 'MedianBlur', 'MotionBlur', 'MultiplicativeNoise',
            'Normalize', 'Posterize', 'RGBShift', 'RandomBrightnessContrast',
            'RandomFog', 'RandomGamma', 'RandomRain', 'RandomShadow',
            'RandomSnow', 'RandomSunFlare', 'Sharpen', 'Solarize', 'ToFloat',
            'ToGray', 'ToSepia'
        ]

        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            if obj_type not in self.allowed_transforms:
                warnings.warn(
                    '{obj_type} is not supported. Please use with caution.')
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {keymap.get(k, k): v for k, v in d.items()}
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)
        # back to the original format
        results = self.mapper(results, self.keymap_back)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str
