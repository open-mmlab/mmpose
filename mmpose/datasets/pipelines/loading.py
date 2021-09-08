# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = mmcv.imread(image_file, self.color_type, self.channel_order)

        if img is None:
            raise ValueError(f'Fail to read {image_file}')

        results['img'] = img
        return results
