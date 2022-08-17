# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import LoadImageFromFile

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    """Load an image from file or from the np.ndarray in ``results['img']``.

    Required Keys:

        - img_path
        - img (optional)

    Modified Keys:

        - img
        - img_shape
        - ori_shape
        - img_path (optional)

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """The transform function of :class:`LoadImage`.

        Args:
            results (dict): The result dict

        Returns:
            dict: The result dict.
        """

        if 'img' not in results:
            # Load image from file by :meth:`LoadImageFromFile.transform`
            results = super().transform(results)
        else:
            img = results['img']
            assert isinstance(img, np.ndarray)
            if self.to_float32:
                img = img.astype(np.float32)

            if 'img_path' not in results:
                results['img_path'] = None
            results['img_shape'] = img.shape[:2]
            results['ori_shape'] = img.shape[:2]

        return results
