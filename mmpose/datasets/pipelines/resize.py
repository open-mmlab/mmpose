import mmcv
import pdb
import cv2
import numpy as np
from ..registry import PIPELINES


@PIPELINES.register_module()
class Resize:
    def __init__(self,
                 resize_shape):
        self.resize_shape = resize_shape

    def __call__(self, results):
        """
        Resize the image to the given input shape, depending on which dimension
        is the largest.
        """
        img = results["img"]
        img_y, img_x, _ = img.shape
        if img_y > img_x:
            target_y = max(self.resize_shape)
            target_x = min(self.resize_shape)
        else:
            target_y = min(self.resize_shape)
            target_x = max(self.resize_shape)
        img = cv2.resize(img, (int(target_x), int(target_y)))
        results["img"] = img
        upscale_factor = np.array([target_x / img_x, target_y / img_y])
        results["center"] *= upscale_factor
        results["scale"] *= upscale_factor
        # img = cv2.resize(img, (img_x, img_y))
        return results
