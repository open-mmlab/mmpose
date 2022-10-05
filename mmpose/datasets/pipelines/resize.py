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
        if "center" in results:
            results["center"] *= upscale_factor
        if "scale" in results:
            results["scale"] *= upscale_factor
        if "bbox" in results:
            results["bbox"] = [results["bbox"][0] * upscale_factor[0],
                               results["bbox"][1] * upscale_factor[1],
                               results["bbox"][2] * upscale_factor[0],
                               results["bbox"][3] * upscale_factor[1]]
        if "joints_3d" in results:
            results["joints_3d"][:, :2] = results["joints_3d"][:, :2] * upscale_factor
        return results
