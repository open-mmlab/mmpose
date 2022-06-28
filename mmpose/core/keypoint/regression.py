# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np

from mmpose.core.post_processing import transform_preds


def keypoints_from_regression(regression_preds: np.ndarray, center: np.ndarray,
                              scale: np.ndarray, img_size: List[int]
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Get final keypoint predictions from regression vectors and transform
    them back to the image.

    Note:
        - batch_size: N
        - num_keypoints: K

    Args:
        regression_preds (np.ndarray[N, K, 2]): model prediction.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        img_size (list(img_width, img_height)): model input image size.

    Returns:
        tuple:

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    N, K, _ = regression_preds.shape
    preds, maxvals = regression_preds, np.ones((N, K, 1), dtype=np.float32)

    preds = preds * img_size

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], img_size)

    return preds, maxvals
