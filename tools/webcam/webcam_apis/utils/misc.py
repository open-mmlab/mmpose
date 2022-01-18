# Copyright (c) OpenMMLab. All rights reserved.
import time
from contextlib import contextmanager
from typing import Optional
from urllib.request import urlopen

import cv2
import numpy as np


@contextmanager
def limit_max_fps(fps: Optional[float]):
    t_start = time.time()
    try:
        yield
    finally:
        t_end = time.time()
        if fps is not None:
            t_sleep = 1.0 / fps - t_end + t_start
            if t_sleep > 0:
                time.sleep(t_sleep)


def _is_url(filename):
    """Check if the file is a url link.

    Args:
        filename (str): the file name or url link.

    Returns:
        bool: is url or not.
    """
    prefixes = ['http://', 'https://']
    for p in prefixes:
        if filename.startswith(p):
            return True
    return False


def load_image_from_disk_or_url(filename, readFlag=cv2.IMREAD_COLOR):
    """Load an image file, from disk or url.

    Args:
        filename (str): file name on the disk or url link.
        readFlag (int): readFlag for imdecode.

    Returns:
        np.ndarray: A loaded image
    """
    if _is_url(filename):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urlopen(filename)
        image = np.asarray(bytearray(resp.read()), dtype='uint8')
        image = cv2.imdecode(image, readFlag)
        return image
    else:
        image = cv2.imread(filename)
        return image
