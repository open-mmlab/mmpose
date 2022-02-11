# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import time
from contextlib import contextmanager
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np
from torch.hub import HASH_REGEX, download_url_to_file


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
        image = cv2.imread(filename, readFlag)
        return image


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get_cached_file_path(url,
                         save_dir=None,
                         progress=True,
                         check_hash=False,
                         file_name=None):
    r"""Loads the Torch serialized object at the given URL.

    If downloaded file is a zip file, it will be automatically decompressed

    If the object is already present in `model_dir`, it's deserialized and
    returned.
    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url (str): URL of the object to download
        save_dir (str, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar
            to stderr. Default: True
        check_hash(bool, optional): If True, the filename part of the URL
            should follow the naming convention ``filename-<sha256>.ext``
            where ``<sha256>`` is the first eight or more digits of the
            SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name (str, optional): name for the downloaded file. Filename
            from ``url`` will be used if not set. Default: None.
    """
    if save_dir is None:
        save_dir = os.path.join('webcam_resources')

    mkdir_or_exist(save_dir)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(save_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def screen_matting(img, color_low=None, color_high=None, color=None):
    """Screen Matting.

    Args:
        img (np.ndarray): Image data.
        color_low (tuple): Lower limit (b, g, r).
        color_high (tuple): Higher limit (b, g, r).
        color (str): Support colors include:

            - 'green' or 'g'
            - 'blue' or 'b'
            - 'black' or 'k'
            - 'white' or 'w'
    """

    if color_high is None or color_low is None:
        if color is not None:
            if color.lower() == 'g' or color.lower() == 'green':
                color_low = (0, 200, 0)
                color_high = (60, 255, 60)
            elif color.lower() == 'b' or color.lower() == 'blue':
                color_low = (230, 0, 0)
                color_high = (255, 40, 40)
            elif color.lower() == 'k' or color.lower() == 'black':
                color_low = (0, 0, 0)
                color_high = (40, 40, 40)
            elif color.lower() == 'w' or color.lower() == 'white':
                color_low = (230, 230, 230)
                color_high = (255, 255, 255)
            else:
                NotImplementedError(f'Not supported color: {color}.')
        else:
            ValueError('color or color_high | color_low should be given.')

    mask = cv2.inRange(img, np.array(color_low), np.array(color_high)) == 0

    return mask.astype(np.uint8)


def expand_and_clamp(box, im_shape, s=1.25):
    """Expand the bbox and clip it to fit the image shape.

    Args:
        box (list): x1, y1, x2, y2
        im_shape (ndarray): image shape (h, w, c)
        s (float): expand ratio

    Returns:
        list: x1, y1, x2, y2
    """

    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    deta_w = w * (s - 1) / 2
    deta_h = h * (s - 1) / 2

    x1, y1, x2, y2 = x1 - deta_w, y1 - deta_h, x2 + deta_w, y2 + deta_h

    img_h, img_w = im_shape[:2]

    x1 = min(max(0, int(x1)), img_w - 1)
    y1 = min(max(0, int(y1)), img_h - 1)
    x2 = min(max(0, int(x2)), img_w - 1)
    y2 = min(max(0, int(y2)), img_h - 1)

    return [x1, y1, x2, y2]


def _find_connected_components(mask):
    """Find connected components and sort with areas.

    Args:
        mask (ndarray): instance segmentation result.

    Returns:
        ndarray (N, 5): Each item contains (x, y, w, h, area).
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    stats = stats[stats[:, 4].argsort()]
    return stats


def _find_bbox(mask):
    """Find the bounding box for the mask.

    Args:
        mask (ndarray): Mask.

    Returns:
        list(4, ): Returned box (x1, y1, x2, y2).
    """
    mask_shape = mask.shape
    if len(mask_shape) == 3:
        assert mask_shape[-1] == 1, 'the channel of the mask should be 1.'
    elif len(mask_shape) == 2:
        pass
    else:
        NotImplementedError()

    h, w = mask_shape[:2]
    mask_w = mask.sum(0)
    mask_h = mask.sum(1)

    left = 0
    right = w - 1
    up = 0
    down = h - 1

    for i in range(w):
        if mask_w[i] > 0:
            break
        left += 1

    for i in range(w - 1, left, -1):
        if mask_w[i] > 0:
            break
        right -= 1

    for i in range(h):
        if mask_h[i] > 0:
            break
        up += 1

    for i in range(h - 1, up, -1):
        if mask_h[i] > 0:
            break
        down -= 1

    return [left, up, right, down]


def copy_and_paste(img,
                   background_img,
                   mask,
                   bbox=None,
                   effect_region=(0.2, 0.2, 0.8, 0.8),
                   min_size=(20, 20)):
    """Copy the image region and paste to the background.

    Args:
        img (np.ndarray): Image data.
        background_img (np.ndarray): Background image data.
        mask (ndarray): instance segmentation result.
        bbox (ndarray): instance bbox, (x1, y1, x2, y2).
        effect_region (tuple(4, )): The region to apply mask, the coordinates
            are normalized (x1, y1, x2, y2).
    """
    background_img = background_img.copy()
    background_h, background_w = background_img.shape[:2]
    region_h = (effect_region[3] - effect_region[1]) * background_h
    region_w = (effect_region[2] - effect_region[0]) * background_w
    region_aspect_ratio = region_w / region_h

    if bbox is None:
        bbox = _find_bbox(mask)
    instance_w = bbox[2] - bbox[0]
    instance_h = bbox[3] - bbox[1]

    if instance_w > min_size[0] and instance_h > min_size[1]:
        aspect_ratio = instance_w / instance_h
        if region_aspect_ratio > aspect_ratio:
            resize_rate = region_h / instance_h
        else:
            resize_rate = region_w / instance_w

        mask_inst = mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_inst = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_inst = cv2.resize(img_inst, (int(
            resize_rate * instance_w), int(resize_rate * instance_h)))
        mask_inst = cv2.resize(
            mask_inst,
            (int(resize_rate * instance_w), int(resize_rate * instance_h)),
            interpolation=cv2.INTER_NEAREST)

        mask_ids = list(np.where(mask_inst == 1))
        mask_ids[1] += int(effect_region[0] * background_w)
        mask_ids[0] += int(effect_region[1] * background_h)

        background_img[tuple(mask_ids)] = img_inst[np.where(mask_inst == 1)]

    return background_img


def is_image_file(path):
    if isinstance(path, str):
        if path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return True
    return False


class ImageCapture:
    """A mock-up version of cv2.VideoCapture that always return a const image.

    Args:
        image (str | ndarray): The image or image path
    """

    def __init__(self, image):
        if isinstance(image, str):
            self.image = load_image_from_disk_or_url(image)
        else:
            self.image = image

    def isOpened(self):
        return (self.image is not None)

    def read(self):
        return True, self.image.copy()

    def release(self):
        pass

    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return self.image.shape[1]
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.image.shape[0]
        elif propId == cv2.CAP_PROP_FPS:
            return np.nan
        else:
            raise NotImplementedError()
