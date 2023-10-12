# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Tuple

import cv2
import numpy as np


def resize_image_to_fixed_height(image: np.ndarray,
                                 fixed_height: int) -> np.ndarray:
    """Resizes an input image to a specified fixed height while maintaining its
    aspect ratio.

    Args:
        image (np.ndarray): Input image as a numpy array [H, W, C]
        fixed_height (int): Desired fixed height of the output image.

    Returns:
        Resized image as a numpy array (fixed_height, new_width, channels).
    """
    original_height, original_width = image.shape[:2]

    scale_ratio = fixed_height / original_height
    new_width = int(original_width * scale_ratio)
    resized_image = cv2.resize(image, (new_width, fixed_height))

    return resized_image


def blend_images(img1: np.ndarray,
                 img2: np.ndarray,
                 blend_ratios: Tuple[float, float] = (1, 1)) -> np.ndarray:
    """Blends two input images with specified blend ratios.

    Args:
        img1 (np.ndarray): First input image as a numpy array [H, W, C].
        img2 (np.ndarray): Second input image as a numpy array [H, W, C]
        blend_ratios (tuple): A tuple of two floats representing the blend
            ratios for the two input images.

    Returns:
        Blended image as a numpy array [H, W, C]
    """

    def normalize_image(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image

    img1 = normalize_image(img1)
    img2 = normalize_image(img2)

    blended_image = img1 * blend_ratios[0] + img2 * blend_ratios[1]
    blended_image = blended_image.clip(min=0, max=1)
    blended_image = (blended_image * 255).astype(np.uint8)

    return blended_image


def convert_video_fps(video):

    input_video = video
    video_name, post_fix = input_video.rsplit('.', 1)
    output_video = f'{video_name}_30fps.{post_fix}'
    if os.path.exists(output_video):
        return output_video

    os.system(
        f"ffmpeg -i {input_video} -vf \"minterpolate='fps=30'\" {output_video}"
    )

    return output_video


def get_smoothed_kpt(kpts, index, sigma=5):
    """Smooths keypoints using a Gaussian filter."""
    assert kpts.shape[1] == 17
    assert kpts.shape[2] == 3
    assert sigma % 2 == 1

    num_kpts = len(kpts)

    start_idx = max(0, index - sigma // 2)
    end_idx = min(num_kpts, index + sigma // 2 + 1)

    # Extract a piece of the keypoints array to apply the filter
    piece = kpts[start_idx:end_idx].copy()
    original_kpt = kpts[index]

    # Split the piece into coordinates and scores
    coords, scores = piece[..., :2], piece[..., 2]

    # Calculate the Gaussian ratio for each keypoint
    gaussian_ratio = np.arange(len(scores)) + start_idx - index
    gaussian_ratio = np.exp(-gaussian_ratio**2 / 2)

    # Update scores using the Gaussian ratio
    scores *= gaussian_ratio[:, None]

    # Compute the smoothed coordinates
    smoothed_coords = (coords * scores[..., None]).sum(axis=0) / (
        scores[..., None].sum(axis=0) + 1e-4)

    original_kpt[..., :2] = smoothed_coords

    return original_kpt
