# Copyright (c) OpenMMLab. All rights reserved.
from .kpt_2d_sview_rgb_img_bottom_up_dataset import \
    Kpt2dSviewRgbImgBottomUpDataset
from .kpt_2d_sview_rgb_img_top_down_dataset import \
    Kpt2dSviewRgbImgTopDownDataset
from .kpt_3d_sview_kpt_2d_dataset import Kpt3dSviewKpt2dDataset
from .kpt_3d_sview_rgb_img_top_down_dataset import \
    Kpt3dSviewRgbImgTopDownDataset

__all__ = [
    'Kpt2dSviewRgbImgTopDownDataset', 'Kpt3dSviewRgbImgTopDownDataset',
    'Kpt2dSviewRgbImgBottomUpDataset', 'Kpt3dSviewKpt2dDataset'
]
