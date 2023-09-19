# Copyright (c) OpenMMLab. All rights reserved.
from .coco_wholebody_hand_dataset import CocoWholeBodyHandDataset
from .freihand_dataset import FreiHandDataset
from .interhand_3d_dataset import InterHand3DDataset
from .onehand10k_dataset import OneHand10KDataset
from .panoptic_hand2d_dataset import PanopticHand2DDataset
from .rhd2d_dataset import Rhd2DDataset

__all__ = [
    'OneHand10KDataset', 'FreiHandDataset', 'PanopticHand2DDataset',
    'Rhd2DDataset', 'CocoWholeBodyHandDataset', 'InterHand3DDataset'
]
