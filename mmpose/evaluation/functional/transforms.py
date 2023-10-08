# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np


def transform_sigmas(sigmas: Union[List, np.ndarray], num_keypoints: int,
                     mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                      int]]]):
    """Transforms the sigmas based on the mapping."""
    if len(mapping):
        source_index, target_index = zip(*mapping)
    else:
        source_index, target_index = [], []

    list_input = False
    if isinstance(sigmas, list):
        sigmas = np.array(sigmas)
        list_input = True

    new_sigmas = np.ones(num_keypoints, dtype=sigmas.dtype)
    new_sigmas[target_index] = sigmas[source_index]

    if list_input:
        new_sigmas = new_sigmas.tolist()

    return new_sigmas


def transform_ann(ann_info: Union[dict, list], num_keypoints: int,
                  mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                   int]]]):
    """Transforms the annotations based on the mapping."""
    if len(mapping):
        source_index, target_index = zip(*mapping)
    else:
        source_index, target_index = [], []

    list_input = True
    if not isinstance(ann_info, list):
        ann_info = [ann_info]
        list_input = False

    for ann in ann_info:
        if 'keypoints' in ann:
            keypoints = np.array(ann['keypoints'])
            c = keypoints.shape[-1]
            keypoints = keypoints.reshape(-1, c)
            new_keypoints = np.zeros((num_keypoints, c), dtype=keypoints.dtype)
            new_keypoints[target_index] = keypoints[source_index]
            ann['keypoints'] = new_keypoints.reshape(-1).tolist()
        if 'num_keypoints' in ann:
            ann['num_keypoints'] = num_keypoints

    if not list_input:
        ann_info = ann_info[0]

    return ann_info
