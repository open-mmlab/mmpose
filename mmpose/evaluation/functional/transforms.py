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


def transform_keypoints(kpt_info: Union[dict, list], num_keypoints: int,
                        mapping: Union[List[Tuple[int, int]],
                                       List[Tuple[Tuple, int]]]):
    """Transforms anns and predictions of keypoints based on the mapping."""
    if len(mapping):
        source_index, target_index = zip(*mapping)
    else:
        source_index, target_index = [], []

    list_input = True
    if not isinstance(kpt_info, list):
        kpt_info = [kpt_info]
        list_input = False

    for each in kpt_info:
        if 'keypoints' in each:
            keypoints = np.array(each['keypoints'])
            c = keypoints.shape[-1]
            keypoints = keypoints.reshape(-1, c)
            new_keypoints = np.zeros((num_keypoints, c), dtype=keypoints.dtype)
            new_keypoints[target_index] = keypoints[source_index]
            each['keypoints'] = new_keypoints.reshape(-1).tolist()
        if 'num_keypoints' in each:
            each['num_keypoints'] = num_keypoints

    if not list_input:
        kpt_info = kpt_info[0]

    return kpt_info
