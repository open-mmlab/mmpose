# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np


def transform_sigmas(sigmas: Union[List, np.ndarray], num_keypoints: int,
                     mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                      int]]]):
    """Transforms the sigmas based on the mapping."""
    if len(mapping):
        source_index, target_index = map(list, zip(*mapping))
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
    """Transforms COCO-format annotations based on the mapping."""
    if len(mapping):
        source_index, target_index = map(list, zip(*mapping))
    else:
        source_index, target_index = [], []

    list_input = True
    if not isinstance(ann_info, list):
        ann_info = [ann_info]
        list_input = False

    for each in ann_info:
        if 'keypoints' in each:
            keypoints = np.array(each['keypoints'])

            C = 3  # COCO-format: x, y, score
            keypoints = keypoints.reshape(-1, C)
            new_keypoints = np.zeros((num_keypoints, C), dtype=keypoints.dtype)
            new_keypoints[target_index] = keypoints[source_index]
            each['keypoints'] = new_keypoints.reshape(-1).tolist()

        if 'num_keypoints' in each:
            each['num_keypoints'] = num_keypoints

    if not list_input:
        ann_info = ann_info[0]

    return ann_info


def transform_pred(pred_info: Union[dict, list], num_keypoints: int,
                   mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                    int]]]):
    """Transforms predictions based on the mapping."""
    if len(mapping):
        source_index, target_index = map(list, zip(*mapping))
    else:
        source_index, target_index = [], []

    list_input = True
    if not isinstance(pred_info, list):
        pred_info = [pred_info]
        list_input = False

    for each in pred_info:
        if 'keypoints' in each:
            keypoints = np.array(each['keypoints'])

            N, _, C = keypoints.shape
            new_keypoints = np.zeros((N, num_keypoints, C),
                                     dtype=keypoints.dtype)
            new_keypoints[:, target_index] = keypoints[:, source_index]
            each['keypoints'] = new_keypoints

            keypoint_scores = np.array(each['keypoint_scores'])
            new_scores = np.zeros((N, num_keypoints),
                                  dtype=keypoint_scores.dtype)
            new_scores[:, target_index] = keypoint_scores[:, source_index]
            each['keypoint_scores'] = new_scores

        if 'num_keypoints' in each:
            each['num_keypoints'] = num_keypoints

    if not list_input:
        pred_info = pred_info[0]

    return pred_info
