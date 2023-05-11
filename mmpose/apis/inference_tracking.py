# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np

from mmpose.evaluation.functional.nms import oks_iou


def _compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn('union_area=0 is unexpected')

    iou = inter_area / union_area

    return iou


def _track_by_iou(res, results_last, thr):
    """Get track id using IoU tracking greedily."""

    bbox = list(np.squeeze(res.pred_instances.bboxes, axis=0))

    max_iou_score = -1
    max_index = -1
    match_result = {}
    for index, res_last in enumerate(results_last):
        bbox_last = list(np.squeeze(res_last.pred_instances.bboxes, axis=0))

        iou_score = _compute_iou(bbox, bbox_last)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = index

    if max_iou_score > thr:
        track_id = results_last[max_index].track_id
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result


def _track_by_oks(res, results_last, thr, sigmas=None):
    """Get track id using OKS tracking greedily."""
    keypoint = np.concatenate((res.pred_instances.keypoints,
                               res.pred_instances.keypoint_scores[:, :, None]),
                              axis=2)
    keypoint = np.squeeze(keypoint, axis=0).reshape((-1))
    area = np.squeeze(res.pred_instances.areas, axis=0)
    max_index = -1
    match_result = {}

    if len(results_last) == 0:
        return -1, results_last, match_result

    keypoints_last = np.array([
        np.squeeze(
            np.concatenate(
                (res_last.pred_instances.keypoints,
                 res_last.pred_instances.keypoint_scores[:, :, None]),
                axis=2),
            axis=0).reshape((-1)) for res_last in results_last
    ])
    area_last = np.array([
        np.squeeze(res_last.pred_instances.areas, axis=0)
        for res_last in results_last
    ])

    oks_score = oks_iou(
        keypoint, keypoints_last, area, area_last, sigmas=sigmas)

    max_index = np.argmax(oks_score)

    if oks_score[max_index] > thr:
        track_id = results_last[max_index].track_id
        match_result = results_last[max_index]
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last, match_result
