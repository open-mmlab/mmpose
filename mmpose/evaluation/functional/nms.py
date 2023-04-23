# ------------------------------------------------------------------------------
# Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# and https://github.com/HRNet/DEKR
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

from typing import List, Optional

import numpy as np


def nms(dets: np.ndarray, thr: float) -> List[int]:
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets (np.ndarray): [[x1, y1, x2, y2, score]].
        thr (float): Retain overlap < thr.

    Returns:
        list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def oks_iou(g: np.ndarray,
            d: np.ndarray,
            a_g: float,
            a_d: np.ndarray,
            sigmas: Optional[np.ndarray] = None,
            vis_thr: Optional[float] = None) -> np.ndarray:
    """Calculate oks ious.

    Note:

        - number of keypoints: K
        - number of instances: N

    Args:
        g (np.ndarray): The instance to calculate OKS IOU with other
            instances. Containing the keypoints coordinates. Shape: (K*3, )
        d (np.ndarray): The rest instances. Containing the keypoints
            coordinates. Shape: (N, K*3)
        a_g (float): Area of the ground truth object.
        a_d (np.ndarray): Area of the detected object. Shape: (N, )
        sigmas (np.ndarray, optional): Keypoint labelling uncertainty.
            Please refer to `COCO keypoint evaluation
            <https://cocodataset.org/#keypoints-eval>`__ for more details.
            If not given, use the sigmas on COCO dataset.
            If specified, shape: (K, ). Defaults to ``None``
        vis_thr(float, optional): Threshold of the keypoint visibility.
            If specified, will calculate OKS based on those keypoints whose
            visibility higher than vis_thr. If not given, calculate the OKS
            based on all keypoints. Defaults to ``None``

    Returns:
        np.ndarray: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list((vg > vis_thr) & (vd > vis_thr))
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db: List[dict],
            thr: float,
            sigmas: Optional[np.ndarray] = None,
            vis_thr: Optional[float] = None,
            score_per_joint: bool = False):
    """OKS NMS implementations.

    Args:
        kpts_db (List[dict]): The keypoints results of the same image.
        thr (float): The threshold of NMS. Will retain oks overlap < thr.
        sigmas (np.ndarray, optional): Keypoint labelling uncertainty.
            Please refer to `COCO keypoint evaluation
            <https://cocodataset.org/#keypoints-eval>`__ for more details.
            If not given, use the sigmas on COCO dataset. Defaults to ``None``
        vis_thr(float, optional): Threshold of the keypoint visibility.
            If specified, will calculate OKS based on those keypoints whose
            visibility higher than vis_thr. If not given, calculate the OKS
            based on all keypoints. Defaults to ``None``
        score_per_joint(bool): Whether the input scores (in kpts_db) are
            per-joint scores. Defaults to ``False``

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def _rescore(overlap: np.ndarray,
             scores: np.ndarray,
             thr: float,
             type: str = 'gaussian'):
    """Rescoring mechanism gaussian or linear.

    Args:
        overlap (np.ndarray): The calculated oks ious.
        scores (np.ndarray): target scores.
        thr (float): retain oks overlap < thr.
        type (str): The rescoring type. Could be 'gaussian' or 'linear'.
            Defaults to ``'gaussian'``

    Returns:
        np.ndarray: indexes to keep
    """
    assert len(overlap) == len(scores)
    assert type in ['gaussian', 'linear']

    if type == 'linear':
        inds = np.where(overlap >= thr)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(-overlap**2 / thr)

    return scores


def soft_oks_nms(kpts_db: List[dict],
                 thr: float,
                 max_dets: int = 20,
                 sigmas: Optional[np.ndarray] = None,
                 vis_thr: Optional[float] = None,
                 score_per_joint: bool = False):
    """Soft OKS NMS implementations.

    Args:
        kpts_db (List[dict]): The keypoints results of the same image.
        thr (float): The threshold of NMS. Will retain oks overlap < thr.
        max_dets (int): Maximum number of detections to keep. Defaults to 20
        sigmas (np.ndarray, optional): Keypoint labelling uncertainty.
            Please refer to `COCO keypoint evaluation
            <https://cocodataset.org/#keypoints-eval>`__ for more details.
            If not given, use the sigmas on COCO dataset. Defaults to ``None``
        vis_thr(float, optional): Threshold of the keypoint visibility.
            If specified, will calculate OKS based on those keypoints whose
            visibility higher than vis_thr. If not given, calculate the OKS
            based on all keypoints. Defaults to ``None``
        score_per_joint(bool): Whether the input scores (in kpts_db) are
            per-joint scores. Defaults to ``False``

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while len(order) > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        order = order[1:]
        scores = _rescore(oks_ovr, scores[1:], thr)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep


def nearby_joints_nms(
    kpts_db: List[dict],
    dist_thr: float,
    num_nearby_joints_thr: Optional[int] = None,
    score_per_joint: bool = False,
    max_dets: int = 30,
):
    """Nearby joints NMS implementations. Instances with non-maximum scores
    will be suppressed if they have too much closed joints with other
    instances. This function is modified from project
    `DEKR<https://github.com/HRNet/DEKR/blob/main/lib/core/nms.py>`.

    Args:
        kpts_db (list[dict]): keypoints and scores.
        dist_thr (float): threshold for judging whether two joints are close.
        num_nearby_joints_thr (int): threshold for judging whether two
            instances are close.
        max_dets (int): max number of detections to keep.
        score_per_joint (bool): the input scores (in kpts_db) are per joint
            scores.

    Returns:
        np.ndarray: indexes to keep.
    """

    assert dist_thr > 0, '`dist_thr` must be greater than 0.'
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'] for k in kpts_db])

    num_people, num_joints, _ = kpts.shape
    if num_nearby_joints_thr is None:
        num_nearby_joints_thr = num_joints // 2
    assert num_nearby_joints_thr < num_joints, '`num_nearby_joints_thr` must '\
        'be less than the number of joints.'

    # compute distance threshold
    pose_area = kpts.max(axis=1) - kpts.min(axis=1)
    pose_area = np.sqrt(np.power(pose_area, 2).sum(axis=1))
    pose_area = pose_area.reshape(num_people, 1, 1)
    pose_area = np.tile(pose_area, (num_people, num_joints))
    close_dist_thr = pose_area * dist_thr

    # count nearby joints between instances
    instance_dist = kpts[:, None] - kpts
    instance_dist = np.sqrt(np.power(instance_dist, 2).sum(axis=3))
    close_instance_num = (instance_dist < close_dist_thr).sum(2)
    close_instance = close_instance_num > num_nearby_joints_thr

    # apply nms
    ignored_pose_inds, keep_pose_inds = set(), list()
    indexes = np.argsort(scores)[::-1]
    for i in indexes:
        if i in ignored_pose_inds:
            continue
        keep_inds = close_instance[i].nonzero()[0]
        keep_ind = keep_inds[np.argmax(scores[keep_inds])]
        if keep_ind not in ignored_pose_inds:
            keep_pose_inds.append(keep_ind)
            ignored_pose_inds = ignored_pose_inds.union(set(keep_inds))

    # limit the number of output instances
    if max_dets > 0 and len(keep_pose_inds) > max_dets:
        sub_inds = np.argsort(scores[keep_pose_inds])[-1:-max_dets - 1:-1]
        keep_pose_inds = [keep_pose_inds[i] for i in sub_inds]

    return keep_pose_inds
