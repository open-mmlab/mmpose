# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple

import numpy as np

from mmpose.codecs.utils import get_heatmap_maximum, get_simcc_maximum
from .mesh_eval import compute_similarity_transform


def _calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray,
                    norm_factor: np.ndarray) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((N, K), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(
        ((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float:
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                          thr: np.ndarray, norm_factor: np.ndarray) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


def keypoint_auc(pred: np.ndarray,
                 gt: np.ndarray,
                 mask: np.ndarray,
                 norm_factor: np.ndarray,
                 num_thrs: int = 20) -> float:
    """Calculate the Area under curve (AUC) of keypoint PCK accuracy.

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (float): Normalization factor.
        num_thrs (int): number of thresholds to calculate auc.

    Returns:
        float: Area under curve (AUC) of keypoint PCK accuracy.
    """
    nor = np.tile(np.array([[norm_factor, norm_factor]]), (pred.shape[0], 1))
    thrs = [1.0 * i / num_thrs for i in range(num_thrs)]
    avg_accs = []
    for thr in thrs:
        _, avg_acc, _ = keypoint_pck_accuracy(pred, gt, mask, thr, nor)
        avg_accs.append(avg_acc)

    auc = 0
    for i in range(num_thrs):
        auc += 1.0 / num_thrs * avg_accs[i]
    return auc


def keypoint_nme(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                 normalize_factor: np.ndarray) -> float:
    """Calculate the normalized mean error (NME).

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize_factor (np.ndarray[N, 2]): Normalization factor.

    Returns:
        float: normalized mean error
    """
    distances = _calc_distances(pred, gt, mask, normalize_factor)
    distance_valid = distances[distances != -1]
    return distance_valid.sum() / max(1, len(distance_valid))


def keypoint_epe(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """Calculate the end-point error.

    Note:
        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.

    Returns:
        float: Average end-point error.
    """

    distances = _calc_distances(
        pred, gt, mask,
        np.ones((pred.shape[0], pred.shape[2]), dtype=np.float32))
    distance_valid = distances[distances != -1]
    return distance_valid.sum() / max(1, len(distance_valid))


def pose_pck_accuracy(output: np.ndarray,
                      target: np.ndarray,
                      mask: np.ndarray,
                      thr: float = 0.05,
                      normalize: Optional[np.ndarray] = None) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from heatmaps.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        output (np.ndarray[N, K, H, W]): Model output heatmaps.
        target (np.ndarray[N, K, H, W]): Groundtruth heatmaps.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    N, K, H, W = output.shape
    if K == 0:
        return None, 0, 0
    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred, _ = get_heatmap_maximum(output)
    gt, _ = get_heatmap_maximum(target)
    return keypoint_pck_accuracy(pred, gt, mask, thr, normalize)


def simcc_pck_accuracy(output: Tuple[np.ndarray, np.ndarray],
                       target: Tuple[np.ndarray, np.ndarray],
                       simcc_split_ratio: float,
                       mask: np.ndarray,
                       thr: float = 0.05,
                       normalize: Optional[np.ndarray] = None) -> tuple:
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints from SimCC.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        output (Tuple[np.ndarray, np.ndarray]): Model predicted SimCC.
        target (Tuple[np.ndarray, np.ndarray]): Groundtruth SimCC.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation. Default 0.05.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - np.ndarray[K]: Accuracy of each keypoint.
        - float: Averaged accuracy across all keypoints.
        - int: Number of valid keypoints.
    """
    pred_x, pred_y = output
    gt_x, gt_y = target

    N, _, Wx = pred_x.shape
    _, _, Wy = pred_y.shape
    W, H = int(Wx / simcc_split_ratio), int(Wy / simcc_split_ratio)

    if normalize is None:
        normalize = np.tile(np.array([[H, W]]), (N, 1))

    pred_coords, _ = get_simcc_maximum(pred_x, pred_y)
    pred_coords /= simcc_split_ratio
    gt_coords, _ = get_simcc_maximum(gt_x, gt_y)
    gt_coords /= simcc_split_ratio

    return keypoint_pck_accuracy(pred_coords, gt_coords, mask, thr, normalize)


def multilabel_classification_accuracy(pred: np.ndarray,
                                       gt: np.ndarray,
                                       mask: np.ndarray,
                                       thr: float = 0.5) -> float:
    """Get multi-label classification accuracy.

    Note:
        - batch size: N
        - label number: L

    Args:
        pred (np.ndarray[N, L, 2]): model predicted labels.
        gt (np.ndarray[N, L, 2]): ground-truth labels.
        mask (np.ndarray[N, 1] or np.ndarray[N, L] ): reliability of
            ground-truth labels.
        thr (float): Threshold for calculating accuracy.

    Returns:
        float: multi-label classification accuracy.
    """
    # we only compute accuracy on the samples with ground-truth of all labels.
    valid = (mask > 0).min(axis=1) if mask.ndim == 2 else (mask > 0)
    pred, gt = pred[valid], gt[valid]

    if pred.shape[0] == 0:
        acc = 0.0  # when no sample is with gt labels, set acc to 0.
    else:
        # The classification of a sample is regarded as correct
        # only if it's correct for all labels.
        acc = (((pred - thr) * (gt - thr)) > 0).all(axis=1).mean()
    return acc


def keypoint_mpjpe(pred: np.ndarray,
                   gt: np.ndarray,
                   mask: np.ndarray,
                   alignment: str = 'none'):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (P-MPJPE).

    Note:
        - batch_size: N
        - num_keypoints: K
        - keypoint_dims: C

    Args:
        pred (np.ndarray): Predicted keypoint location with shape [N, K, C].
        gt (np.ndarray): Groundtruth keypoint location with shape [N, K, C].
        mask (np.ndarray): Visibility of the target with shape [N, K].
            False for invisible joints, and True for visible.
            Invisible joints will be ignored for accuracy calculation.
        alignment (str, optional): method to align the prediction with the
            groundtruth. Supported options are:

                - ``'none'``: no alignment will be applied
                - ``'scale'``: align in the least-square sense in scale
                - ``'procrustes'``: align in the least-square sense in
                    scale, rotation and translation.

    Returns:
        tuple: A tuple containing joint position errors

        - (float | np.ndarray): mean per-joint position error (mpjpe).
        - (float | np.ndarray): mpjpe after rigid alignment with the
            ground truth (p-mpjpe).
    """
    assert mask.any()

    if alignment == 'none':
        pass
    elif alignment == 'procrustes':
        pred = np.stack([
            compute_similarity_transform(pred_i, gt_i)
            for pred_i, gt_i in zip(pred, gt)
        ])
    elif alignment == 'scale':
        pred_dot_pred = np.einsum('nkc,nkc->n', pred, pred)
        pred_dot_gt = np.einsum('nkc,nkc->n', pred, gt)
        scale_factor = pred_dot_gt / pred_dot_pred
        pred = pred * scale_factor[:, None, None]
    else:
        raise ValueError(f'Invalid value for alignment: {alignment}')
    error = np.linalg.norm(pred - gt, ord=2, axis=-1)[mask].mean()

    return error
