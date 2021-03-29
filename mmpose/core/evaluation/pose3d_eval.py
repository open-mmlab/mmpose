import numpy as np

from .mesh_eval import compute_similarity_transform


def keypoint_mpjpe(pred, gt, mask):
    """Calculate the mean per-joint position error (MPJPE) and the error after
    rigid alignment with the ground truth (P-MPJPE).

    batch_size: N
    num_keypoints: K
    keypoint_dims: C

    Args:
        pred (np.ndarray[N, K, C]): Predicted keypoint location.
        gt (np.ndarray[N, K, C]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
    Returns:
        tuple: A tuple containing joint position errors

        - mpjpe (float|np.ndarray[N]): mean per-joint position error.
        - p-mpjpe (float|np.ndarray[N]): mpjpe after rigid alignment with the
            ground truth
    """
    assert mask.any()

    pred_aligned = np.stack(
        compute_similarity_transform(pred_i, gt_i)
        for pred_i, gt_i in zip(pred, gt))

    mpjpe = np.linalg.norm(pred - gt, ord=2, axis=-1)[mask].mean()
    p_mpjpe = np.linalg.norm(pred_aligned - gt, ord=2, axis=-1)[mask].mean()

    return mpjpe, p_mpjpe
