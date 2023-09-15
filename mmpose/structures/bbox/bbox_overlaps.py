# Copyright (c) OpenMMLab. All rights reserved.
import torch


def fp16_clamp(x, min_val=None, max_val=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min_val, max_val).half()
    return x.clamp(min_val, max_val)


def bbox_overlaps(bboxes1,
                  bboxes2,
                  mode='iou',
                  is_aligned=False,
                  eps=1e-6) -> torch.Tensor:
    """Calculate overlap between two sets of bounding boxes.

    Args:
        bboxes1 (torch.Tensor): Bounding boxes of shape (..., m, 4) or empty.
        bboxes2 (torch.Tensor): Bounding boxes of shape (..., n, 4) or empty.
        mode (str): "iou" (intersection over union),
                    "iof" (intersection over foreground),
                    or "giou" (generalized intersection over union).
                    Defaults to "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A small constant added to the denominator for
            numerical stability. Default 1e-6.

    Returns:
        torch.Tensor: Overlap values of shape (..., m, n) if is_aligned is
            False, else shape (..., m).

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    if bboxes1.ndim == 1:
        bboxes1 = bboxes1.unsqueeze(0)
    if bboxes2.ndim == 1:
        bboxes2 = bboxes2.unsqueeze(0)

    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = fp16_clamp(rb - lt, min_val=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps_tensor = union.new_tensor([eps])
    union = torch.max(union, eps_tensor)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    elif mode == 'giou':
        enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min_val=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = torch.max(enclose_area, eps_tensor)
        gious = ious - (enclose_area - union) / enclose_area
        return gious
