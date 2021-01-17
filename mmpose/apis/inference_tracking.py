import warnings

import numpy as np


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
    """Get track id using IoU tracking greedily.

    Args:
        res (dict): The bbox & pose results of the person instance.
        results_last (list[dict]): The bbox & pose & track_id info of the
                last frame (bbox_result, pose_result, track_id).
        thr (float): The threshold for iou tracking.

    Returns:
        int: The track id for the new person instance.
        list[dict]: The bbox & pose & track_id info of the persons
                that have not been matched on the last frame.
    """

    bbox = list(res['bbox'])

    max_iou_score = -1
    max_index = -1

    for index, res_last in enumerate(results_last):
        bbox_last = list(res_last['bbox'])

        iou_score = _compute_iou(bbox, bbox_last)
        if iou_score > max_iou_score:
            max_iou_score = iou_score
            max_index = index

    if max_iou_score > thr:
        track_id = results_last[max_index]['track_id']
        del results_last[max_index]
    else:
        track_id = -1

    return track_id, results_last


def get_track_id(results, results_last, next_id, iou_thr=0.3):
    """Get track id for each person instance on the current frame.

    Args:
        results (list[dict]): The bbox & pose results of the current frame
                (bbox_result, pose_result).
        results_last (list[dict]): The bbox & pose & track_id info of the
                last frame (bbox_result, pose_result, track_id).
        next_id (int): The track id for the new person instance.
        iou_thr (float): The threshold for iou tracking.

    Returns:
        list[dict]: The bbox & pose & track_id info of the
                current frame (bbox_result, pose_result, track_id).
        int: The track id for the new person instance.
    """

    for result in results:
        track_id, results_last = _track_by_iou(result, results_last, iou_thr)

        if track_id == -1:
            result['track_id'] = next_id
            next_id += 1
        else:
            result['track_id'] = track_id

    return results, next_id


def vis_pose_tracking_result(model,
                             img,
                             result,
                             kpt_score_thr=0.3,
                             dataset='TopDownCocoDataset',
                             show=False,
                             out_file=None):
    """Visualize the pose tracking results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    radius = 4

    if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                   'TopDownOCHumanDataset'):
        kpt_num = 17
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    elif dataset == 'TopDownCocoWholeBodyDataset':
        kpt_num = 133
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [16, 18],
                    [16, 19], [16, 20], [17, 21], [17, 22], [17, 23], [92, 93],
                    [93, 94], [94, 95], [95, 96], [92, 97], [97, 98], [98, 99],
                    [99, 100], [92, 101], [101, 102], [102, 103], [103, 104],
                    [92, 105], [105, 106], [106, 107], [107, 108], [92, 109],
                    [109, 110], [110, 111], [111, 112], [113, 114], [114, 115],
                    [115, 116], [116, 117], [113, 118], [118, 119], [119, 120],
                    [120, 121], [113, 122], [122, 123], [123, 124], [124, 125],
                    [113, 126], [126, 127], [127, 128], [128, 129], [113, 130],
                    [130, 131], [131, 132], [132, 133]]
        radius = 1

    elif dataset == 'TopDownAicDataset':
        kpt_num = 14
        skeleton = [[3, 2], [2, 1], [1, 14], [14, 4], [4, 5], [5, 6], [9, 8],
                    [8, 7], [7, 10], [10, 11], [11, 12], [13, 14], [1, 7],
                    [4, 10]]

    elif dataset == 'TopDownMpiiDataset':
        kpt_num = 16
        skeleton = [[1, 2], [2, 3], [3, 7], [7, 4], [4, 5], [5, 6], [7, 8],
                    [8, 9], [9, 10], [9, 13], [13, 12], [12, 11], [9, 14],
                    [14, 15], [15, 16]]

    elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                     'PanopticDataset'):
        kpt_num = 21
        skeleton = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8],
                    [8, 9], [1, 10], [10, 11], [11, 12], [12, 13], [1, 14],
                    [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20],
                    [20, 21]]

    elif dataset == 'InterHand2DDataset':
        kpt_num = 21
        skeleton = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10],
                    [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18],
                    [18, 19], [19, 20], [4, 21], [8, 21], [12, 21], [16, 21],
                    [20, 21]]

    else:
        raise NotImplementedError()

    for res in result:
        track_id = res['track_id']
        bbox_color = palette[track_id % len(palette)]
        pose_kpt_color = palette[[track_id % len(palette)] * kpt_num]
        pose_limb_color = palette[[track_id % len(palette)] * len(skeleton)]
        img = model.show_result(
            img, [res],
            skeleton,
            radius=radius,
            pose_kpt_color=pose_kpt_color,
            pose_limb_color=pose_limb_color,
            bbox_color=tuple(bbox_color.tolist()),
            kpt_score_thr=kpt_score_thr,
            show=show,
            out_file=out_file)

    return img
