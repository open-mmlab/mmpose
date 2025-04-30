# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Dict, List, Optional

import numpy as np
from mmengine.fileio import dump, load
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from .coco_metric import CocoMetric

try:
    from poseval import eval_helpers
    from poseval.evaluateAP import evaluateAP
    has_poseval = True
except (ImportError, ModuleNotFoundError):
    has_poseval = False


@METRICS.register_module()
class PoseTrack18Metric(CocoMetric):
    """PoseTrack18 evaluation metric.

    Evaluate AP, and mAP for keypoint detection tasks.
    Support PoseTrack18 (video) dataset. Please refer to
    `<https://github.com/leonid-pishchulin/poseval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.

            Defaults to ``'bbox_keypoint'`
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``bbox_keypoint``. Defaults to ``0.2``
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

                - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
                    perform NMS.
                - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
                    to perform soft NMS.
                - ``'none'``: Do not perform NMS. Typically for bottomup mode
                    output.

            Defaults to ``'oks_nms'`
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to ``0.9``
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to ``False``
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Defaults to ``None``
        **kwargs: Keyword parameters passed to :class:`mmeval.BaseMetric`
    """
    default_prefix: Optional[str] = 'posetrack18'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        # raise an error to avoid long time running without getting results
        if not has_poseval:
            raise ImportError('Please install ``poseval`` package for '
                              'evaluation on PoseTrack dataset '
                              '(see `requirements/optional.txt`)')
        super().__init__(
            ann_file=ann_file,
            score_mode=score_mode,
            keypoint_score_thr=keypoint_score_thr,
            nms_mode=nms_mode,
            nms_thr=nms_thr,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            collect_device=collect_device,
            prefix=prefix)

    def results2json(self, keypoints: Dict[int, list],
                     outfile_prefix: str) -> str:
        """Dump the keypoint detection results into a json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json".

        Returns:
            str: The json file name of keypoint results.
        """
        categories = []

        cat = {}
        cat['supercategory'] = 'person'
        cat['id'] = 1
        cat['name'] = 'person'
        cat['keypoints'] = [
            'nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ]
        cat['skeleton'] = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                           [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
                           [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                           [4, 6], [5, 7]]
        categories.append(cat)

        # path of directory for official gt files
        gt_folder = osp.join(
            osp.dirname(self.ann_file),
            osp.splitext(self.ann_file.split('_')[-1])[0])
        # the json file for each video sequence
        json_files = [
            pos for pos in os.listdir(gt_folder) if pos.endswith('.json')
        ]

        for json_file in json_files:
            gt = load(osp.join(gt_folder, json_file))
            annotations = []
            images = []

            for image in gt['images']:
                img = {}
                img['id'] = image['id']
                img['file_name'] = image['file_name']
                images.append(img)

                img_kpts = keypoints[img['id']]

                for track_id, img_kpt in enumerate(img_kpts):
                    ann = {}
                    ann['image_id'] = img_kpt['img_id']
                    ann['keypoints'] = np.array(
                        img_kpt['keypoints']).reshape(-1).tolist()
                    ann['scores'] = np.array(ann['keypoints']).reshape(
                        [-1, 3])[:, 2].tolist()
                    ann['score'] = float(img_kpt['score'])
                    ann['track_id'] = track_id
                    annotations.append(ann)

            pred_file = osp.join(osp.dirname(outfile_prefix), json_file)
            info = {}
            info['images'] = images
            info['categories'] = categories
            info['annotations'] = annotations

            dump(info, pred_file, sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> List[tuple]:
        """Do keypoint evaluation using `poseval` package.

        Args:
            outfile_prefix (str): The filename prefix of the json files.
                If the prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json".

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # path of directory for official gt files
        # 'xxx/posetrack18_train.json' -> 'xxx/train/'
        gt_folder = osp.join(
            osp.dirname(self.ann_file),
            osp.splitext(self.ann_file.split('_')[-1])[0])
        pred_folder = osp.dirname(outfile_prefix)

        argv = ['', gt_folder + '/', pred_folder + '/']

        logger.info('Loading data')
        gtFramesAll, prFramesAll = eval_helpers.load_data_dir(argv)

        logger.info(f'# gt frames  : {len(gtFramesAll)}')
        logger.info(f'# pred frames: {len(prFramesAll)}')

        # evaluate per-frame multi-person pose estimation (AP)
        # compute AP
        logger.info('Evaluation of per-frame multi-person pose estimation')
        apAll, _, _ = evaluateAP(gtFramesAll, prFramesAll, None, False, False)

        # print AP
        logger.info('Average Precision (AP) metric:')
        eval_helpers.printTable(apAll)

        stats = eval_helpers.getCum(apAll)

        stats_names = [
            'Head AP', 'Shou AP', 'Elb AP', 'Wri AP', 'Hip AP', 'Knee AP',
            'Ankl AP', 'AP'
        ]

        info_str = list(zip(stats_names, stats))

        return info_str
