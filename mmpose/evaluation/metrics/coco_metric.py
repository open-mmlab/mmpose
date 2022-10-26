# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import numpy as np
from mmeval import COCOPoseMetric as _COCOMetric

from mmpose.registry import METRICS


@METRICS.register_module()
class COCOMetric(_COCOMetric):
    """COCO pose estimation task evaluation metric.

    Evaluate AR, AP, and mAP for keypoint detection tasks. Support COCO
    dataset and other datasets in COCO format. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to ``True``
        iou_type (str): The same parameter as `iouType` in
            :class:`xtcocotools.COCOeval`, which can be ``'keypoints'``, or
            ``'keypoints_crowd'`` (used in CrowdPose dataset).
            Defaults to ``'keypoints'``
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.
                - ``'bbox_rle'``: Use rle_score to rescore the
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

    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(
            ann_file=ann_file,
            use_area=use_area,
            iou_type=iou_type,
            score_mode=score_mode,
            keypoint_score_thr=keypoint_score_thr,
            nms_mode=nms_mode,
            nms_thr=nms_thr,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            **kwargs)

    #  TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model, each of which has the following keys:

                - 'id': The id of the sample
                - 'img_id': The image_id of the sample
                - 'pred_instances': The prediction results of instance(s)
        """
        predictions, groundtruths = [], []
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample['pred_instances']['keypoints']
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            assert keypoint_scores.shape == keypoints.shape[:2]

            # parse prediction results
            pred = dict()
            pred['id'] = data_sample['id']
            pred['img_id'] = data_sample['img_id']
            pred['keypoints'] = keypoints
            pred['keypoint_scores'] = keypoint_scores
            pred['bbox_scores'] = data_sample['gt_instances']['bbox_scores']
            pred['category_id'] = data_sample.get('category_id', 1)

            # get area information
            if 'bbox_scales' in data_sample['gt_instances']:
                pred['areas'] = np.prod(
                    data_sample['gt_instances']['bbox_scales'], axis=1)
            predictions.append(pred)

            # parse gt
            gt = dict()
            if self.coco is None:
                gt['width'] = data_sample['ori_shape'][1]
                gt['height'] = data_sample['ori_shape'][0]
                gt['img_id'] = data_sample['img_id']
                if self.iou_type == 'keypoints_crowd':
                    assert 'crowd_index' in data_sample, \
                        '`crowd_index` is required when `self.iou_type` is ' \
                        '`keypoints_crowd`'
                    gt['crowd_index'] = data_sample['crowd_index']
                assert 'raw_ann_info' in data_sample, \
                    'The row ground truth annotations are required for ' \
                    'evaluation when `ann_file` is not provided'
                anns = data_sample['raw_ann_info']
                gt['raw_ann_info'] = anns if isinstance(anns, list) else [anns]

            groundtruths.append(gt)

        # add converted result to the results list
        self.add(predictions, groundtruths)

    def evaluate(self, *args, **kwargs) -> dict:
        """Returns metric results and print pretty table of metrics per class.

        This method would be invoked by ``mmengine.Evaluator``.
        """
        metric_results = self.compute(*args, **kwargs)
        self.reset()
        return metric_results
