# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS


@METRICS.register_module()
class CocoMetric(BaseMetric):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for keypoint detection tasks. Support COCO
    dataset and other datasets in COCO format. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str): Path to the coco format annotation file.
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Default: ``True``.
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Default: ``False``.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Default: ``None``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Default: ``'cpu'``.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Default: ``None``.
    """
    default_prefix: Optional[str] = 'coco'

    def __init__(self,
                 ann_file: str,
                 use_area: bool = True,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # initialize coco helper with the annotation json file
        self.coco = COCO(ann_file)

        self.use_area = use_area

        self.format_only = format_only
        if format_only:
            assert outfile_prefix is not None, 'Please set `outfile_prefix`' \
                'to specify the path to store the output results.'
        self.outfile_prefix = outfile_prefix

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data, pred in zip(data_batch, predictions):
            pred = pred['pred_instances']
            # keypoints.shape: [N, K, 3],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = pred['keypoints'].cpu().numpy()
            # [N, 1], the scores for all instances
            scores = pred['scores'].cpu().numpy()
            assert len(scores) == len(keypoints)

            result = dict()
            result['id'] = data['data_sample']['id']
            result['img_id'] = data['data_sample']['img_id']
            result['keypoints'] = keypoints
            result['scores'] = scores
            # add converted result to the results list
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        kpts = defaultdict(list)

        # group the results by img_id
        for result in results:
            img_id = result['img_id']
            for idx in range(len(result['scores'])):
                kpts[img_id].append({
                    'id': result['id'],
                    'img_id': result['img_id'],
                    'keypoints': result['keypoints'][idx],
                    'score': result['scores'][idx],
                })

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='id')

        # convert results to coco style and dump into a json file
        res_file = self.results2json(kpts, outfile_prefix=outfile_prefix)

        # only format the results without doing quantitative evaluation
        if self.format_only:
            return {}
        else:
            # do evaluation only if the ground truth annotations exist
            assert 'annotations' in self.coco.dataset, \
                'Ground truth annotations are required for evaluation.'

        # evaluation results
        eval_results = OrderedDict()
        logger.info(f'Evaluating {self.__class__.__name__}...')
        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def results2json(self, keypoints: Dict[int, list],
                     outfile_prefix: str) -> str:
        """Dump the keypoint detection results to a COCO style json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            str: The json file name of keypoint results.
        """
        # the results with category_id
        cat_id = 1
        cat_results = []

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            # use the keypoints of the first person in current image
            num_keypoints = len(img_kpts[0]['keypoints'])
            # collect all the person keypoints in current image
            keypoints = _keypoints.reshape(-1, num_keypoints * 3)

            result = [{
                'image_id': img_kpt['img_id'],
                'category_id': cat_id,
                'keypoints': keypoint.tolist(),
                'score': float(img_kpt['score']),
            } for img_kpt, keypoint in zip(img_kpts, keypoints)]

            cat_results.extend(result)

        res_file = f'{outfile_prefix}.keypoints.json'

        dump(cat_results, res_file, sort_keys=True, indent=4)

        return res_file

    def _do_python_keypoint_eval(self, res_file: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            res_file (str): The filename of the keypoint result json file.

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta['sigmas']
        coco_eval = COCOeval(self.coco, coco_det, 'keypoints', sigmas,
                             self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self,
                                kpts: Dict[int, list],
                                key: str = 'id') -> Dict[int, list]:
        """Sort keypoint detection results in each image and remove the
        duplicate ones. Usually performed in multi-batch testing.

        Args:
            kpts (Dict[int, list]): keypoint prediction results. The keys are
                '`img_id`' and the values are list that may contain
                keypoints of multiple persons. Each element in the list is a
                dict containing the ``'key'`` field.
                See the argument ``key`` for details.
            key (str): The key name in each person prediction results. The
                corresponding value will be used for sorting the results.
                Default: ``'id'``.

        Returns:
            Dict[int, list]: The sorted keypoint detection results.
        """
        for img_id, persons in kpts.items():
            # deal with bottomup-style output
            if isinstance(kpts[img_id][0][key], Sequence):
                return kpts
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
