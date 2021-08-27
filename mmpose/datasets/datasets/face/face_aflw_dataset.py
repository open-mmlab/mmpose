# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict

import json_tricks as json
import numpy as np

from mmpose.core.evaluation.top_down_eval import keypoint_nme
from mmpose.datasets.builder import DATASETS
from .face_base_dataset import FaceBaseDataset


@DATASETS.register_module()
class FaceAFLWDataset(FaceBaseDataset):
    """Face AFLW dataset for top-down face keypoint localization.

    `Annotated Facial Landmarks in the Wild: A Large-scale,
    Real-world Database for Facial Landmark Localization.
    In Proc. First IEEE International Workshop on Benchmarking
    Facial Image Analysis Technologies, 2011`

    The dataset loads raw images and apply specified transforms
    to return a dict containing the image tensors and other information.

    The landmark annotations follow the 19 points mark-up. The definition
    can be found in `https://www.tugraz.at/institute/icg/research`
    `/team-bischof/lrs/downloads/aflw/`

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.ann_info['use_different_joint_weights'] = False
        assert self.ann_info['num_joints'] == 19
        self.ann_info['joint_weights'] = \
            np.ones((self.ann_info['num_joints'], 1), dtype=np.float32)

        self.ann_info['flip_pairs'] = [[0, 5], [1, 4], [2, 3], [6, 11],
                                       [7, 10], [8, 9], [12, 14], [15, 17]]

        self.dataset_name = 'aflw'
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        for img_id in self.img_ids:

            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            objs = self.coco.loadAnns(ann_ids)

            for obj in objs:
                if self.test_mode:
                    # 'box_size' is used as normalization factor
                    assert 'box_size' in obj
                if max(obj['keypoints']) == 0:
                    continue
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                if 'center' in obj and 'scale' in obj:
                    center = np.array(obj['center'])
                    scale = np.array([obj['scale'], obj['scale']]) * 1.25
                else:
                    center, scale = self._xywh2cs(*obj['bbox'][:4], 1.25)

                image_file = os.path.join(self.img_prefix,
                                          self.id2name[img_id])

                gt_db.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'box_size': obj['box_size'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    def _get_normalize_factor(self, box_sizes):
        """Get normalize factor for evaluation.

        Args:
            box_sizes (np.ndarray[N, 1]): box size

        Return:
            np.ndarray[N, 2]: normalized factor
        """

        return np.tile(box_sizes, [1, 2])

    def _report_metric(self, res_file, metrics):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []
        box_sizes = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            box_sizes.append(item['box_size'])

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        box_sizes = np.array(box_sizes).reshape([-1, 1])

        if 'NME' in metrics:
            normalize_factor = self._get_normalize_factor(box_sizes)
            info_str.append(
                ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

        return info_str

    def evaluate(self, outputs, res_folder, metric='NME', **kwargs):
        """Evaluate freihand keypoint results. The pose prediction results will
        be saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[1,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str]): For example, ['3', '0', '0', 'W', '/',
                    'i', 'b', 'u', 'g', '/', 'i', 'm', 'a', 'g', 'e', '_', '0',
                    '1', '8', '.', 'j', 'p', 'g']
                :output_heatmap (np.ndarray[N, K, H, W]): model outpus.
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['NME']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = []
        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value
