# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from mmcv import Config
from xtcocotools.cocoeval import COCOeval

from ...builder import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownHalpeDataset(TopDownCocoDataset):
    """HalpeDataset for top-down pose estimation.

    'https://github.com/Fang-Haoshu/Halpe-FullBody'

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    In total, we have 136 keypoints for wholebody pose estimation.

    Halpe keypoint indexes::
        0-19: 20 body keypoints
        20-25: 6 foot keypoints
        26-93: 68 face keypoints
        94-135: 42 hand keypoints

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/halpe.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super(TopDownCocoDataset, self).__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.ann_info['use_different_joint_weights'] = False

        self.body_num = 20
        self.foot_num = 6
        self.face_num = 68
        self.left_hand_num = 21
        self.right_hand_num = 21

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get coco keypoint results."""
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            key_points = _key_points.reshape(-1,
                                             self.ann_info['num_joints'] * 3)

            cuts = np.cumsum([
                0, self.body_num, self.foot_num, self.face_num,
                self.left_hand_num, self.right_hand_num
            ]) * 3

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': key_point[cuts[0]:cuts[1]].tolist(),
                'foot_kpts': key_point[cuts[1]:cuts[2]].tolist(),
                'face_kpts': key_point[cuts[2]:cuts[3]].tolist(),
                'lefthand_kpts': key_point[cuts[3]:cuts[4]].tolist(),
                'righthand_kpts': key_point[cuts[4]:cuts[5]].tolist(),
                'score': float(img_kpt['score']),
                'center': img_kpt['center'].tolist(),
                'scale': img_kpt['scale'].tolist()
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num, self.left_hand_num,
            self.right_hand_num
        ])

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_body',
            self.sigmas[cuts[0]:cuts[1]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_foot',
            self.sigmas[cuts[1]:cuts[2]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_face',
            self.sigmas[cuts[2]:cuts[3]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_lefthand',
            self.sigmas[cuts[3]:cuts[4]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_righthand',
            self.sigmas[cuts[4]:cuts[5]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_wholebody',
            self.sigmas,
            use_area=True)
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
