import os
import warnings
from collections import OrderedDict

import json_tricks as json
import numpy as np
from xtcocotools.coco import COCO

from mmpose.core.evaluation.top_down_eval import keypoint_pck_accuracy
from ...registry import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownJhmdbDataset(TopDownCocoDataset):
    """JhmdbDataset dataset for top-down pose estimation.

    `Towards understanding action recognition
     <https://openaccess.thecvf.com/content_iccv_2013/papers/
     Jhuang_Towards_Understanding_Action_2013_ICCV_paper.pdf>`__

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    sub-JHMDB keypoint indexes::
        0: "neck",
        1: "belly",
        2: "head",
        3: "right_shoulder",
        4: "left_shoulder",
        5: "right_hip",
        6: "left_hip",
        7: "right_elbow",
        8: "left_elbow",
        9: "right_knee",
        10: "left_knee",
        11: "right_wrist",
        12: "left_wrist",
        13: "right_ankle",
        14: "left_ankle"

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
        super(TopDownCocoDataset, self).__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        if 'image_thr' in data_cfg:
            warnings.warn(
                'image_thr is deprecated, '
                'please use det_bbox_thr instead', DeprecationWarning)
            self.det_bbox_thr = data_cfg['image_thr']
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.ann_info['flip_pairs'] = [[3, 4], [5, 6], [7, 8], [9, 10],
                                       [9, 10], [11, 12], [13, 14]]

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 7, 8, 11, 12)
        self.ann_info['lower_body_ids'] = (5, 6, 9, 10, 13, 14)

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.2, 1.2, 1.5, 1.5, 1.5,
                1.5
            ],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        # Adapted from COCO dataset
        self.sigmas = np.array([
            .25, 1.07, .25, .79, .79, 1.07, 1.07, .72, .72, .87, .87, .62, .62,
            .89, .89
        ]) / 10.0

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            (self._class_to_coco_ind[cls], self._class_to_ind[cls])
            for cls in self.classes[1:])
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'jhmdb'

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, img_id):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            img_id: coco image id
        Returns:
            dict: db entry
        """
        img_ann = self.coco.loadImgs(img_id)[0]
        width = img_ann['width']
        height = img_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            # JHMDB uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            x -= 1
            y -= 1
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        bbox_id = 0
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)

            # JHMDB uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            joints_3d[:, :2] = keypoints[:, :2] - 1
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'bbox': obj['clean_bbox'][:4],
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1,
                'bbox_id': f'{img_id}_{bbox_id:03}'
            })
            bbox_id = bbox_id + 1

        return rec

    def _write_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file, metrics, pck_thr=0.2):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE'.
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.

        Returns:
            List: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []
        threshold_bbox = []
        threshold_torso = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))

            if 'tPCK' in metrics:
                torso_thr = np.linalg.norm(item['joints_3d'][4, :2] -
                                           item['joints_3d'][5, :2])
                if torso_thr < 1:
                    torso_thr = np.linalg.norm(
                        np.array(pred['keypoints'])[4, :2] -
                        np.array(pred['keypoints'])[5, :2])
                    warnings.warn('Torso Size < 1.')
                threshold_torso.append(np.array([torso_thr, torso_thr]))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_torso = np.array(threshold_torso)

        if 'PCK' in metrics:
            pck_p, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                                  threshold_bbox)

            stats_names = [
                'Head PCK', 'Sho PCK', 'Elb PCK', 'Wri PCK', 'Hip PCK',
                'Knee PCK', 'Ank PCK', 'Mean PCK'
            ]

            stats = [
                pck_p[2], 0.5 * pck_p[3] + 0.5 * pck_p[4],
                0.5 * pck_p[7] + 0.5 * pck_p[8],
                0.5 * pck_p[11] + 0.5 * pck_p[12],
                0.5 * pck_p[5] + 0.5 * pck_p[6],
                0.5 * pck_p[9] + 0.5 * pck_p[10],
                0.5 * pck_p[13] + 0.5 * pck_p[14], pck
            ]

            info_str.extend(list(zip(stats_names, stats)))

        if 'tPCK' in metrics:
            pck_p, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                                  threshold_torso)

            stats_names = [
                'Head tPCK', 'Sho tPCK', 'Elb tPCK', 'Wri tPCK', 'Hip tPCK',
                'Knee tPCK', 'Ank tPCK', 'Mean tPCK'
            ]

            stats = [
                pck_p[2], 0.5 * pck_p[3] + 0.5 * pck_p[4],
                0.5 * pck_p[7] + 0.5 * pck_p[8],
                0.5 * pck_p[11] + 0.5 * pck_p[12],
                0.5 * pck_p[5] + 0.5 * pck_p[6],
                0.5 * pck_p[9] + 0.5 * pck_p[10],
                0.5 * pck_p[13] + 0.5 * pck_p[14], pck
            ]

            info_str.extend(list(zip(stats_names, stats)))

        return info_str

    def evaluate(self, outputs, res_folder, metric='PCK', **kwargs):
        """Evaluate onehand10k keypoint results. The pose prediction results
        will be saved in `${res_folder}/result_keypoints.json`.

        Note:
            batch_size: N
            num_keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            outputs (list(preds, boxes, image_path, output_heatmap))
                :preds (np.ndarray[N,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str])
                :output_heatmap (np.ndarray[N, K, H, W]): model outpus.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'tPCK'.
                PCK means normalized by the bounding boxes, while tPCK
                means normalized by the torso size.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'tPCK']
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

            # convert 0-based index to 1-based index,
            # and get the first two dimensions.
            preds[..., :2] += 1.0
            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]
                kpts.append({
                    'keypoints': preds[i],
                    'center': boxes[i][0:2],
                    'scale': boxes[i][2:4],
                    'area': boxes[i][4],
                    'score': boxes[i][5],
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
