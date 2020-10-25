import os
import os.path as osp
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from poseval import eval_helpers
from poseval.evaluateAP import evaluateAP
from xtcocotools.coco import COCO

from ....core.post_processing import oks_nms, soft_oks_nms
from ...registry import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownPoseTrack18Dataset(TopDownCocoDataset):
    """PoseTrack18 dataset for top-down pose estimation.

    `Posetrack: A benchmark for human pose estimation and tracking' CVPR'2018
    More details can be found in the `paper
    <https://arxiv.org/abs/1710.10000>`_ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    PoseTrack2018 keypoint indexes::
        0: 'nose',
        1: 'head_bottom',
        2: 'head_top',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

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
        self.image_thr = data_cfg['image_thr']

        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.bbox_thr = data_cfg['bbox_thr']

        self.ann_info['flip_pairs'] = [[3, 4], [5, 6], [7, 8], [9, 10],
                                       [11, 12], [13, 14], [15, 16]]

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.ann_info['lower_body_ids'] = (11, 12, 13, 14, 15, 16)

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
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
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'posetrack18'

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def evaluate(self, outputs, res_folder, metric='mAP', **kwargs):
        """Evaluate coco keypoint results. The pose prediction results will be
        saved in `${res_folder}/result_keypoints.json`.

        Note:
            num_keypoints: K

        Args:
            outputs (list(preds, boxes, image_path))
                :preds (np.ndarray[1,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str]): For example, [ '/', 'v','a', 'l',
                    '2', '0', '1', '7', '/', '0', '0', '0', '0', '0',
                    '0', '3', '9', '7', '1', '3', '3', '.', 'j', 'p', 'g']
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed. Defaults: 'mAP'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mAP']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        pred_folder = osp.join(res_folder, 'preds')
        os.makedirs(pred_folder, exist_ok=True)
        gt_folder = osp.join(
            osp.dirname(self.annotations_path),
            osp.splitext(self.annotations_path.split('_')[-1])[0])

        kpts = defaultdict(list)
        for preds, boxes, image_path in outputs:
            str_image_path = ''.join(image_path)
            image_id = self.name2id[str_image_path[len(self.img_prefix):]]

            kpts[image_id].append({
                'keypoints': preds[0],
                'center': boxes[0][0:2],
                'scale': boxes[0][2:4],
                'area': boxes[0][4],
                'score': boxes[0][5],
                'image_id': image_id,
            })

        # rescoring and oks nms
        num_joints = self.ann_info['num_joints']
        vis_thr = self.vis_thr
        oks_thr = self.oks_thr
        oks_nmsed_kpts = defaultdict(list)
        for image_id in kpts.keys():
            img_kpts = kpts[image_id]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > vis_thr:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            nms = soft_oks_nms if self.soft_nms else oks_nms
            keep = nms(list(img_kpts), oks_thr, sigmas=self.sigmas)

            if len(keep) == 0:
                oks_nmsed_kpts[image_id].append(img_kpts)
            else:
                oks_nmsed_kpts[image_id].append(
                    [img_kpts[_keep] for _keep in keep])

        self._write_posetrack18_keypoint_results(oks_nmsed_kpts, gt_folder,
                                                 pred_folder)

        info_str = self._do_python_keypoint_eval(gt_folder, pred_folder)
        name_value = OrderedDict(info_str)

        return name_value

    def _write_posetrack18_keypoint_results(self, keypoint_results, gt_folder,
                                            pred_folder):
        """Write results into a json file.

        Args:
            keypoint_results (dict): keypoint results organized by image_id.
            gt_folder (str): Path of directory for official gt files.
            pred_folder (str): Path of directory to save the results.
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

        json_files = [
            pos for pos in os.listdir(gt_folder) if pos.endswith('.json')
        ]
        for json_file in json_files:

            with open(osp.join(gt_folder, json_file), 'r') as f:
                gt = json.load(f)

            annotations = []
            images = []

            for image in gt['images']:
                im = {}
                im['id'] = image['id']
                im['file_name'] = image['file_name']
                images.append(im)

                img_kpts = keypoint_results[im['id']]

                if len(img_kpts) == 0:
                    continue
                for track_id, img_kpt in enumerate(img_kpts):
                    ann = {}
                    ann['image_id'] = img_kpt['image_id']
                    ann['keypoints'] = np.array(
                        img_kpt['keypoints']).reshape(-1).tolist()
                    ann['scores'] = np.array(ann['keypoints']).reshape(
                        [-1, 3])[:, 2].tolist()
                    ann['score'] = float(img_kpt['score'])
                    ann['track_id'] = track_id
                    annotations.append(ann)

            info = {}
            info['images'] = images
            info['categories'] = categories
            info['annotations'] = annotations

            with open(osp.join(pred_folder, json_file), 'w') as f:
                json.dump(info, f, sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, gt_folder, pred_folder):
        """Keypoint evaluation using poseval."""

        argv = ['', gt_folder, pred_folder]

        print('Loading data')
        gtFramesAll, prFramesAll = eval_helpers.load_data_dir(argv)

        print('# gt frames  :', len(gtFramesAll))
        print('# pred frames:', len(prFramesAll))

        # evaluate per-frame multi-person pose estimation (AP)
        # compute AP
        print('Evaluation of per-frame multi-person pose estimation')
        apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll, None,
                                           False, False)

        # print AP
        print('Average Precision (AP) metric:')
        eval_helpers.printTable(apAll)

        stats = eval_helpers.getCum(apAll)

        stats_names = [
            'Head AP', 'Shou AP', 'Elb AP', 'Wri AP', 'Hip AP', 'Knee AP',
            'Ankl AP', 'Total AP'
        ]

        info_str = list(zip(stats_names, stats))

        return info_str
