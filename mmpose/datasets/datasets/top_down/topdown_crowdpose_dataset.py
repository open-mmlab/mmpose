import os
from collections import OrderedDict, defaultdict

import json_tricks as json
import numpy as np
from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval

from ....core.post_processing import candidate_reselect, convert_crowd
from ...registry import DATASETS
from .topdown_base_dataset import TopDownBaseDataset


@DATASETS.register_module()
class TopDownCrowdPoseDataset(TopDownBaseDataset):
    """CrowdPoseDataset dataset for top-down pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    CrowdPose keypoint indexes::

        0: 'left_shoulder',
        1: 'right_shoulder',
        2: 'left_elbow',
        3: 'right_elbow',
        4: 'left_wrist',
        5: 'right_wrist',
        6: 'left_hip',
        7: 'right_hip',
        8: 'left_knee',
        9: 'right_knee',
        10: 'left_ankle',
        11: 'right_ankle',
        12: 'top_head',
        13: 'neck',

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

        self.use_gt_bbox = data_cfg['use_gt_bbox']
        self.bbox_file = data_cfg['bbox_file']
        self.image_thr = data_cfg['image_thr']

        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']
        self.bbox_thr = data_cfg['bbox_thr']

        self.ann_info['flip_pairs'] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
                                       [10, 11]]

        self.ann_info['joint_to_joint'] = {}
        for pair in self.ann_info['flip_pairs']:
            self.ann_info['joint_to_joint'][pair[0]] = pair[1]
            self.ann_info['joint_to_joint'][pair[1]] = pair[0]

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 12, 13)
        self.ann_info['lower_body_ids'] = (6, 7, 8, 9, 10, 11)

        self.ann_info['use_different_joints_weight'] = False
        self.ann_info['joints_weight'] = np.array(
            [
                0.2, 0.2, 0.2, 1.3, 1.5, 0.2, 1.3, 1.5, 0.2, 0.2, 0.5, 0.2,
                0.2, 0.5
            ],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.coco = COCO(ann_file)

        cats = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        if (not self.test_mode) or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """Ground truth bbox and keypoints."""
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """load annotation from CrowdPoseAPI.

        Note:
            bbox:[x1, y1, w, h]
            interference_joints_3d(np.ndarray[Nx3]): the joints in someones
                bbox but not belong him.
        Args:
            index: crowdpose image id
        Returns:
            db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            if max(obj['keypoints']) == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float)
            for ipt in range(num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_visible[ipt, 0] = t_vis
                joints_3d_visible[ipt, 1] = t_vis
                joints_3d_visible[ipt, 2] = 0

            interference_joints_3d = np.array(obj['keypoints']).reshape(
                -1, 3)[num_joints:, :]

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image_file': self._image_path_from_index(index),
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'interference_joints': interference_joints_3d,
                'dataset': 'crowdpose',
                'bbox_score': 1
            })

        return rec

    def _box2cs(self, box):
        """Get box center & scale given box (x, y, w, h)."""
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            center (np.ndarray[float32](2,)): center of the bbox (x, y).
            scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if (not self.test_mode) and np.random.rand() < 0.3:
            center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

        scale = scale * 1.25

        return center, scale

    def _image_path_from_index(self, index):
        """ example: images/119993.jpg """
        image_path = os.path.join(self.img_prefix, '%06d.jpg' % index)
        return image_path

    def _load_coco_person_detection_results(self):
        """Load crowdpose person detection results."""
        num_joints = self.ann_info['num_joints']
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            raise ValueError('=> Load %s fail!' % self.bbox_file)

        print(f'=> Total boxes: {len(all_boxes)}')

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue

            img_name = self._image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thr:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((num_joints, 3), dtype=np.float)
            joints_3d_visible = np.ones((num_joints, 3), dtype=np.float)
            interference_joints_3d = np.zeros((0, 3), dtype=np.float)
            kpt_db.append({
                'image_file': img_name,
                'center': center,
                'scale': scale,
                'bbox_score': score,
                'dataset': 'crowdpose',
                'rotation': 0,
                'imgnum': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'interference_joints': interference_joints_3d,
            })
        print(f'=> Total boxes after fliter '
              f'low score@{self.image_thr}: {num_boxes}')
        return kpt_db

    def evaluate(self, outputs, res_folder, metric='mAP', **kwargs):
        """Evaluate crowdpose keypoint results. The pose prediction results
        will be saved in `${res_folder}/result_keypoints.json`.

        Note:
            num_keypoints: K

        Args:
            outputs (list(preds, boxes, image_path))
                :preds (np.ndarray[1,K,5,3]): The first dimensions is the
                    number of candidates, second third dimensions are
                    coordinates, score is the fourth dimension of the array.
                :boxes (np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str]): For example, [ '/', 'i','m', 'a',
                    'g', 'e', 's', '/', '0', '0', '0', '1', '3', '3', '.',
                    'j', 'p', 'g']
            res_folder (str): Path of directory to save the results.
            metric (str): Metric to be performed. Defaults: 'mAP'.

        Returns:
            name_value (dict): Evaluation results for evaluation metric.
        """
        assert metric == 'mAP'

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = defaultdict(list)
        for preds, boxes, image_path in outputs:
            str_image_path = ''.join(image_path)
            image_id = int(str_image_path[-10:-4])

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

        final_result = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            boxes, preds, box_scores = convert_crowd(img_kpts)
            result = candidate_reselect(boxes, preds, num_joints, img,
                                        box_scores, vis_thr)
            final_result.append(result)

        self._write_coco_keypoint_results(final_result, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)

        return name_value

    def _write_coco_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""
        data_pack = [{
            'cat_id': self._class_to_coco_ind[cls],
            'cls_ind': cls_ind,
            'cls': cls,
            'ann_type': 'keypoints',
            'keypoints': keypoints
        } for cls_ind, cls in enumerate(self.classes)
                     if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])

        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        """Get crowdpose keypoint results."""
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

            result = [{
                'image_id': img_kpt['image_id'],
                'category_id': cat_id,
                'keypoints': list(key_point),
                'score': img_kpt['score'],
                'center': list(img_kpt['center']),
                'scale': list(img_kpt['scale'])
            } for img_kpt, key_point in zip(img_kpts, key_points)]

            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using CrowdPoseAPI."""
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AP(E)', 'AP(M)', 'AP(H)'
        ]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
