import copy as cp
import os
from collections import OrderedDict

import json_tricks as json
import numpy as np

from mmpose.datasets.builder import DATASETS
from .topdown_base_dataset import TopDownBaseDataset


@DATASETS.register_module()
class TopDownOneHand10KDataset(TopDownBaseDataset):
    """OneHand10K dataset for top-down hand pose estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    OneHand10K keypoint indexes::

        0: 'wrist',
        1: 'thumb1',
        2: 'thumb2',
        3: 'thumb3',
        4: 'thumb4',
        5: 'forefinger1',
        6: 'forefinger2',
        7: 'forefinger3',
        8: 'forefinger4',
        9: 'middle_finger1',
        10: 'middle_finger2',
        11: 'middle_finger3',
        12: 'middle_finger4',
        13: 'ring_finger1',
        14: 'ring_finger2',
        15: 'ring_finger3',
        16: 'ring_finger4',
        17: 'pinky_finger1',
        18: 'pinky_finger2',
        19: 'pinky_finger3',
        20: 'pinky_finger4'

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

        self.ann_info['flip_pairs'] = []

        self.ann_info['use_different_joints_weight'] = False
        self.ann_info['joints_weight'] =  \
            np.ones(21, dtype=np.float32).reshape(
                (self.ann_info['num_joints'], 1))

        self.db = self._get_db(ann_file)
        self.image_set = set([x['image_file'] for x in self.db])
        self.num_images = len(self.image_set)

        print('=> num_images: {}'.format(self.num_images))
        print('=> load {} samples'.format(len(self.db)))

    def _get_db(self, ann_file):
        """Load dataset."""
        with open(ann_file, 'r') as f:
            data = json.load(f)
        tmpl = dict(
            image_file=None,
            center=None,
            scale=None,
            rotation=0,
            joints_3d=None,
            joints_3d_visible=None,
            dataset='OneHand10K')

        imid2info = {x['id']: x for x in data['images']}

        num_joints = self.ann_info['num_joints']
        gt_db = []

        for anno in data['annotations']:
            newitem = cp.deepcopy(tmpl)
            image_id = anno['image_id']
            newitem['image_file'] = os.path.join(
                self.img_prefix, imid2info[image_id]['file_name'])

            if max(anno['keypoints']) == 0:
                continue

            joints_3d = np.zeros((num_joints, 3), dtype=np.float)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float)

            for ipt in range(num_joints):
                joints_3d[ipt, 0] = anno['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = anno['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = anno['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_visible[ipt, 0] = t_vis
                joints_3d_visible[ipt, 1] = t_vis
                joints_3d_visible[ipt, 2] = 0

            center, scale = self._box2cs(anno['bbox'][:4])
            newitem['center'] = center
            newitem['scale'] = scale
            newitem['joints_3d'] = joints_3d
            newitem['joints_3d_visible'] = joints_3d_visible
            gt_db.append(newitem)

        return gt_db

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

    def evaluate_kernal(self, pred, joints_3d, joints_3d_visible, bbox):
        """Evaluate one example.

        ||pre[i] - joints_3d[i]|| < 0.2 * max(w, h)
        """
        num_joints = self.ann_info['num_joints']
        bbox = np.array(bbox)
        threshold = np.max(bbox[2:]) * 0.2
        hit = np.zeros(num_joints, dtype=np.float32)
        exist = np.zeros(num_joints, dtype=np.float32)

        for i in range(num_joints):
            pred_pt = pred[i]
            gt_pt = joints_3d[i]
            vis = joints_3d_visible[i][0]
            if vis:
                exist[i] = 1
            else:
                continue
            distance = np.linalg.norm(pred_pt[:2] - gt_pt[:2])
            if distance < threshold:
                hit[i] = 1
        return hit, exist

    def evaluate(self, outputs, res_folder, metrics='PCK', **kwargs):
        """Evaluate OneHand10K keypoint results."""
        res_file = os.path.join(res_folder, 'result_keypoints.json')

        all_preds, all_boxes, all_image_path = list(map(list, zip(*outputs)))

        kpts = []

        for idx, kpt in enumerate(all_preds):
            kpts.append({
                'keypoints':
                kpt[0],
                'center':
                all_boxes[idx][0][0:2],
                'scale':
                all_boxes[idx][0][2:4],
                'area':
                all_boxes[idx][0][4],
                'score':
                all_boxes[idx][0][5],
                'image_id':
                int(''.join(all_image_path[idx]).split('/')[-1][:-4]),
            })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file)
        name_value = OrderedDict(info_str)

        return name_value

    def _write_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file):
        """Keypoint evaluation.

        Report Mean Acc of skeleton, contour and all joints.
        """
        num_joints = self.ann_info['num_joints']
        hit = np.zeros(num_joints)
        exist = np.zeros(num_joints)

        with open(res_file, 'r') as fin:
            preds = json.load(fin)

        assert len(preds) == len(self.db)
        for pred, item in zip(preds, self.db):
            h, e = self.evaluate_kernal(pred['keypoints'], item['joints_3d'],
                                        item['joints_3d_visible'],
                                        item['headbox'])
            hit += h
            exist += e
        p_all = np.sum(hit) / np.sum(exist)

        info_str = []
        info_str.append(('p_acc', p_all))
        return info_str
