import copy as cp
import os
import os.path as osp
from collections import OrderedDict

import json_tricks as json
import numpy as np

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_pck_accuracy)
from mmpose.datasets.builder import DATASETS
from .topdown_base_dataset import TopDownBaseDataset


@DATASETS.register_module()
class TopDownOneHand10KDataset(TopDownBaseDataset):
    """OneHand10K dataset for top-down hand pose estimation.

    `Mask-pose Cascaded CNN for 2D Hand Pose Estimation from
    Single Color Images' TCSVT'2019
    More details can be found in the `paper
    <https://www.yangangwang.com/papers/WANG-MCC-2018-10.pdf>`__ .

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

        self.ann_info['use_different_joint_weights'] = False
        assert self.ann_info['num_joints'] == 21
        self.ann_info['joint_weights'] = \
            np.ones((self.ann_info['num_joints'], 1), dtype=np.float32)

        self.db = self._get_db(ann_file)
        self.image_set = set(x['image_file'] for x in self.db)
        self.num_images = len(self.image_set)

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

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
            bbox=None,
            dataset='onehand10k')

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

            joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

            for ipt in range(num_joints):
                joints_3d[ipt, 0] = anno['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = anno['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = min(anno['keypoints'][ipt * 3 + 2], 1)
                joints_3d_visible[ipt, :] = (t_vis, t_vis, 0)

            center, scale = self._xywh2cs(*anno['bbox'][:4])
            newitem['center'] = center
            newitem['scale'] = scale
            newitem['joints_3d'] = joints_3d
            newitem['joints_3d_visible'] = joints_3d_visible
            newitem['bbox'] = anno['bbox'][:4]
            gt_db.append(newitem)

        return gt_db

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

    def evaluate(self, outputs, res_folder, metric='PCK', **kwargs):
        """Evaluate OneHand10K keypoint results. metric (str | list[str]):
        Metrics to be evaluated. Options are 'PCK', 'AUC', 'EPE'.

        'PCK': ||pre[i] - joints_3d[i]|| < 0.2 * max(w, h)
        'AUC': area under curve
        'EPE': end-point error
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = []

        for preds, boxes, image_path in outputs:
            str_image_path = ''.join(image_path)
            image_id = int(osp.basename(osp.splitext(str_image_path)[0]))

            kpts.append({
                'keypoints': preds[0].tolist(),
                'center': boxes[0][0:2].tolist(),
                'scale': boxes[0][2:4].tolist(),
                'area': float(boxes[0][4]),
                'score': float(boxes[0][5]),
                'image_id': image_id,
            })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        return name_value

    def _write_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file, metrics):
        """Keypoint evaluation.

        Report PCK, AUC or EPE.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        for pred, item in zip(preds, self.db):
            outputs.append(pred['keypoints'])
            gts.append(item['joints_3d'])
        outputs = np.array(outputs)[:, :, :-1]
        gts = np.array(gts)[:, :, :-1]

        if 'PCK' in metrics:
            hit = 0
            exist = 0

            for pred, item in zip(preds, self.db):
                bbox = np.array(item['bbox'])
                threshold = np.max(bbox[2:]) * 0.2
                h, _, e = keypoint_pck_accuracy(
                    np.array(pred['keypoints'])[None, :, :-1],
                    np.array(item['joints_3d'])[None, :, :-1], 1,
                    np.array([[threshold, threshold]]))
                hit += len(h[h > 0])
                exist += e
            pck = hit / exist

            info_str.append(('PCK', pck))

        if 'AUC' in metrics:
            info_str.append(('AUC', keypoint_auc(outputs, gts, 30)))

        if 'EPE' in metrics:

            info_str.append(('EPE', keypoint_epe(outputs, gts)))
        return info_str
