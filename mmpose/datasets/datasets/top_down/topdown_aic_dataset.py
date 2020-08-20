import os
from collections import OrderedDict, defaultdict

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from ....core.post_processing import oks_nms, soft_oks_nms
from ...registry import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


def _get_mapping_id_name(imgs):
    """
    Args:
        imgs (dict): dict of image info.
    Returns:
        id2name (dict): mapping image id to name.
        name2id (dict): mapping image name to id.
    """
    id2name = {}
    name2id = {}
    for image_id, image in imgs.items():
        file_name = image['file_name']
        id2name[image_id] = file_name
        name2id[file_name] = image_id

    return id2name, name2id


@DATASETS.register_module()
class TopDownAicDataset(TopDownCocoDataset):
    """AicDataset dataset for top-down pose estimation.

    `AI Challenger : A Large-scale Dataset for Going Deeper
    in Image Understanding <https://arxiv.org/abs/1711.06475>`_

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    AIC keypoint indexes::
        0: "right_shoulder",
        1: "right_elbow",
        2: "right_wrist",
        3: "left_shoulder",
        4: "left_elbow",
        5: "left_wrist",
        6: "right_hip",
        7: "right_knee",
        8: "right_ankle",
        9: "left_hip",
        10: "left_knee",
        11: "left_ankle",
        12: "head_top",
        13: "neck"

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

        self.ann_info['flip_pairs'] = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10],
                                       [8, 11]]

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 12, 13)
        self.ann_info['lower_body_ids'] = (6, 7, 8, 9, 10, 11)

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array(
            [1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.sigmas = np.array([
            0.01388152, 0.01515228, 0.01057665, 0.01417709, 0.01497891,
            0.01402144, 0.03909642, 0.03686941, 0.01981803, 0.03843971,
            0.03412318, 0.02415081, 0.01291456, 0.01236173
        ])

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
        self.id2name, self.name2id = _get_mapping_id_name(self.coco.imgs)

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotation_kernel(self, index):
        """load annotation from COCOAPI.

        Note:
            bbox:[x1, y1, w, h]
        Args:
            index: coco image id
        Returns:
            db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        num_joints = self.ann_info['num_joints']

        ann_ids = self.coco.getAnnIds(imgIds=index)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj
                    or obj['area'] > 0) and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            if max(obj['keypoints']) == 0:
                continue
            joints_3d = np.zeros((num_joints, 3), dtype=np.float)
            joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:, :2] = keypoints[:, :2]
            joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[index])
            rec.append({
                'image_file': image_file,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': 'aic',
                'bbox_score': 1
            })

        return rec

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
            metric (str): Metric to be performed. Defaults: 'mAP'.

        Returns:
            name_value (dict): Evaluation results for evaluation metric.
        """
        assert metric == 'mAP'

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = defaultdict(list)
        for preds, boxes, image_path in outputs:
            str_image_path = ''.join(image_path)
            image_id = self.name2id[os.path.basename(str_image_path)]

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
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
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

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thr,
                    sigmas=self.sigmas)
            else:
                keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
                               oks_thr,
                               sigmas=self.sigmas)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)

        return name_value

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""

        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(
            self.coco, coco_dt, 'keypoints', self.sigmas, use_area=False)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            'AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
            'AR .75', 'AR (M)', 'AR (L)'
        ]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
