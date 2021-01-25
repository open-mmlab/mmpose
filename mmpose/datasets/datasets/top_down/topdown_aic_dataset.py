import warnings

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from ...registry import DATASETS
from .topdown_coco_dataset import TopDownCocoDataset


@DATASETS.register_module()
class TopDownAicDataset(TopDownCocoDataset):
    """AicDataset dataset for top-down pose estimation.

    `AI Challenger : A Large-scale Dataset for Going Deeper
    in Image Understanding <https://arxiv.org/abs/1711.06475>`__

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
        self.det_bbox_thr = data_cfg.get('det_bbox_thr', 0.0)
        if 'image_thr' in data_cfg:
            warnings.warn(
                'image_thr is deprecated, '
                'please use det_bbox_thr instead', DeprecationWarning)
            self.det_bbox_thr = data_cfg['image_thr']
        self.use_nms = data_cfg.get('use_nms', True)
        self.soft_nms = data_cfg['soft_nms']
        self.nms_thr = data_cfg['nms_thr']
        self.oks_thr = data_cfg['oks_thr']
        self.vis_thr = data_cfg['vis_thr']

        self.ann_info['flip_pairs'] = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10],
                                       [8, 11]]

        self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 12, 13)
        self.ann_info['lower_body_ids'] = (6, 7, 8, 9, 10, 11)

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array(
            [1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.2, 1.5, 1., 1.],
            dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        # 'https://github.com/AIChallenger/AI_Challenger_2017/blob/master/'
        # 'Evaluation/keypoint_eval/keypoint_eval.py#L50'
        # delta = 2 x sigma
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
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'aic'

        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        assert self.use_gt_bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)
        coco_eval = COCOeval(
            self.coco, coco_det, 'keypoints', self.sigmas, use_area=False)
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
