import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.datasets.builder import DATASETS
from .bottom_up_coco import BottomUpCocoDataset


@DATASETS.register_module()
class BottomUpCocoWholeBodyDataset(BottomUpCocoDataset):
    """CocoWholeBodyDataset dataset for bottom-up pose estimation.

    `Whole-Body Human Pose Estimation in the Wild' ECCV'2020
    More details can be found in the `paper
    <https://arxiv.org/abs/2007.11858>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    In total, we have 133 keypoints for wholebody pose estimation.

    COCO-WholeBody keypoint indexes::
        0-16: 17 body keypoints
        17-22: 6 foot keypoints
        23-90: 68 face keypoints
        91-132: 42 hand keypoints

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
        super(BottomUpCocoDataset,
              self).__init__(ann_file, img_prefix, data_cfg, pipeline,
                             test_mode)

        self.ann_info['flip_pairs'] = self._make_flip_pairs()
        self.ann_info['flip_index'] = self.get_flip_index_from_flip_pairs(
            self.ann_info['flip_pairs'])

        # joint index starts from 0
        skeleton_body = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                         [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                         [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                         [3, 5], [4, 6]]
        skeleton_foot = [[15, 17], [15, 18], [15, 19], [16, 20], [16, 21],
                         [16, 22]]
        skeleton_face = [[0, 53], [53, 52], [52, 51], [51, 50], [50, 65],
                         [65, 66], [66, 67], [67, 68], [65, 70], [70, 69],
                         [50, 62], [62, 61], [61, 60], [60, 59], [62, 63],
                         [63, 64], [50, 45], [45, 46], [46, 47], [47, 48],
                         [48, 49], [50, 44], [44, 43], [43, 42], [42, 41],
                         [41, 40], [53, 56], [56, 57], [57, 58], [56, 55],
                         [55, 54], [56, 74], [74, 75], [75, 76], [76, 77],
                         [74, 73], [73, 72], [72, 71], [74, 85], [85, 86],
                         [86, 87], [85, 84], [84, 83], [85, 89], [89, 88],
                         [89, 90], [89, 80], [80, 79], [79, 78], [80, 81],
                         [81, 82], [80, 31], [31, 32], [32, 33], [33, 34],
                         [34, 35], [35, 36], [36, 37], [37, 38], [38, 39],
                         [31, 30], [30, 29], [29, 28], [28, 27], [27, 26],
                         [26, 25], [25, 24], [24, 23]]
        skeleton_lefthand = [[9, 91], [91, 92], [92, 93], [93, 94], [94, 95],
                             [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
                             [100, 101], [101, 102], [102, 103], [91, 104],
                             [104, 105], [105, 106], [106, 107], [91, 108],
                             [108, 109], [109, 110], [110, 111]]
        skeleton_righthand = [[10, 112], [112, 113], [113, 114], [114, 115],
                              [115, 116], [112, 117], [117, 118], [118, 119],
                              [119, 120], [112, 121], [121, 122], [122, 123],
                              [123, 124], [112, 125], [125, 126], [126, 127],
                              [127, 128], [112, 129], [129, 130], [130, 131],
                              [131, 132]]

        self.ann_info['skeleton'] = (
            skeleton_body + skeleton_foot + skeleton_face + skeleton_lefthand +
            skeleton_righthand)

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = \
            np.ones((self.ann_info['num_joints'], 1), dtype=np.float32)

        self.body_num = 17
        self.foot_num = 6
        self.face_num = 68
        self.left_hand_num = 21
        self.right_hand_num = 21

        # 'https://github.com/jin-s13/COCO-WholeBody/blob/master/'
        # 'evaluation/myeval_wholebody.py#L170'
        self.sigmas_body = [
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]
        self.sigmas_foot = [0.068, 0.066, 0.066, 0.092, 0.094, 0.094]
        self.sigmas_face = [
            0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025, 0.020,
            0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045, 0.013,
            0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
            0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010,
            0.017, 0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011,
            0.012, 0.010, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007,
            0.010, 0.008, 0.009, 0.009, 0.009, 0.007, 0.007, 0.008, 0.011,
            0.008, 0.008, 0.008, 0.01, 0.008
        ]
        self.sigmas_lefthand = [
            0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
            0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
            0.019, 0.022, 0.031
        ]
        self.sigmas_righthand = [
            0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
            0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
            0.019, 0.022, 0.031
        ]

        self.sigmas_wholebody = (
            self.sigmas_body + self.sigmas_foot + self.sigmas_face +
            self.sigmas_lefthand + self.sigmas_righthand)

        self.sigmas = np.array(self.sigmas_wholebody)

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
        if not test_mode:
            self.img_ids = [
                img_id for img_id in self.img_ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'coco_wholebody'

        print(f'=> num_images: {self.num_images}')

    @staticmethod
    def _make_flip_pairs():
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                [111, 132]]

        return body + foot + face + hand

    def _get_joints(self, anno):
        """Get joints for all people in an image."""
        num_people = len(anno)

        if self.ann_info['scale_aware_sigma']:
            joints = np.zeros((num_people, self.ann_info['num_joints'], 4),
                              dtype=np.float32)
        else:
            joints = np.zeros((num_people, self.ann_info['num_joints'], 3),
                              dtype=np.float32)

        for i, obj in enumerate(anno):
            keypoints = np.array(obj['keypoints'] + obj['foot_kpts'] +
                                 obj['face_kpts'] + obj['lefthand_kpts'] +
                                 obj['righthand_kpts']).reshape(-1, 3)

            joints[i, :self.ann_info['num_joints'], :3] = keypoints
            if self.ann_info['scale_aware_sigma']:
                # get person box
                box = obj['bbox']
                size = max(box[2], box[3])
                sigma = size / self.base_size * self.base_sigma
                if self.int_sigma:
                    sigma = int(np.ceil(sigma))
                assert sigma > 0, sigma
                joints[i, :, 3] = sigma

        return joints

    def _get_part_score(self, keypoints):
        """Get part score for new evaluation tools."""
        kpt_score = 0
        valid_num = 0
        num_joints = int(len(keypoints) / 3)
        for n_jt in range(0, num_joints):
            t_s = keypoints[n_jt * 3 + 2]
            if t_s > 0.2:
                kpt_score = kpt_score + t_s
                valid_num = valid_num + 1
        if valid_num != 0:
            kpt_score = kpt_score / valid_num
        part_score = kpt_score

        return float(part_score)

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

            for img_kpt, key_point in zip(img_kpts, key_points):
                kpt = key_point.reshape((self.ann_info['num_joints'], 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id':
                    img_kpt['image_id'],
                    'category_id':
                    cat_id,
                    'keypoints':
                    key_point[cuts[0]:cuts[1]].tolist(),
                    'foot_kpts':
                    key_point[cuts[1]:cuts[2]].tolist(),
                    'face_kpts':
                    key_point[cuts[2]:cuts[3]].tolist(),
                    'lefthand_kpts':
                    key_point[cuts[3]:cuts[4]].tolist(),
                    'righthand_kpts':
                    key_point[cuts[4]:cuts[5]].tolist(),
                    'score':
                    img_kpt['score'],
                    # 'score':
                    # self._get_part_score(key_point[cuts[0]:cuts[1]]),
                    # 'foot_score':
                    # self._get_part_score(key_point[cuts[1]:cuts[2]]),
                    # 'face_score':
                    # self._get_part_score(key_point[cuts[2]:cuts[3]]),
                    # 'lefthand_score':
                    # self._get_part_score(key_point[cuts[3]:cuts[4]]),
                    # 'righthand_score':
                    # self._get_part_score(key_point[cuts[4]:cuts[5]]),
                    # 'wholebody_score':
                    # img_kpt['score'],
                    'bbox': [left_top[0], left_top[1], w, h]
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI."""
        coco_det = self.coco.loadRes(res_file)

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_body',
            np.array(self.sigmas_body),
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_foot',
            np.array(self.sigmas_foot),
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_face',
            np.array(self.sigmas_face),
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_lefthand',
            np.array(self.sigmas_lefthand),
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_righthand',
            np.array(self.sigmas_righthand),
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_wholebody',
            np.array(self.sigmas_wholebody),
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
