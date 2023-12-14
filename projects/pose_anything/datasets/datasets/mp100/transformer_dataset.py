import os
import random
from collections import OrderedDict

import numpy as np
from xtcocotools.coco import COCO

from mmpose.datasets import DATASETS
from .transformer_base_dataset import TransformerBaseDataset


@DATASETS.register_module()
class TransformerPoseDataset(TransformerBaseDataset):

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 valid_class_ids,
                 max_kpt_num=None,
                 num_shots=1,
                 num_queries=100,
                 num_episodes=1,
                 test_mode=False):
        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        self.ann_info['flip_pairs'] = []

        self.ann_info['upper_body_ids'] = []
        self.ann_info['lower_body_ids'] = []

        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['joint_weights'] = np.array([
            1.,
        ], dtype=np.float32).reshape((self.ann_info['num_joints'], 1))

        self.coco = COCO(ann_file)

        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.img_ids = self.coco.getImgIds()
        self.classes = [
            cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())
        ]

        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, self.coco.getCatIds()))
        self._ind_to_class = dict(zip(self.coco.getCatIds(), self.classes))

        if valid_class_ids is not None:  # None by default
            self.valid_class_ids = valid_class_ids
        else:
            self.valid_class_ids = self.coco.getCatIds()
        self.valid_classes = [
            self._ind_to_class[ind] for ind in self.valid_class_ids
        ]

        self.cats = self.coco.cats
        self.max_kpt_num = max_kpt_num

        # Also update self.cat2obj
        self.db = self._get_db()

        self.num_shots = num_shots

        if not test_mode:
            # Update every training epoch
            self.random_paired_samples()
        else:
            self.num_queries = num_queries
            self.num_episodes = num_episodes
            self.make_paired_samples()

    def random_paired_samples(self):
        num_datas = [
            len(self.cat2obj[self._class_to_ind[cls]])
            for cls in self.valid_classes
        ]

        # balance the dataset
        max_num_data = max(num_datas)

        all_samples = []
        for cls in self.valid_class_ids:
            for i in range(max_num_data):
                shot = random.sample(self.cat2obj[cls], self.num_shots + 1)
                all_samples.append(shot)

        self.paired_samples = np.array(all_samples)
        np.random.shuffle(self.paired_samples)

    def make_paired_samples(self):
        random.seed(1)
        np.random.seed(0)

        all_samples = []
        for cls in self.valid_class_ids:
            for _ in range(self.num_episodes):
                shots = random.sample(self.cat2obj[cls],
                                      self.num_shots + self.num_queries)
                sample_ids = shots[:self.num_shots]
                query_ids = shots[self.num_shots:]
                for query_id in query_ids:
                    all_samples.append(sample_ids + [query_id])

        self.paired_samples = np.array(all_samples)

    def _select_kpt(self, obj, kpt_id):
        obj['joints_3d'] = obj['joints_3d'][kpt_id:kpt_id + 1]
        obj['joints_3d_visible'] = obj['joints_3d_visible'][kpt_id:kpt_id + 1]
        obj['kpt_id'] = kpt_id

        return obj

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        id2name = {}
        name2id = {}
        for image_id, image in imgs.items():
            file_name = image['file_name']
            id2name[image_id] = file_name
            name2id[file_name] = image_id

        return id2name, name2id

    def _get_db(self):
        """Ground truth bbox and keypoints."""
        self.obj_id = 0

        self.cat2obj = {}
        for i in self.coco.getCatIds():
            self.cat2obj.update({i: []})

        gt_db = []
        for img_id in self.img_ids:
            gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))

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

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            if 'bbox' not in obj:
                continue
            x, y, w, h = obj['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                valid_objs.append(obj)
        objs = valid_objs

        bbox_id = 0
        rec = []
        for obj in objs:
            if 'keypoints' not in obj:
                continue
            if max(obj['keypoints']) == 0:
                continue
            if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
                continue

            category_id = obj['category_id']
            # the number of keypoint for this specific category
            cat_kpt_num = int(len(obj['keypoints']) / 3)
            if self.max_kpt_num is None:
                kpt_num = cat_kpt_num
            else:
                kpt_num = self.max_kpt_num

            joints_3d = np.zeros((kpt_num, 3), dtype=np.float32)
            joints_3d_visible = np.zeros((kpt_num, 3), dtype=np.float32)

            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            joints_3d[:cat_kpt_num, :2] = keypoints[:, :2]
            joints_3d_visible[:cat_kpt_num, :2] = np.minimum(
                1, keypoints[:, 2:3])

            center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

            image_file = os.path.join(self.img_prefix, self.id2name[img_id])
            if os.path.exists(image_file):
                self.cat2obj[category_id].append(self.obj_id)

                rec.append({
                    'image_file':
                    image_file,
                    'center':
                    center,
                    'scale':
                    scale,
                    'rotation':
                    0,
                    'bbox':
                    obj['clean_bbox'][:4],
                    'bbox_score':
                    1,
                    'joints_3d':
                    joints_3d,
                    'joints_3d_visible':
                    joints_3d_visible,
                    'category_id':
                    category_id,
                    'cat_kpt_num':
                    cat_kpt_num,
                    'bbox_id':
                    self.obj_id,
                    'skeleton':
                    self.coco.cats[obj['category_id']]['skeleton'],
                })
                bbox_id = bbox_id + 1
                self.obj_id += 1

        return rec

    def _xywh2cs(self, x, y, w, h):
        """This encodes bbox(x,y,w,w) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - center (np.ndarray[float32](2,)): center of the bbox (x, y).
            - scale (np.ndarray[float32](2,)): scale of the bbox w & h.
        """
        aspect_ratio = self.ann_info['image_size'][0] / self.ann_info[
            'image_size'][1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
        #
        # if (not self.test_mode) and np.random.rand() < 0.3:
        #     center += 0.4 * (np.random.rand(2) - 0.5) * [w, h]

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
        # padding to include proper amount of context
        scale = scale * 1.25

        return center, scale

    def evaluate(self, outputs, res_folder, metric='PCK', **kwargs):
        """Evaluate interhand2d keypoint results. The pose prediction results
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
                :image_paths (list[str]): For example, ['C', 'a', 'p', 't',
                    'u', 'r', 'e', '1', '2', '/', '0', '3', '9', '0', '_',
                    'd', 'h', '_', 't', 'o', 'u', 'c', 'h', 'R', 'O', 'M',
                    '/', 'c', 'a', 'm', '4', '1', '0', '2', '0', '9', '/',
                    'i', 'm', 'a', 'g', 'e', '6', '2', '4', '3', '4', '.',
                    'j', 'p', 'g']
                :output_heatmap (np.ndarray[N, K, H, W]): model outputs.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE', 'NME']
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
