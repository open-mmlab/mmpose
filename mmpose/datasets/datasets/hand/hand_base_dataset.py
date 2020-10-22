import copy
from abc import ABCMeta, abstractmethod

import json_tricks as json
import numpy as np
from torch.utils.data import Dataset
from xtcocotools.coco import COCO

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_pck_accuracy)
from mmpose.datasets.pipelines import Compose


class HandBaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for hand datasets.

    All hand datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`, 'evaluate'

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

        self.image_info = {}
        self.ann_info = {}

        self.annotations_path = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['flip_pairs'] = []

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        self.db = []

        self.pipeline = Compose(self.pipeline)

    def _get_mapping_id_name(self, imgs):
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

    def _xywh2cs(self, x, y, w, h, padding=1.25):
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

        scale = scale * padding

        return center, scale

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """Evaluate keypoint results."""
        raise NotImplementedError

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

        if 'PCK' in metrics:
            hit = 0
            exist = 0

            for pred, item in zip(preds, self.db):
                bbox = np.array(item['bbox'])
                threshold = np.max(bbox[2:]) * 0.2

                h, _, e = keypoint_pck_accuracy(
                    np.array(pred['keypoints'])[None, :, :-1],
                    np.array(item['joints_3d'])[None, :, :-1],
                    (np.array(item['joints_3d_visible'])[None, :, 0]) > 0, 1,
                    np.array([[threshold, threshold]]))
                hit += len(h[h > 0])
                exist += e
            pck = hit / exist

            info_str.append(('PCK', pck))

        if 'PCKh' in metrics:
            hit = 0
            exist = 0

            for pred, item in zip(preds, self.db):
                threshold = item['head_size'] * 0.7
                h, _, e = keypoint_pck_accuracy(
                    np.array(pred['keypoints'])[None, :, :-1],
                    np.array(item['joints_3d'])[None, :, :-1],
                    (np.array(item['joints_3d_visible'])[None, :, 0]) > 0, 1,
                    np.array([[threshold, threshold]]))
                hit += len(h[h > 0])
                exist += e
            pck = hit / exist

            info_str.append(('PCKh', pck))

        if 'AUC' in metrics or 'EPE' in metrics:
            outputs = []
            gts = []
            masks = []

            for pred, item in zip(preds, self.db):
                outputs.append(np.array(pred['keypoints'])[None, :, :-1])
                gts.append(np.array(item['joints_3d'])[None, :, :-1])
                masks.append(
                    (np.array(item['joints_3d_visible'])[None, :, 0]) > 0)

            outputs = np.concatenate(outputs, axis=1)
            gts = np.concatenate(gts, axis=1)
            masks = np.concatenate(masks, axis=1)

            if 'AUC' in metrics:
                info_str.append(('AUC', keypoint_auc(outputs, gts, masks, 30)))

            if 'EPE' in metrics:
                info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))

        return info_str

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)
