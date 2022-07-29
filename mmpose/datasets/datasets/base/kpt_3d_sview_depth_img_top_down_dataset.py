# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

import json_tricks as json
import numpy as np
from torch.utils.data import Dataset
from xtcocotools.coco import COCO

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_pck_accuracy)
from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose


class Kpt3dSviewDepthImgTopDownDataset(Dataset, metaclass=ABCMeta):
    """Base class for keypoint 3D top-down pose estimation with single-view
    depth image as the input.

    All depth-based datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_db`, 'evaluate'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        coco_style (bool): Whether the annotation json is coco-style.
            Default: True
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 coco_style=True,
                 test_mode=False):

        self.image_info = {}
        self.ann_info = {}

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        self.ann_info['num_joints'] = data_cfg['num_joints']

        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        self.ann_info['num_output_channels'] = data_cfg['num_output_channels']
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        if dataset_info is None:
            raise ValueError(
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.')

        dataset_info = DatasetInfo(dataset_info)

        assert self.ann_info['num_joints'] == dataset_info.keypoint_num
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        self.ann_info['flip_index'] = dataset_info.flip_index
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        self.ann_info['skeleton'] = dataset_info.skeleton
        self.sigmas = dataset_info.sigmas
        self.dataset_name = dataset_info.dataset_name

        if coco_style:
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                self.classes = ['__background__'] + cats
                self.num_classes = len(self.classes)
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            self.img_ids = self.coco.getImgIds()
            self.num_images = len(self.img_ids)
            self.id2name, self.name2id = self._get_mapping_id_name(
                self.coco.imgs)

        self.db = []

        self.pipeline = Compose(self.pipeline)

    @staticmethod
    def _cam2pixel(cam_coord, f, c):
        """Transform the joints from their camera coordinates to their pixel
        coordinates.

        Note:
            N: number of joints

        Args:
            cam_coord (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            img_coord (ndarray[N, 3]): the coordinates (x, y, 0)
                in the image plane.
        """
        x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
        y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
        z = np.zeros_like(x)
        img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return img_coord

    @staticmethod
    def _world2cam(world_coord, R, T):
        """Transform the joints from their world coordinates to their camera
        coordinates.

        Note:
            N: number of joints

        Args:
            world_coord (ndarray[3, N]): 3D joints coordinates
                in the world coordinate system
            R (ndarray[3, 3]): camera rotation matrix
            T (ndarray[3, 1]): camera position (x, y, z)

        Returns:
            cam_coord (ndarray[3, N]): 3D joints coordinates
                in the camera coordinate system
        """
        cam_coord = np.dot(R, world_coord - T)
        return cam_coord

    @staticmethod
    def _pixel2cam(pixel_coord, f, c):
        """Transform the joints from their pixel coordinates to their camera
        coordinates.

        Note:
            N: number of joints

        Args:
            pixel_coord (ndarray[N, 3]): 3D joints coordinates
                in the pixel coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            cam_coord (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
        """
        x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
        y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
        z = pixel_coord[:, 2]
        cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return cam_coord

    @staticmethod
    def _xyz2uvd(xyz, f, c):
        """Transform the joints from their 3d xyz camera coordinates to their
        2.5D uvd coordinates.

        Note:
            N: number of joints

        Args:
            xyz (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            uvd (ndarray[N, 3]): the 2.5D coordinates (u, v, d) in the spatial.
        """
        u = xyz[:, 0] / (xyz[:, 2] + 1e-8) * f[0] + c[0]
        v = xyz[:, 1] / (xyz[:, 2] + 1e-8) * f[1] + c[1]
        d = xyz[:, 2]
        uvd = np.concatenate((u[:, None], v[:, None], d[:, None]), 1)
        return uvd

    @staticmethod
    def _uvd2xyz(uvd, f, c):
        """Transform the joints from their 2.5D uvd coordinates to their 3D xyz
        camera coordinates.

        Note:
            N: number of joints

        Args:
            uvd (ndarray[N, 3]): 3D joints coordinates
                in the pixel coordinate system
            f (ndarray[2]): focal length of x and y axis
            c (ndarray[2]): principal point of x and y axis

        Returns:
            xyz (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
        """
        x = (uvd[:, 0] - c[0]) / f[0] * uvd[:, 2]
        y = (uvd[:, 1] - c[1]) / f[1] * uvd[:, 2]
        z = uvd[:, 2]
        xyz = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
        return xyz

    @staticmethod
    def _center2bounds(center_uvd, cube_size, f):
        """

        Args:
            center_uvd (ndarray[1, 3]):
            cube_size (ndarray[3]):
            f (ndarray[2]): focal length of x and y axis

        Returns:
            bounds (ndarray[1, 6]): 2.5D bounds
        """

        ustart = center_uvd[:,
                            0] - (cube_size[0] / 2.) / center_uvd[:, 2] * f[0]
        vstart = center_uvd[:,
                            1] - (cube_size[1] / 2.) / center_uvd[:, 2] * f[1]
        uend = center_uvd[:, 0] + (cube_size[0] / 2.) / center_uvd[:, 2] * f[0]
        vend = center_uvd[:, 1] + (cube_size[1] / 2.) / center_uvd[:, 2] * f[1]
        dstart = center_uvd[:, 2] - cube_size[2] / 2.
        dend = center_uvd[:, 2] + cube_size[2] / 2.
        bounds = np.concatenate(
            (ustart[:, None], uend[:, None], vstart[:, None], vend[:, None],
             dstart[:, None], dend[:, None]), 1)
        return bounds

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

    def _xywh2cs(self, x, y, w, h, padding=1.25):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h (float): left, top, width and height
            padding (float): bounding box padding factor

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
        # padding to include proper amount of context
        scale = scale * padding

        return center, scale

    @abstractmethod
    def _get_db(self):
        """Load dataset."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.db)

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = copy.deepcopy(self.db[idx])
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts

    def _report_metric(self,
                       res_file,
                       metrics,
                       pck_thr=0.2,
                       pckh_thr=0.7,
                       auc_nor=30):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'PCK', 'PCKh', 'AUC', 'EPE', 'NME'.
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
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, item in zip(preds, self.db):

            self.ann_info['image_size']

            # pred_joint_xyz = self._uvd2xyz(
            #     np.array(pred['keypoints'], dtype=np.float32), item['focal'],
            #     item['princpt'])
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            if 'PCK' in metrics:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])

        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              threshold_bbox)
            info_str.append(('PCK', pck))

        if 'AUC' in metrics:
            info_str.append(('AUC', keypoint_auc(outputs, gts, masks,
                                                 auc_nor)))

        if 'EPE' in metrics:
            info_str.append(('EPE', keypoint_epe(outputs, gts, masks)))

        return info_str
