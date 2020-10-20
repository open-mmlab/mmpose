import os
from collections import OrderedDict

import json_tricks as json
import numpy as np
from xtcocotools.coco import COCO

from mmpose.core.evaluation.top_down_eval import (keypoint_auc, keypoint_epe,
                                                  keypoint_pck_accuracy)
from mmpose.datasets.builder import DATASETS
from .topdown_base_dataset import TopDownBaseDataset


@DATASETS.register_module()
class TopDownInterHand2DDataset(TopDownBaseDataset):
    """InterHand2.6M 2D dataset for top-down hand pose estimation.

    `InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image' Moon, Gyeongsik etal. ECCV'2020
    More details can be found in the `paper
    <https://arxiv.org/pdf/2008.09309.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    InterHand2.6M keypoint indexes::

        0: 'thumb4',
        1: 'thumb3',
        2: 'thumb2',
        3: 'thumb1',
        4: 'forefinger4',
        5: 'forefinger3',
        6: 'forefinger2',
        7: 'forefinger1',
        8: 'middle_finger4',
        9: 'middle_finger3',
        10: 'middle_finger2',
        11: 'middle_finger1',
        12: 'ring_finger4',
        13: 'ring_finger3',
        14: 'ring_finger2',
        15: 'ring_finger1',
        16: 'pinky_finger4',
        17: 'pinky_finger3',
        18: 'pinky_finger2',
        19: 'pinky_finger1',
        20: 'wrist'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (str): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 camera_file,
                 joint_file,
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

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.num_images = len(self.img_ids)
        self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)
        self.dataset_name = 'interhand2d'

        self.camera_file = camera_file
        self.joint_file = joint_file
        self.db = self._get_db()

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

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

    def _cam2pixel_(self, cam_coord, f, c):
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

    def _world2cam_(self, world_coord, R, T):
        """Transform the joints from their world coordinates to their camera
        coordinates.

        Note:
            N: number of joints

        Args:
            world_coord (ndarray[N, 3]): 3D joints coordinates
                in the world coordinate system
            R (ndarray[3, 3]): camera rotation matrix
            T (ndarray[3]): camera position (x, y, z)

        Returns:
            cam_coord (ndarray[N, 3]): 3D joints coordinates
                in the camera coordinate system
        """
        cam_coord = np.dot(R, world_coord - T)
        return cam_coord

    def _get_db(self):
        """Load dataset.

        Adapted from https://github.com/facebookresearch/InterHand2.6M
        Copyright (c) FaceBook Research, under CC-BY-NC 4.0 license.
        """
        with open(self.camera_file, 'r') as f:
            cameras = json.load(f)
        with open(self.joint_file, 'r') as f:
            joints = json.load(f)
        gt_db = []
        for img_id in self.img_ids:
            num_joints = self.ann_info['num_joints']

            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            ann = self.coco.loadAnns(ann_id)[0]
            img = self.coco.loadImgs(img_id)[0]

            capture_id = str(img['capture'])
            camera_name = img['camera']
            frame_idx = str(img['frame_idx'])
            image_file = os.path.join(self.img_prefix, self.id2name[img_id])

            camera_pos, camera_rot = np.array(
                cameras[capture_id]['campos'][camera_name],
                dtype=np.float32), np.array(
                    cameras[capture_id]['camrot'][camera_name],
                    dtype=np.float32)
            focal, principal_pt = np.array(
                cameras[capture_id]['focal'][camera_name],
                dtype=np.float32), np.array(
                    cameras[capture_id]['princpt'][camera_name],
                    dtype=np.float32)
            joint_world = np.array(
                joints[capture_id][frame_idx]['world_coord'], dtype=np.float32)
            joint_cam = self._world2cam_(
                joint_world.transpose(1, 0), camera_rot,
                camera_pos.reshape(3, 1)).transpose(1, 0)
            joint_img = self._cam2pixel_(joint_cam, focal, principal_pt)[:, :2]

            joint_valid = np.array(
                ann['joint_valid'], dtype=np.float32).reshape(2 * num_joints)
            # if root is not valid -> root-relative 3D pose is also not valid.
            # Therefore, mark all joints as invalid
            joint_valid[:num_joints] *= joint_valid[num_joints - 1]
            joint_valid[num_joints:] *= joint_valid[2 * num_joints - 1]

            left_valid = True if np.sum(
                joint_valid[:num_joints]) > 11 else False
            right_valid = True if np.sum(
                joint_valid[num_joints:]) > 11 else False

            if left_valid:
                rec = []
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d[:, :2] = joint_img[:num_joints, :]
                joints_3d_visible[:, :2] = np.minimum(
                    1,
                    np.array(joint_valid[:num_joints]).reshape(
                        (num_joints, 1)))

                bbox = [img['width'], img['height'], 0, 0]
                for i in range(num_joints):
                    if joints_3d_visible[i][0]:
                        bbox[0] = min(bbox[0], joints_3d[i][0])
                        bbox[1] = min(bbox[1], joints_3d[i][1])
                        bbox[2] = max(bbox[2], joints_3d[i][0])
                        bbox[3] = max(bbox[3], joints_3d[i][1])

                center, scale = self._xywh2cs(*bbox)

                rec.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': bbox,
                    'bbox_score': 1
                })
                gt_db.extend(rec)

            if right_valid:
                rec = []
                joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
                joints_3d[:, :2] = joint_img[num_joints:, :]
                joints_3d_visible[:, :2] = np.minimum(
                    1,
                    np.array(joint_valid[num_joints:]).reshape(
                        (num_joints, 1)))

                bbox = [img['width'], img['height'], 0, 0]
                for i in range(num_joints):
                    if joints_3d_visible[i][0]:
                        bbox[0] = min(bbox[0], joints_3d[i][0])
                        bbox[1] = min(bbox[1], joints_3d[i][1])
                        bbox[2] = max(bbox[2], joints_3d[i][0])
                        bbox[3] = max(bbox[3], joints_3d[i][1])

                center, scale = self._xywh2cs(*bbox)

                rec.append({
                    'image_file': image_file,
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': bbox,
                    'bbox_score': 1
                })
                gt_db.extend(rec)

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

        scale = scale * 1.5

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
                :preds (np.ndarray[1,K,3]): The first two dimensions are
                    coordinates, score is the third dimension of the array.
                :boxes (np.ndarray[1,6]): [center[0], center[1], scale[0]
                    , scale[1],area, score]
                :image_path (list[str]): For example, ['C', 'a', 'p', 't',
                    'u', 'r', 'e', '1', '2', '/', '0', '3', '9', '0', '_',
                    'd', 'h', '_', 't', 'o', 'u', 'c', 'h', 'R', 'O', 'M',
                    '/', 'c', 'a', 'm', '4', '1', '0', '2', '0', '9', '/',
                    'i', 'm', 'a', 'g', 'e', '6', '2', '4', '3', '4', '.',
                    'j', 'p', 'g']
                :output_heatmap (np.ndarray[N, K, H, W]): model outpus.

            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'PCK', 'AUC', 'EPE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'AUC', 'EPE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')

        kpts = []

        for preds, boxes, image_path, _ in outputs:
            str_image_path = ''.join(image_path)
            image_id = self.name2id[str_image_path[len(self.img_prefix):]]

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
