import os
from collections import OrderedDict

import json_tricks as json
import numpy as np

from mmpose.core.evaluation.mesh_eval import compute_similarity_transform
from mmpose.datasets.builder import DATASETS
from .mesh_base_dataset import MeshBaseDataset


@DATASETS.register_module()
class MeshH36MDataset(MeshBaseDataset):
    """Human3.6M Dataset for 3D human mesh estimation. It inherits all function
    from MeshBaseDataset and has its own evaluate fuction.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def evaluate(self, outputs, res_folder, metric='joint_error', logger=None):
        """Evaluate 3D keypoint results."""
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['joint_error']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        kpts = []
        for preds, boxes, image_path in outputs:
            kpts.append({
                'keypoints': preds[0].tolist(),
                'center': boxes[0][0:2].tolist(),
                'scale': boxes[0][2:4].tolist(),
                'area': float(boxes[0][4]),
                'score': float(boxes[0][5]),
                'image': image_path,
            })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file)
        name_value = OrderedDict(info_str)
        return name_value

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        joint_error = []
        joint_error_pa = []

        for pred, item in zip(preds, self.db):
            error, error_pa = self.evaluate_kernel(pred['keypoints'][0],
                                                   item['joints_3d'],
                                                   item['joints_3d_visible'])
            joint_error.append(error)
            joint_error_pa.append(error_pa)

        mpjpe = np.array(joint_error).mean()
        mpjpe_pa = np.array(joint_error_pa).mean()

        info_str = []
        info_str.append(('MPJPE', mpjpe * 1000))
        info_str.append(('MPJPE-PA', mpjpe_pa * 1000))
        return info_str

    @staticmethod
    def evaluate_kernel(pred_joints_3d, joints_3d, joints_3d_visible):
        """Evaluate one example."""
        # Only 14 lsp joints are used for evaluation
        joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
        assert joints_3d_visible[joint_mapper].min() > 0

        pred_joints_3d = np.array(pred_joints_3d)
        pred_joints_3d = pred_joints_3d[joint_mapper, :]
        pred_pelvis = (pred_joints_3d[[2]] + pred_joints_3d[[3]]) / 2
        pred_joints_3d = pred_joints_3d - pred_pelvis

        gt_joints_3d = joints_3d[joint_mapper, :]
        gt_pelvis = (gt_joints_3d[[2]] + gt_joints_3d[[3]]) / 2
        gt_joints_3d = gt_joints_3d - gt_pelvis

        error = pred_joints_3d - gt_joints_3d
        error = np.linalg.norm(error, ord=2, axis=-1).mean(axis=-1)

        pred_joints_3d_aligned = compute_similarity_transform(
            pred_joints_3d, gt_joints_3d)
        error_pa = pred_joints_3d_aligned - gt_joints_3d
        error_pa = np.linalg.norm(error_pa, ord=2, axis=-1).mean(axis=-1)

        return error, error_pa
