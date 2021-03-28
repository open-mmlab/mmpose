import os.path as osp
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np

from mmpose.core.evaluation.mesh_eval import compute_similarity_transform
from ...registry import DATASETS
from .body3d_base_dataset import Body3DBaseDataset


@DATASETS.register_module()
class Body3DH36MDataset(Body3DBaseDataset):
    """Human3.6M dataset for 3D human pose estimation.

    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Human3.6M keypoint indexes::
        0: 'root (pelvis)',
        1: 'left_hip',
        2: 'left_knee',
        3: 'left_foot',
        4: 'right_hip',
        5: 'right_knee',
        6: 'right_foot',
        7: 'spine',
        8: 'thorax',
        9: 'neck_base',
        10: 'head',
        11: 'left_shoulder',
        12: 'left_elbow',
        13: 'left_wrist',
        14: 'right_shoulder',
        15: 'right_elbow',
        16: 'right_wrist'


    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    JOINT_NAMES = [
        'Root', 'LHip', 'LKnee', 'LFoot', 'RHip', 'RKnee', 'RFoot', 'Spine',
        'Thorax', 'NeckBase', 'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist'
    ]

    # The joint indices in the annotation file
    JOINT_IDX_ANNOTATION = [
        14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6
    ]
    # The total joint number in the annotatino file
    JOINT_NUM_ANNOTATION = 24

    # 2D joint source options:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate in the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'joint_error'}

    def load_config(self, data_cfg):
        super().load_config(data_cfg)
        # h36m specific attributes
        self.joint_2d_src = data_cfg.get('joint_2d_src', 'gt')
        if self.joint_2d_src not in self.SUPPORTED_JOINT_2D_SRC:
            raise ValueError(
                f'Unsupported joint_2d_src "{self.joint_2d_src}". '
                f'Supported options are {self.SUPPORTED_JOINT_2D_SRC}')

        self.joint_2d_det_file = data_cfg.get('joint_2d_det_file', None)

        self.need_camera_param = data_cfg.get('need_camera_param', False)
        if self.need_camera_param:
            assert 'camera_param_file' in data_cfg
            self.camera_param = self._load_camera_param(
                data_cfg['camera_param_file'])

        # h36m specific annotation info
        ann_info = {}
        ann_info['flip_pairs'] = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15],
                                  [13, 16]]
        ann_info['upper_body_ids'] = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        ann_info['lower_body_ids'] = (1, 2, 3, 4, 5, 6)
        ann_info['use_different_joint_weights'] = False

        self.ann_info.update(ann_info)

    def load_annotations(self):
        data_info = super().load_annotations()

        # convert 3D joints from original 24-keypoint to standard 17-keypoint
        assert data_info['joints_3d'].shape[1] == self.JOINT_NUM_ANNOTATION
        data_info['joints_3d'] = data_info['joints_3d'][:, self.
                                                        JOINT_IDX_ANNOTATION]

        # get 2D joints
        if self.joint_2d_src == 'gt':
            assert data_info['joints_2d'].shape[1] == self.JOINT_NUM_ANNOTATION
            data_info['joints_2d'] = data_info[
                'joints_2d'][:, self.JOINT_IDX_ANNOTATION]
        elif self.joint_2d_src == 'detection':
            data_info['joints_2d'] = self._load_joint_2d_detection(
                self.joint_2d_det_file)
        elif self.joint_2d_src == 'pipeline':
            pass
        else:
            raise NotImplementedError(
                f'Unhandled joint_2d_src option {self.joint_2d_src}')

        return data_info

    @staticmethod
    def _parse_h36m_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        A typical h36m image filename is like:
        S1_Directions_1.54138969_000001.jpg
        """
        subj, rest = osp.basename(imgname).split('_', 1)
        action, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)

        return subj, action, camera

    def build_sample_indices(self):
        """Split original videos into sequences and build frame indices.

        This method overrides the default one in the base class.
        """

        # Group frames into videos. Assume that self.data_info is
        # chronological.
        video_frames = defaultdict(list)
        for idx, imgname in enumerate(self.data_info['imgnames']):
            subj, action, camera = self._parse_h36m_imgname(imgname)
            video_frames[(subj, action, camera)].append(idx)

        # build sample indices
        sample_indices = []
        _len = (self.seq_len - 1) * self.seq_frame_interval + 1
        _step = self.seq_frame_interval
        for _, _indices in sorted(video_frames.items()):
            n_frame = len(_indices)
            seqs_from_video = [
                _indices[i:(i + _len):_step]
                for i in range(0, n_frame - _len + 1)
            ]
            sample_indices.extend(seqs_from_video)

        return sample_indices

    def _load_joint_2d_detection(self, det_file):
        """"Load 2D joint detection results from file."""
        joints_2d = np.load(det_file).astype(np.float32)
        assert joints_2d.shape[0] == self.data_info['joint_3d'].shape[0]
        assert joints_2d.shape[2] == 3

        return joints_2d

    def evaluate(self, outputs, res_folder, metric='joint_error', **kwargs):
        # bound the kwargs
        assert len(kwargs) == 0

        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}" for human3.6 dataset.'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        res_file = osp.join(res_folder, 'result_keypoints.json')
        kpts = []
        for output in outputs:
            preds = output['preds']
            image_paths = output['target_image_paths']
            batch_size = len(image_paths)
            for i in range(batch_size):
                target_id = self.name2id[image_paths[i]]
                kpts.append({
                    'keypoints': preds[i],
                    'target_id': target_id,
                })

        mmcv.dump(kpts, res_file)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'joint_error':
                _nv_tuples = self._report_joint_error(kpts)
            else:
                raise NotImplementedError
            name_value_tuples.extend(_nv_tuples)

        return OrderedDict(name_value_tuples)

    def _report_joint_error(self, keypoint_results):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """

        joint_error = []
        joint_error_pa = []
        for result in keypoint_results:
            pred = result['keypoints']
            target_id = result['target_id']
            target, target_visible = np.split(
                self.data_info['joints_3d'][target_id], [3], axis=-1)

            error, error_pa = self._joint_error_kernel(pred, target,
                                                       target_visible)

            joint_error.append(error)
            joint_error_pa.append(error_pa)

        name_value_tuples = [('MPJPE', np.mean(joint_error)),
                             ('MPJPE-PA', np.mean(joint_error_pa))]

        return name_value_tuples

    @staticmethod
    def _joint_error_kernel(pred_joints_3d, gt_joints_3d, joints_3d_visible):
        """Evaluate one example."""
        assert (joints_3d_visible > 0).all()

        root_joint_idx = 0
        pred_joints_3d = np.array(pred_joints_3d)
        pred_root = pred_joints_3d[root_joint_idx]
        pred_joints_3d = pred_joints_3d - pred_root

        gt_root = gt_joints_3d[root_joint_idx]
        gt_joints_3d = gt_joints_3d - gt_root

        error = pred_joints_3d - gt_joints_3d
        error = np.linalg.norm(error, ord=2, axis=-1).mean(axis=-1)

        pred_joints_3d_aligned = compute_similarity_transform(
            pred_joints_3d, gt_joints_3d)
        error_pa = pred_joints_3d_aligned - gt_joints_3d
        error_pa = np.linalg.norm(error_pa, ord=2, axis=-1).mean(axis=-1)

        return error, error_pa

    def _load_camera_param(self, camear_param_file):
        """Load camera parameters from file."""
        return mmcv.load(camear_param_file)

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name."""
        assert hasattr(self, 'camera_param')
        subj, _, camera = self._parse_h36m_imgname(imgname)
        return self.camera_param[(subj, camera)]
