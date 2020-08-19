import copy as cp
import os

import numpy as np

from mmpose.datasets.builder import DATASETS
from .mesh_base_dataset import MeshBaseDataset


@DATASETS.register_module()
class MeshLspDataset(MeshBaseDataset):
    """LSP Dataset dataset for 3D human mesh estimation.

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

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        # flip_pairs in mesh dataset
        # For all mesh dataset, we use 24 joints as CMR and SPIN.
        self.ann_info['flip_pairs'] = [[0, 5], [1, 4], [2, 3], [6, 11],
                                       [7, 10], [8, 9], [20, 21], [22, 23]]

        # origin_part:  [0, 1, 2, 3, 4, 5, 6,  7,  8, 9, 10,11, 12, 13,
        # 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        # flipped_part: [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6,  12, 13,
        # 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]

        self.ann_info['use_different_joints_weight'] = False
        self.ann_info['joints_weight'] =  \
            np.ones(24, dtype=np.float32).reshape(
                (self.ann_info['num_joints'], 1))

        self.ann_info['uv_type'] = data_cfg['uv_type']
        self.ann_info['use_IUV'] = data_cfg['use_IUV']
        self.iuv_prefix = os.path.join(
            self.img_prefix, '{}_IUV_gt'.format(self.ann_info['uv_type']))
        self.db = self._get_db(ann_file)

    def _get_db(self, ann_file):
        """Load dataset."""
        data = np.load(ann_file)
        tmpl = dict(
            image_file=None,
            center=None,
            scale=None,
            rotation=0,
            joints_2d=None,
            joints_2d_visible=None,
            joints_3d=None,
            joints_3d_visible=None,
            gender=None,
            pose=None,
            beta=None,
            has_smpl=0,
            iuv_file=None,
            has_iuv=0,
            dataset='lsp')
        gt_db = []

        _imgnames = data['imgname']
        _scales = data['scale']
        _centers = data['center']
        dataset_len = len(_imgnames)

        # Get 2D keypoints
        if 'part' in data.keys():
            _keypoints = data['part']
        else:
            _keypoints = np.zeros((dataset_len, 24, 3), dtype=np.float)

        # Get gt 3D joints, if available
        if 'S' in data.keys():
            _joints_3d = data['S']
        else:
            _joints_3d = np.zeros((dataset_len, 24, 4), dtype=np.float)

        # Get gt SMPL parameters, if available
        if 'pose' in data.keys() and 'shape' in data.keys():
            _poses = data['pose'].astype(np.float)
            _betas = data['shape'].astype(np.float)
            has_smpl = 1
        else:
            _poses = np.zeros((dataset_len, 72), dtype=np.float)
            _betas = np.zeros((dataset_len, 10), dtype=np.float)
            has_smpl = 0

        # Get gender data, if available
        if 'gender' in data.keys():
            _genders = data['gender']
            _genders = np.array([0 if str(g) == 'm' else 1
                                 for g in _genders]).astype(np.int32)
        else:
            _genders = -1 * np.ones(dataset_len).astype(np.int32)

        # Get IUV image, if available
        if 'iuv_names' in data.keys():
            _iuv_names = data['iuv_names']
            has_iuv = has_smpl
        else:
            _iuv_names = [''] * dataset_len
            has_iuv = 0

        for i in range(len(_imgnames)):
            newitem = cp.deepcopy(tmpl)
            newitem['image_file'] = os.path.join(self.img_prefix, _imgnames[i])
            # newitem['scale'] = scales_[i].item()
            newitem['scale'] = self.ann_info['image_size'] / _scales[i].item(
            ) / 200.0
            newitem['center'] = _centers[i]

            newitem['joints_2d'] = _keypoints[i, :, :2]
            newitem['joints_2d_visible'] = _keypoints[i, :, -1][:, np.newaxis]
            newitem['joints_3d'] = _joints_3d[i, :, :3]
            newitem['joints_3d_visible'] = _keypoints[i, :, -1][:, np.newaxis]
            newitem['pose'] = _poses[i]
            newitem['beta'] = _betas[i]
            newitem['has_smpl'] = has_smpl
            newitem['gender'] = _genders[i]
            newitem['iuv_file'] = os.path.join(self.iuv_prefix, _iuv_names[i])
            newitem['has_iuv'] = has_iuv
            gt_db.append(newitem)
        return gt_db

    # We do NOT use lsp dataset for evaluation
    def evaluate(self, outputs, res_folder, metric='joint_error', **kwargs):
        """We do NOT evaluate on LSP dataset."""
        return 0
