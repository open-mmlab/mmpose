# Copyright (c) OpenMMLab. All rights reserved.
import copy
import json
import os.path as osp
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.fileio import exists, get_local_path
from mmengine.utils import is_abs
from xtcocotools.coco import COCO

from mmpose.codecs.utils import camera_to_pixel
from mmpose.datasets.datasets import BaseCocoStyleDataset
from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy


@DATASETS.register_module()
class InterHand2DDoubleDataset(BaseCocoStyleDataset):
    """InterHand2.6M dataset for 2d double hands.

    "InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose
    Estimation from a Single RGB Image", ECCV'2020.
    More details can be found in the `paper
    <https://arxiv.org/pdf/2008.09309.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    InterHand2.6M keypoint indexes::

        0: 'r_thumb4',
        1: 'r_thumb3',
        2: 'r_thumb2',
        3: 'r_thumb1',
        4: 'r_index4',
        5: 'r_index3',
        6: 'r_index2',
        7: 'r_index1',
        8: 'r_middle4',
        9: 'r_middle3',
        10: 'r_middle2',
        11: 'r_middle1',
        12: 'r_ring4',
        13: 'r_ring3',
        14: 'r_ring2',
        15: 'r_ring1',
        16: 'r_pinky4',
        17: 'r_pinky3',
        18: 'r_pinky2',
        19: 'r_pinky1',
        20: 'r_wrist',
        21: 'l_thumb4',
        22: 'l_thumb3',
        23: 'l_thumb2',
        24: 'l_thumb1',
        25: 'l_index4',
        26: 'l_index3',
        27: 'l_index2',
        28: 'l_index1',
        29: 'l_middle4',
        30: 'l_middle3',
        31: 'l_middle2',
        32: 'l_middle1',
        33: 'l_ring4',
        34: 'l_ring3',
        35: 'l_ring2',
        36: 'l_ring1',
        37: 'l_pinky4',
        38: 'l_pinky3',
        39: 'l_pinky2',
        40: 'l_pinky1',
        41: 'l_wrist'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        camera_param_file (str): Cameras' parameters file. Default: ''.
        joint_file (str): Path to the joint file. Default: ''.
        use_gt_root_depth (bool): Using the ground truth depth of the wrist
            or given depth from rootnet_result_file. Default: ``True``.
        rootnet_result_file (str): Path to the wrist depth file.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data.
            Default: ``dict(img='')``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
        sample_interval (int, optional): The sample interval of the dataset.
            Default: 1.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/interhand3d.py')

    def __init__(self,
                 ann_file: str = '',
                 camera_param_file: str = '',
                 joint_file: str = '',
                 use_gt_root_depth: bool = True,
                 rootnet_result_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 sample_interval: int = 1):
        _ann_file = ann_file
        if data_root is not None and not is_abs(_ann_file):
            _ann_file = osp.join(data_root, _ann_file)
        assert exists(_ann_file), 'Annotation file does not exist.'
        self.ann_file = _ann_file

        _camera_param_file = camera_param_file
        if data_root is not None and not is_abs(_camera_param_file):
            _camera_param_file = osp.join(data_root, _camera_param_file)
        assert exists(_camera_param_file), 'Camera file does not exist.'
        self.camera_param_file = _camera_param_file

        _joint_file = joint_file
        if data_root is not None and not is_abs(_joint_file):
            _joint_file = osp.join(data_root, _joint_file)
        assert exists(_joint_file), 'Joint file does not exist.'
        self.joint_file = _joint_file

        self.use_gt_root_depth = use_gt_root_depth
        if not self.use_gt_root_depth:
            assert rootnet_result_file is not None
            _rootnet_result_file = rootnet_result_file
            if data_root is not None and not is_abs(_rootnet_result_file):
                _rootnet_result_file = osp.join(data_root,
                                                _rootnet_result_file)
            assert exists(
                _rootnet_result_file), 'Rootnet result file does not exist.'
            self.rootnet_result_file = _rootnet_result_file

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_mode=data_mode,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            sample_interval=sample_interval)

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""

        assert exists(self.ann_file), 'Annotation file does not exist'

        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        # set the metainfo about categories, which is a list of dict
        # and each dict contains the 'id', 'name', etc. about this category
        if 'categories' in self.coco.dataset:
            self._metainfo['CLASSES'] = self.coco.loadCats(
                self.coco.getCatIds())

        with get_local_path(self.camera_param_file) as local_path:
            with open(local_path, 'r') as f:
                self.cameras = json.load(f)
        with get_local_path(self.joint_file) as local_path:
            with open(local_path, 'r') as f:
                self.joints = json.load(f)

        instance_list = []
        image_list = []

        for idx, img_id in enumerate(self.coco.getImgIds()):
            if idx % self.sample_interval != 0:
                continue
            img = self.coco.loadImgs(img_id)[0]
            img.update({
                'img_id':
                img_id,
                'img_path':
                osp.join(self.data_prefix['img'], img['file_name']),
            })
            image_list.append(img)

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            ann = self.coco.loadAnns(ann_ids)[0]

            instance_info = self.parse_data_info(
                dict(raw_ann_info=ann, raw_img_info=img))

            # skip invalid instance annotation.
            if not instance_info:
                continue

            instance_list.append(instance_info)
        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict | None: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        if not self.use_gt_root_depth:
            rootnet_result = {}
            with get_local_path(self.rootnet_result_file) as local_path:
                rootnet_annot = json.load(local_path)
            for i in range(len(rootnet_annot)):
                rootnet_result[str(
                    rootnet_annot[i]['annot_id'])] = rootnet_annot[i]

        num_keypoints = self.metainfo['num_keypoints']

        capture_id = str(img['capture'])
        camera_name = img['camera']
        frame_idx = str(img['frame_idx'])
        camera_pos = np.array(
            self.cameras[capture_id]['campos'][camera_name], dtype=np.float32)
        camera_rot = np.array(
            self.cameras[capture_id]['camrot'][camera_name], dtype=np.float32)
        focal = np.array(
            self.cameras[capture_id]['focal'][camera_name], dtype=np.float32)
        principal_pt = np.array(
            self.cameras[capture_id]['princpt'][camera_name], dtype=np.float32)
        joint_world = np.array(
            self.joints[capture_id][frame_idx]['world_coord'],
            dtype=np.float32)
        joint_valid = np.array(ann['joint_valid'], dtype=np.float32).flatten()

        keypoints_cam = np.dot(
            camera_rot,
            joint_world.transpose(1, 0) -
            camera_pos.reshape(3, 1)).transpose(1, 0)

        if self.use_gt_root_depth:
            bbox_xywh = np.array(ann['bbox'], dtype=np.float32).reshape(1, 4)

        else:
            rootnet_ann_data = rootnet_result[str(ann['id'])]
            bbox_xywh = np.array(
                rootnet_ann_data['bbox'], dtype=np.float32).reshape(1, 4)

        bbox = bbox_xywh2xyxy(bbox_xywh)

        # 41: 'l_wrist', left hand root
        # 20: 'r_wrist', right hand root

        # if root is not valid -> root-relative 3D pose is also not valid.
        # Therefore, mark all joints as invalid
        joint_valid[:20] *= joint_valid[20]
        joint_valid[21:] *= joint_valid[41]

        joints_3d_visible = np.minimum(1,
                                       joint_valid.reshape(-1,
                                                           1)).reshape(1, -1)
        keypoints_img = camera_to_pixel(
            keypoints_cam,
            focal[0],
            focal[1],
            principal_pt[0],
            principal_pt[1],
            shift=True)[..., :2]
        joints_3d = np.zeros((keypoints_cam.shape[-2], 3),
                             dtype=np.float32).reshape(1, -1, 3)
        joints_3d[..., :2] = keypoints_img
        joints_3d[..., :21,
                  2] = keypoints_cam[..., :21, 2] - keypoints_cam[..., 20, 2]
        joints_3d[..., 21:,
                  2] = keypoints_cam[..., 21:, 2] - keypoints_cam[..., 41, 2]

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            'keypoints': joints_3d[:, :, :2],
            'keypoints_visible': joints_3d_visible,
            'hand_type': self.encode_handtype(ann['hand_type']),
            'hand_type_valid': np.array([ann['hand_type_valid']]),
            'dataset': self.metainfo['dataset_name'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'iscrowd': ann.get('iscrowd', False),
            'id': ann['id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        return data_info

    @staticmethod
    def encode_handtype(hand_type):
        if hand_type == 'right':
            return np.array([[1, 0]], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([[0, 1]], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([[1, 1]], dtype=np.float32)
        else:
            assert 0, f'Not support hand type: {hand_type}'
