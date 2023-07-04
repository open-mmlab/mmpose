# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from copy import deepcopy
from itertools import filterfalse, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.fileio import exists, get_local_path, load
from mmengine.utils import is_abs
from PIL import Image

from mmpose.registry import DATASETS
from ..utils import parse_pose_metainfo


@DATASETS.register_module()
class BaseMocapDataset(BaseDataset):
    """Base class for 3d body datasets.

    Args:
        ann_file (str): Annotation file path. Default: ''.
        seq_len (int): Number of frames in a sequence. Default: 1.
        causal (bool): If set to ``True``, the rightmost input frame will be
            the target frame. Otherwise, the middle input frame will be the
            target frame. Default: ``True``.
        subset_frac (float): The fraction to reduce dataset size. If set to 1,
            the dataset size is not reduced. Default: 1.
        camera_param_file (str): Cameras' parameters file. Default: ``None``.
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
    """

    METAINFO: dict = dict()

    def __init__(self,
                 ann_file: str = '',
                 seq_len: int = 1,
                 causal: bool = True,
                 subset_frac: float = 1.0,
                 camera_param_file: Optional[str] = None,
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
                 max_refetch: int = 1000):

        if data_mode not in {'topdown', 'bottomup'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid data_mode: '
                f'{data_mode}. Should be "topdown" or "bottomup".')
        self.data_mode = data_mode

        _ann_file = ann_file
        if not is_abs(_ann_file):
            _ann_file = osp.join(data_root, _ann_file)
        assert exists(_ann_file), 'Annotation file does not exist.'
        with get_local_path(_ann_file) as local_path:
            self.ann_data = np.load(local_path)

        self.camera_param_file = camera_param_file
        if self.camera_param_file:
            if not is_abs(self.camera_param_file):
                self.camera_param_file = osp.join(data_root,
                                                  self.camera_param_file)
            assert exists(self.camera_param_file)
            self.camera_param = load(self.camera_param_file)

        self.seq_len = seq_len
        self.causal = causal

        assert 0 < subset_frac <= 1, (
            f'Unsupported `subset_frac` {subset_frac}. Supported range '
            'is (0, 1].')
        self.subset_frac = subset_frac

        self.sequence_indices = self.get_sequence_indices()

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (dict): Raw data of pose meta information.

        Returns:
            dict: Parsed meta information.
        """

        if metainfo is None:
            metainfo = deepcopy(cls.METAINFO)

        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        # parse pose metainfo if it has been assigned
        if metainfo:
            metainfo = parse_pose_metainfo(metainfo)
        return metainfo

    @force_full_init
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        :class:`BaseCocoStyleDataset` overrides this method from
        :class:`mmengine.dataset.BaseDataset` to add the metainfo into
        the ``data_info`` before it is passed to the pipeline.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)

        return self.pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get data info by index.

        Args:
            idx (int): Index of data info.

        Returns:
            dict: Data info.
        """
        data_info = super().get_data_info(idx)

        # Add metainfo items that are required in the pipeline and the model
        metainfo_keys = [
            'upper_body_ids', 'lower_body_ids', 'flip_pairs',
            'dataset_keypoint_weights', 'flip_indices', 'skeleton_links'
        ]

        for key in metainfo_keys:
            assert key not in data_info, (
                f'"{key}" is a reserved key for `metainfo`, but already '
                'exists in the `data_info`.')

            data_info[key] = deepcopy(self._metainfo[key])

        return data_info

    def load_data_list(self) -> List[dict]:
        """Load data list from COCO annotation file or person detection result
        file."""

        instance_list, image_list = self._load_annotations()

        if self.data_mode == 'topdown':
            data_list = self._get_topdown_data_infos(instance_list)
        else:
            data_list = self._get_bottomup_data_infos(instance_list,
                                                      image_list)

        return data_list

    def get_img_info(self, img_idx, img_name):
        try:
            with get_local_path(osp.join(self.data_prefix['img'],
                                         img_name)) as local_path:
                im = Image.open(local_path)
                w, h = im.size
                im.close()
        except:  # noqa: E722
            return None

        img = {
            'file_name': img_name,
            'height': h,
            'width': w,
            'id': img_idx,
            'img_id': img_idx,
            'img_path': osp.join(self.data_prefix['img'], img_name),
        }
        return img

    def get_sequence_indices(self) -> List[List[int]]:
        """Build sequence indices.

        The default method creates sample indices that each sample is a single
        frame (i.e. seq_len=1). Override this method in the subclass to define
        how frames are sampled to form data samples.

        Outputs:
            sample_indices: the frame indices of each sample.
                For a sample, all frames will be treated as an input sequence,
                and the ground-truth pose of the last frame will be the target.
        """
        sequence_indices = []
        if self.seq_len == 1:
            num_imgs = len(self.ann_data['imgname'])
            sequence_indices = [[idx] for idx in range(num_imgs)]
        else:
            raise NotImplementedError('Multi-frame data sample unsupported!')
        return sequence_indices

    def _load_annotations(self) -> Tuple[List[dict], List[dict]]:
        """Load data from annotations in COCO format."""
        num_keypoints = self.metainfo['num_keypoints']

        img_names = self.ann_data['imgname']
        num_imgs = len(img_names)

        if 'S' in self.ann_data.keys():
            kpts_3d = self.ann_data['S']
        else:
            kpts_3d = np.zeros((num_imgs, num_keypoints, 4), dtype=np.float32)

        if 'part' in self.ann_data.keys():
            kpts_2d = self.ann_data['part']
        else:
            kpts_2d = np.zeros((num_imgs, num_keypoints, 3), dtype=np.float32)

        if 'center' in self.ann_data.keys():
            centers = self.ann_data['center']
        else:
            centers = np.zeros((num_imgs, 2), dtype=np.float32)

        if 'scale' in self.ann_data.keys():
            scales = self.ann_data['scale'].astype(np.float32)
        else:
            scales = np.zeros(num_imgs, dtype=np.float32)

        instance_list = []
        image_list = []

        for idx, frame_ids in enumerate(self.sequence_indices):
            assert len(frame_ids) == self.seq_len

            _img_names = img_names[frame_ids]

            _keypoints = kpts_2d[frame_ids].astype(np.float32)
            keypoints = _keypoints[..., :2]
            keypoints_visible = _keypoints[..., 2]

            _keypoints_3d = kpts_3d[frame_ids].astype(np.float32)
            keypoints_3d = _keypoints_3d[..., :3]
            keypoints_3d_visible = _keypoints_3d[..., 3]

            target_idx = -1 if self.causal else int(self.seq_len) // 2

            instance_info = {
                'num_keypoints': num_keypoints,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'keypoints_3d': keypoints_3d,
                'keypoints_3d_visible': keypoints_3d_visible,
                'scale': scales[idx],
                'center': centers[idx].astype(np.float32).reshape(1, -1),
                'id': idx,
                'category_id': 1,
                'iscrowd': 0,
                'img_paths': list(_img_names),
                'img_ids': frame_ids,
                'lifting_target': keypoints_3d[target_idx],
                'lifting_target_visible': keypoints_3d_visible[target_idx],
                'target_img_path': _img_names[target_idx],
            }

            if self.camera_param_file:
                _cam_param = self.get_camera_param(_img_names[0])
                instance_info['camera_param'] = _cam_param

            instance_list.append(instance_info)

        for idx, imgname in enumerate(img_names):
            img_info = self.get_img_info(idx, imgname)
            image_list.append(img_info)

        return instance_list, image_list

    def get_camera_param(self, imgname):
        """Get camera parameters of a frame by its image name.

        Override this method to specify how to get camera parameters.
        """
        raise NotImplementedError

    @staticmethod
    def _is_valid_instance(data_info: Dict) -> bool:
        """Check a data info is an instance with valid bbox and keypoint
        annotations."""
        # crowd annotation
        if 'iscrowd' in data_info and data_info['iscrowd']:
            return False
        # invalid keypoints
        if 'num_keypoints' in data_info and data_info['num_keypoints'] == 0:
            return False
        # invalid keypoints
        if 'keypoints' in data_info:
            if np.max(data_info['keypoints']) <= 0:
                return False
        return True

    def _get_topdown_data_infos(self, instance_list: List[Dict]) -> List[Dict]:
        """Organize the data list in top-down mode."""
        # sanitize data samples
        data_list_tp = list(filter(self._is_valid_instance, instance_list))

        return data_list_tp

    def _get_bottomup_data_infos(self, instance_list: List[Dict],
                                 image_list: List[Dict]) -> List[Dict]:
        """Organize the data list in bottom-up mode."""

        # bottom-up data list
        data_list_bu = []

        used_img_ids = set()

        # group instances by img_id
        for img_ids, data_infos in groupby(instance_list,
                                           lambda x: x['img_ids']):
            for img_id in img_ids:
                used_img_ids.add(img_id)
            data_infos = list(data_infos)

            # image data
            img_paths = data_infos[0]['img_paths']
            data_info_bu = {
                'img_ids': img_ids,
                'img_paths': img_paths,
            }

            for key in data_infos[0].keys():
                if key not in data_info_bu:
                    seq = [d[key] for d in data_infos]
                    if isinstance(seq[0], np.ndarray):
                        seq = np.concatenate(seq, axis=0)
                    data_info_bu[key] = seq

            # The segmentation annotation of invalid objects will be used
            # to generate valid region mask in the pipeline.
            invalid_segs = []
            for data_info_invalid in filterfalse(self._is_valid_instance,
                                                 data_infos):
                if 'segmentation' in data_info_invalid:
                    invalid_segs.append(data_info_invalid['segmentation'])
            data_info_bu['invalid_segs'] = invalid_segs

            data_list_bu.append(data_info_bu)

        # add images without instance for evaluation
        if self.test_mode:
            for img_info in image_list:
                if img_info['img_id'] not in used_img_ids:
                    data_info_bu = {
                        'img_ids': [img_info['img_id']],
                        'img_path': [img_info['img_path']],
                        'id': list(),
                    }
                    data_list_bu.append(data_info_bu)

        return data_list_bu
