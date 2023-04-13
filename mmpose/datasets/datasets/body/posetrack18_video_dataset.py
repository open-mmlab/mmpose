# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from mmengine.fileio import exists, get_local_path, load
from mmengine.utils import is_list_of
from xtcocotools.coco import COCO

from mmpose.registry import DATASETS
from mmpose.structures.bbox import bbox_xywh2xyxy
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class PoseTrack18VideoDataset(BaseCocoStyleDataset):
    """PoseTrack18 dataset for video pose estimation.

    "Posetrack: A benchmark for human pose estimation and tracking", CVPR'2018.
    More details can be found in the `paper
    <https://arxiv.org/abs/1710.10000>`__ .

    PoseTrack2018 keypoints::

        0: 'nose',
        1: 'head_bottom',
        2: 'head_top',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        frame_weights (List[Union[int, float]] ): The weight of each frame
            for aggregation. The first weight is for the center frame, then on
            ascending order of frame indices. Note that the length of
            ``frame_weights`` should be consistent with the number of sampled
            frames. Default: [0.0, 1.0]
        frame_sampler_mode (str): Specifies the mode of frame sampler:
            ``'fixed'`` or ``'random'``. In ``'fixed'`` mode, each frame
            index relative to the center frame is fixed, specified by
            ``frame_indices``, while in ``'random'`` mode, each frame index
            relative to the center frame is sampled from ``frame_range``
            with certain randomness. Default: ``'random'``.
        frame_range (int | List[int], optional): The sampling range of
            supporting frames in the same video for center frame.
            Only valid when ``frame_sampler_mode`` is ``'random'``.
            Default: ``None``.
        num_sampled_frame(int, optional): The number of sampled frames, except
            the center frame. Only valid when ``frame_sampler_mode`` is
            ``'random'``. Default: 1.
        frame_indices (Sequence[int], optional): The sampled frame indices,
            including the center frame indicated by 0. Only valid when
            ``frame_sampler_mode`` is ``'fixed'``. Default: ``None``.
        ph_fill_len (int): The length of the placeholder to fill in the
            image filenames.  Default: 6
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img='')``.
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

    METAINFO: dict = dict(from_file='configs/_base_/datasets/posetrack18.py')

    def __init__(self,
                 ann_file: str = '',
                 bbox_file: Optional[str] = None,
                 data_mode: str = 'topdown',
                 frame_weights: List[Union[int, float]] = [0.0, 1.0],
                 frame_sampler_mode: str = 'random',
                 frame_range: Optional[Union[int, List[int]]] = None,
                 num_sampled_frame: Optional[int] = None,
                 frame_indices: Optional[Sequence[int]] = None,
                 ph_fill_len: int = 6,
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
        assert sum(frame_weights) == 1, 'Invalid `frame_weights`: should sum'\
            f' to 1.0, but got {frame_weights}.'
        for weight in frame_weights:
            assert weight >= 0, 'frame_weight can not be a negative value.'
        self.frame_weights = np.array(frame_weights)

        if frame_sampler_mode not in {'fixed', 'random'}:
            raise ValueError(
                f'{self.__class__.__name__} got invalid frame_sampler_mode: '
                f'{frame_sampler_mode}. Should be `"fixed"` or `"random"`.')
        self.frame_sampler_mode = frame_sampler_mode

        if frame_sampler_mode == 'random':
            assert frame_range is not None, \
                '`frame_sampler_mode` is set as `random`, ' \
                'please specify the `frame_range`.'

            if isinstance(frame_range, int):
                assert frame_range >= 0, \
                    'frame_range can not be a negative value.'
                self.frame_range = [-frame_range, frame_range]

            elif isinstance(frame_range, Sequence):
                assert len(frame_range) == 2, 'The length must be 2.'
                assert frame_range[0] <= 0 and frame_range[
                    1] >= 0 and frame_range[1] > frame_range[
                        0], 'Invalid `frame_range`'
                for i in frame_range:
                    assert isinstance(i, int), 'Each element must be int.'
                self.frame_range = frame_range
            else:
                raise TypeError(
                    f'The type of `frame_range` must be int or Sequence, '
                    f'but got {type(frame_range)}.')

            assert num_sampled_frame is not None, \
                '`frame_sampler_mode` is set as `random`, please specify ' \
                '`num_sampled_frame`, e.g. the number of sampled frames.'

            assert len(frame_weights) == num_sampled_frame + 1, \
                f'the length of frame_weights({len(frame_weights)}) '\
                f'does not match the number of sampled adjacent '\
                f'frames({num_sampled_frame})'
            self.frame_indices = None
            self.num_sampled_frame = num_sampled_frame

        if frame_sampler_mode == 'fixed':
            assert frame_indices is not None, \
                '`frame_sampler_mode` is set as `fixed`, ' \
                'please specify the `frame_indices`.'
            assert len(frame_weights) == len(frame_indices), \
                f'the length of frame_weights({len(frame_weights)}) does not '\
                f'match the length of frame_indices({len(frame_indices)}).'
            frame_indices.sort()
            self.frame_indices = frame_indices
            self.frame_range = None
            self.num_sampled_frame = None

        self.ph_fill_len = ph_fill_len

        super().__init__(
            ann_file=ann_file,
            bbox_file=bbox_file,
            data_mode=data_mode,
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

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw annotation of an instance.

        Args:
            raw_data_info (dict): Raw data information loaded from
                ``ann_file``. It should have following contents:

                - ``'raw_ann_info'``: Raw annotation of an instance
                - ``'raw_img_info'``: Raw information of the image that
                    contains the instance

        Returns:
            dict: Parsed instance annotation
        """

        ann = raw_data_info['raw_ann_info']
        img = raw_data_info['raw_img_info']

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann or max(
                ann['keypoints']) == 0:
            return None

        img_w, img_h = img['width'], img['height']
        # get the bbox of the center frame
        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # get the keypoints of the center frame
        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        # deal with multiple image paths
        img_paths: list = []
        # get the image path of the center frame
        center_img_path = osp.join(self.data_prefix['img'], img['file_name'])
        # append the center image path first
        img_paths.append(center_img_path)

        # select the frame indices
        if self.frame_sampler_mode == 'fixed':
            indices = self.frame_indices
        else:  # self.frame_sampler_mode == 'random':
            low, high = self.frame_range
            indices = np.random.randint(low, high + 1, self.num_sampled_frame)

        nframes = int(img['nframes'])
        file_name = img['file_name']
        ref_idx = int(osp.splitext(osp.basename(file_name))[0])

        for idx in indices:
            if self.test_mode and idx == 0:
                continue
            # the supporting frame index
            support_idx = ref_idx + idx
            # clip the frame index to make sure that it does not exceed
            # the boundings of frame indices
            support_idx = np.clip(support_idx, 0, nframes - 1)
            sup_img_path = osp.join(
                osp.dirname(center_img_path),
                str(support_idx).zfill(self.ph_fill_len) + '.jpg')

            img_paths.append(sup_img_path)

        data_info = {
            'img_id': int(img['frame_id']),
            'img_path': img_paths,
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': ann['num_keypoints'],
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'frame_weights': self.frame_weights,
            'id': ann['id'],
        }

        return data_info

    def _load_detection_results(self) -> List[dict]:
        """Load data from detection results with dummy keypoint annotations."""
        assert exists(self.ann_file), 'Annotation file does not exist'
        assert exists(self.bbox_file), 'Bbox file does not exist'

        # load detection results
        det_results = load(self.bbox_file)
        assert is_list_of(det_results, dict)

        # load coco annotations to build image id-to-name index
        with get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)

        # mapping image name to id
        name2id = {}
        # mapping image id to name
        id2name = {}
        for img_id, image in self.coco.imgs.items():
            file_name = image['file_name']
            id2name[img_id] = file_name
            name2id[file_name] = img_id

        num_keypoints = self.metainfo['num_keypoints']
        data_list = []
        id_ = 0
        for det in det_results:
            # remove non-human instances
            if det['category_id'] != 1:
                continue

            # get the predicted bbox and bbox_score
            bbox_xywh = np.array(
                det['bbox'][:4], dtype=np.float32).reshape(1, 4)
            bbox = bbox_xywh2xyxy(bbox_xywh)
            bbox_score = np.array(det['score'], dtype=np.float32).reshape(1)

            # use dummy keypoint location and visibility
            keypoints = np.zeros((1, num_keypoints, 2), dtype=np.float32)
            keypoints_visible = np.ones((1, num_keypoints), dtype=np.float32)

            # deal with different bbox file formats
            if 'nframes' in det:
                nframes = int(det['nframes'])
            else:
                if 'image_name' in det:
                    img_id = name2id[det['image_name']]
                else:
                    img_id = det['image_id']
                img_ann = self.coco.loadImgs(img_id)[0]
                nframes = int(img_ann['nframes'])

            # deal with multiple image paths
            img_paths: list = []
            if 'image_name' in det:
                image_name = det['image_name']
            else:
                image_name = id2name[det['image_id']]
            # get the image path of the center frame
            center_img_path = osp.join(self.data_prefix['img'], image_name)
            # append the center image path first
            img_paths.append(center_img_path)

            # "images/val/012834_mpii_test/000000.jpg" -->> "000000.jpg"
            center_image_name = image_name.split('/')[-1]
            ref_idx = int(center_image_name.replace('.jpg', ''))

            # select the frame indices
            if self.frame_sampler_mode == 'fixed':
                indices = self.frame_indices
            else:  # self.frame_sampler_mode == 'random':
                low, high = self.frame_range
                indices = np.random.randint(low, high + 1,
                                            self.num_sampled_frame)

            for idx in indices:
                if self.test_mode and idx == 0:
                    continue
                # the supporting frame index
                support_idx = ref_idx + idx
                # clip the frame index to make sure that it does not exceed
                # the boundings of frame indices
                support_idx = np.clip(support_idx, 0, nframes - 1)
                sup_img_path = center_img_path.replace(
                    center_image_name,
                    str(support_idx).zfill(self.ph_fill_len) + '.jpg')

                img_paths.append(sup_img_path)

            data_list.append({
                'img_id': det['image_id'],
                'img_path': img_paths,
                'frame_weights': self.frame_weights,
                'bbox': bbox,
                'bbox_score': bbox_score,
                'keypoints': keypoints,
                'keypoints_visible': keypoints_visible,
                'id': id_,
            })

            id_ += 1

        return data_list
