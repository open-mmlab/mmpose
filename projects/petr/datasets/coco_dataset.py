import copy
from itertools import filterfalse, groupby
from typing import Dict, List, Optional

import numpy as np
from mmdet.registry import DATASETS

from mmpose.datasets import CocoDataset as MMPoseCocoDataset


@DATASETS.register_module(force=True)
class CocoDataset(MMPoseCocoDataset):

    def parse_data_info(self, raw_data_info: dict) -> Optional[dict]:
        """Parse raw COCO annotation of an instance. Compared with original
        implementation, this method add image width and height in `data_info`

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

        # filter invalid instance
        if 'bbox' not in ann or 'keypoints' not in ann:
            return None

        img_w, img_h = img['width'], img['height']

        # get bbox in shape [1, 4], formatted as xywh
        x, y, w, h = ann['bbox']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)

        bbox = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)

        # keypoints in shape [1, K, 2] and keypoints_visible in [1, K]
        _keypoints = np.array(
            ann['keypoints'], dtype=np.float32).reshape(1, -1, 3)
        keypoints = _keypoints[..., :2]
        keypoints_visible = np.minimum(1, _keypoints[..., 2])

        if 'num_keypoints' in ann:
            num_keypoints = ann['num_keypoints']
        else:
            num_keypoints = np.count_nonzero(keypoints.max(axis=2))

        data_info = {
            'img_id': ann['image_id'],
            'img_path': img['img_path'],
            # Addition: add image width and height
            'width': img_w,
            'height': img_h,
            'area': ann['area'],
            'bbox': bbox,
            'bbox_score': np.ones(1, dtype=np.float32),
            'num_keypoints': num_keypoints,
            'keypoints': keypoints,
            'keypoints_visible': keypoints_visible,
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', None),
            'id': ann['id'],
            'category_id': ann['category_id'],
            # store the raw annotation of the instance
            # it is useful for evaluation without providing ann_file
            'raw_ann_info': copy.deepcopy(ann),
        }

        if 'crowdIndex' in img:
            data_info['crowd_index'] = img['crowdIndex']

        return data_info

    def _get_bottomup_data_infos(self, instance_list: List[Dict],
                                 image_list: List[Dict]) -> List[Dict]:
        """Organize the data list in bottom-up mode."""

        # bottom-up data list
        data_list_bu = []

        used_img_ids = set()

        # group instances by img_id
        for img_id, data_infos in groupby(instance_list,
                                          lambda x: x['img_id']):
            used_img_ids.add(img_id)
            data_infos = list(data_infos)

            # image data
            img_path = data_infos[0]['img_path']
            data_info_bu = {
                'img_id': img_id,
                'img_path': img_path,
                'width': data_infos[0]['width'],
                'height': data_infos[0]['height'],
            }

            for key in data_infos[0].keys():
                if key not in data_info_bu:
                    seq = [d[key] for d in data_infos]
                    if isinstance(seq[0], np.ndarray):
                        seq = np.concatenate(seq, axis=0)
                    seq = np.array(seq) if key == 'area' else seq
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
                        'img_id': img_info['img_id'],
                        'img_path': img_info['img_path'],
                        'id': list(),
                        'raw_ann_info': None,
                    }
                    data_list_bu.append(data_info_bu)

        return data_list_bu
