import copy
import numpy as np

from mmpose.datasets import BaseCocoStyleDataset
from mmpose.registry import DATASETS

from typing import Optional, Sequence, Union, List, Callable


@DATASETS.register_module()
class LineMODKeypointCocoDataset(BaseCocoStyleDataset):
    """Dataset for LineMOD with COCO keypoint style"""
    def __init__(self,
                 *args,
                 background_path: str = '',
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.background_path = background_path
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format."""
        img = raw_data_info['raw_img_info']
        ann = raw_data_info['raw_ann_info']
        
        img_path = img['img_path']
        mask_path = img_path.replace('rgb', 'mask')
        
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
            'img_path': img_path,
            'mask_path': mask_path,
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