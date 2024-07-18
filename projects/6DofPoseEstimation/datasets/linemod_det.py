import os.path as osp
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

from typing import Optional, Sequence, Union, List, Callable

@DATASETS.register_module()
class LineMODDetCocoDataset(CocoDataset):
    """Dataset for LineMOD with COCO keypoint style"""
    def __init__(self,
                 *args,
                 background_path: str = '',
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.background_path = background_path
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format."""
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        
        data_info = {}
        
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        data_info['img_path'] = img_path
        
        mask_path = img_path.replace('rgb', 'mask')
        data_info['mask_path'] = mask_path
        
        data_info['img_id'] = img_info['img_id']
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        
        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info