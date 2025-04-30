# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import numpy as np
from mmengine import Config

from mmpose.evaluation.functional import nms
from mmpose.registry import DATASETS
from mmpose.structures import bbox_xyxy2xywh
from mmpose.utils import register_all_modules

try:
    from mmdet.apis import DetInferencer
    has_mmdet = True
except ImportError:
    print('Please install mmdet to use this script!')
    has_mmdet = False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('det_config')
    parser.add_argument('det_weight')
    parser.add_argument('output', type=str)
    parser.add_argument(
        '--pose-config',
        default='configs/body_2d_keypoint/topdown_heatmap/'
        'coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py')
    parser.add_argument('--score-thr', default=0.1)
    parser.add_argument('--nms-thr', default=0.65)
    args = parser.parse_args()

    register_all_modules()

    config = Config.fromfile(args.pose_config)
    config.test_dataloader.dataset.data_mode = 'bottomup'
    config.test_dataloader.dataset.bbox_file = None
    test_set = DATASETS.build(config.test_dataloader.dataset)
    print(f'number of images: {len(test_set)}')

    detector = DetInferencer(args.det_config, args.det_weight)

    new_bbox_files = []
    for i in range(len(test_set)):
        data = test_set.get_data_info(i)
        image_id = data['img_id']
        img_path = data['img_path']
        result = detector(
            img_path,
            return_datasamples=True)['predictions'][0].pred_instances.numpy()
        bboxes = np.concatenate((result.bboxes, result.scores[:, None]),
                                axis=1)
        bboxes = bboxes[bboxes[..., -1] > args.score_thr]
        bboxes = bboxes[nms(bboxes, args.nms_thr)]
        scores = bboxes[..., -1].tolist()
        bboxes = bbox_xyxy2xywh(bboxes[..., :4]).tolist()

        for bbox, score in zip(bboxes, scores):
            new_bbox_files.append(
                dict(category_id=1, image_id=image_id, score=score, bbox=bbox))

    with open(args.output, 'w') as f:
        json.dump(new_bbox_files, f, indent='')


if __name__ == '__main__':
    if has_mmdet:
        main()
