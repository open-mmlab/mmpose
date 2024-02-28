# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import os
import os.path as osp

import cv2
import numpy as np


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def convert_wflw_to_coco(ann_file, out_file):
    annotations = []
    images = []
    files = []
    cnt = 0
    image_cnt = 0

    data_infos = open(ann_file).readlines()
    data_infos = [x.strip().split() for x in data_infos]
    for data in data_infos:
        file_name = data[-1]
        img_path = osp.join('data/wflw/WFLW_images', file_name)
        img = cv2.imread(img_path)

        keypoints = []

        coordinates = [data[i:i + 2] for i in range(0, 196, 2)]

        for coordinate in coordinates:
            x, y = coordinate[0], coordinate[1]
            x, y = float(x), float(y)
            keypoints.append([x, y, 1])
        keypoints = np.array(keypoints)

        x1, y1, _ = np.amin(keypoints, axis=0)
        x2, y2, _ = np.amax(keypoints, axis=0)
        w, h = x2 - x1, y2 - y1
        scale = math.ceil(max(w, h)) / 200
        w_new = w / scale
        h_new = h / scale
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        x1_new = center[0] - w_new / 2
        y1_new = center[1] - h_new / 2
        bbox = [x1_new, y1_new, w_new, h_new]

        image = {}
        # check if the image already exists
        if file_name in files:
            image = images[files.index(file_name)]
        else:
            image['id'] = image_cnt
            image['file_name'] = f'{file_name}'
            image['height'] = img.shape[0]
            image['width'] = img.shape[1]
            image_cnt = image_cnt + 1
            files.append(file_name)
            images.append(image)

        ann = {}
        ann['keypoints'] = keypoints.reshape(-1).tolist()
        ann['image_id'] = image['id']
        ann['id'] = cnt
        ann['num_keypoints'] = len(keypoints)
        ann['bbox'] = bbox
        ann['is_crowd'] = 0
        ann['area'] = w * h
        ann['category_id'] = 1
        ann['center'] = center
        ann['scale'] = scale
        annotations.append(ann)
        cnt = cnt + 1

    cocotype = {}
    cocotype['images'] = images
    cocotype['annotations'] = annotations
    cocotype['categories'] = [{
        'supercategory': 'person',
        'id': 1,
        'name': 'face',
        'keypoints': [],
        'skeleton': []
    }]

    json.dump(
        cocotype,
        open(out_file, 'w'),
        ensure_ascii=False,
        default=default_dump)
    print(f'done {out_file}')


if __name__ == '__main__':
    if not osp.exists('data/wflw/annotations'):
        os.makedirs('data/wflw/annotations')
    root_folder = 'data/wflw'
    ann_folder = f'{root_folder}/WFLW_annotations'
    for root, dirs, files in os.walk(ann_folder):
        for file in files:
            if not file.endswith('txt'):
                continue
            print(f'Processing {file}')
            sub_class = file.split('_')[-1].replace('.txt', '')
            if sub_class != 'train' and sub_class != 'test':
                out_file = f'face_landmarks_wflw_test_{sub_class}.json'
            else:
                out_file = f'face_landmarks_wflw_{sub_class}.json'
            convert_wflw_to_coco(f'{root}/{file}',
                                 f'{root_folder}/annotations/{out_file}')
