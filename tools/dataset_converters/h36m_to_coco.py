# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from functools import wraps

import mmengine
import numpy as np
from PIL import Image

from mmpose.utils import SimpleCamera


def _keypoint_camera_to_world(keypoints,
                              camera_params,
                              image_name=None,
                              dataset='Body3DH36MDataset'):
    """Project 3D keypoints from the camera space to the world space.

    Args:
        keypoints (np.ndarray): 3D keypoints in shape [..., 3]
        camera_params (dict): Parameters for all cameras.
        image_name (str): The image name to specify the camera.
        dataset (str): The dataset type, e.g., Body3DH36MDataset.
    """
    cam_key = None
    if dataset == 'Body3DH36MDataset':
        subj, rest = osp.basename(image_name).split('_', 1)
        _, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        cam_key = (subj, camera)
    else:
        raise NotImplementedError

    camera = SimpleCamera(camera_params[cam_key])
    keypoints_world = keypoints.copy()
    keypoints_world[..., :3] = camera.camera_to_world(keypoints[..., :3])

    return keypoints_world


def _get_bbox_xywh(center, scale, w=200, h=200):
    w = w * scale
    h = h * scale
    x = center[0] - w / 2
    y = center[1] - h / 2
    return [x, y, w, h]


def mmcv_track_func(func):

    @wraps(func)
    def wrapped_func(args):
        return func(*args)

    return wrapped_func


@mmcv_track_func
def _get_img_info(img_idx, img_name, img_root):
    try:
        im = Image.open(osp.join(img_root, img_name))
        w, h = im.size
    except:  # noqa: E722
        return None

    img = {
        'file_name': img_name,
        'height': h,
        'width': w,
        'id': img_idx + 1,
    }
    return img


@mmcv_track_func
def _get_ann(idx, kpt_2d, kpt_3d, center, scale, imgname, camera_params):
    bbox = _get_bbox_xywh(center, scale)
    kpt_3d = _keypoint_camera_to_world(kpt_3d, camera_params, imgname)

    ann = {
        'id': idx + 1,
        'category_id': 1,
        'image_id': idx + 1,
        'iscrowd': 0,
        'bbox': bbox,
        'area': bbox[2] * bbox[3],
        'num_keypoints': 17,
        'keypoints': kpt_2d.reshape(-1).tolist(),
        'keypoints_3d': kpt_3d.reshape(-1).tolist()
    }

    return ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann-file', type=str, default='tests/data/h36m/test_h36m_body3d.npz')
    parser.add_argument(
        '--camera-param-file', type=str, default='tests/data/h36m/cameras.pkl')
    parser.add_argument('--img-root', type=str, default='tests/data/h36m')
    parser.add_argument(
        '--out-file', type=str, default='tests/data/h36m/h36m_coco.json')
    parser.add_argument('--full-img-name', action='store_true')

    args = parser.parse_args()

    h36m_data = np.load(args.ann_file)
    h36m_camera_params = mmengine.load(args.camera_param_file)
    h36m_coco = {}

    # categories
    h36m_cats = [{
        'supercategory':
        'person',
        'id':
        1,
        'name':
        'person',
        'keypoints': [
            'root (pelvis)', 'left_hip', 'left_knee', 'left_foot', 'right_hip',
            'right_knee', 'right_foot', 'spine', 'thorax', 'neck_base', 'head',
            'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder',
            'right_elbow', 'right_wrist'
        ],
        'skeleton': [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                     [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
                     [8, 14], [14, 15], [15, 16]],
    }]

    # images
    imgnames = h36m_data['imgname']
    if not args.full_img_name:
        imgnames = [osp.basename(fn) for fn in imgnames]
    tasks = [(idx, fn, args.img_root) for idx, fn in enumerate(imgnames)]

    h36m_imgs = mmengine.track_parallel_progress(
        _get_img_info, tasks, nproc=12)

    # annotations
    kpts_2d = h36m_data['part']
    kpts_3d = h36m_data['S']
    centers = h36m_data['center']
    scales = h36m_data['scale']
    tasks = [(idx, ) + args + (h36m_camera_params, )
             for idx, args in enumerate(
                 zip(kpts_2d, kpts_3d, centers, scales, imgnames))]

    h36m_anns = mmengine.track_parallel_progress(_get_ann, tasks, nproc=12)

    # remove invalid data
    h36m_imgs = [img for img in h36m_imgs if img is not None]
    h36m_img_ids = set([img['id'] for img in h36m_imgs])
    h36m_anns = [ann for ann in h36m_anns if ann['image_id'] in h36m_img_ids]

    h36m_coco = {
        'categories': h36m_cats,
        'images': h36m_imgs,
        'annotations': h36m_anns,
    }

    mmengine.dump(h36m_coco, args.out_file)


if __name__ == '__main__':
    main()
