# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import os.path as osp
from mmpose.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv import Config
from argparse import ArgumentParser
import mmcv
import numpy as np
from mmpose.apis.inference import init_pose_model
from mmpose.core.post_processing import get_affine_transform
import json
from glob import glob
from mmpose.datasets.dataset_info import DatasetInfo


def get_scale(target_size, raw_image_size):
    w, h = raw_image_size
    w_resized, h_resized = target_size
    if w / w_resized < h / h_resized:
        w_pad = h / h_resized * w_resized
        h_pad = h
    else:
        w_pad = w
        h_pad = w / w_resized * h_resized

    scale = np.array([w_pad, h_pad], dtype=np.float32)

    return scale


def get_panoptic_camera_parameters(cam_file, camera_names,
                                   M=[[1.0, 0.0, 0.0],
                                      [0.0, 0.0, -1.0],
                                      [0.0, 1.0, 0.0]]):
    with open(cam_file) as cfile:
        calib = json.load(cfile)

    M = np.array(M)
    cameras = {}
    for cam in calib['cameras']:
        if cam['name'] in camera_names:
            sel_cam = {}
            R_w2c = np.array(cam['R']).dot(M)
            T_w2c = np.array(cam['t']).reshape((3, 1)) * 10.0  # cm to mm
            R_c2w = R_w2c.T
            T_c2w = -R_w2c.T @ T_w2c
            sel_cam['R'] = R_c2w.tolist()
            sel_cam['T'] = T_c2w.tolist()
            sel_cam['K'] = cam['K'][:2]
            distCoef = cam['distCoef']
            sel_cam['k'] = [distCoef[0], distCoef[1], distCoef[4]]
            sel_cam['p'] = [distCoef[2], distCoef[3]]
            cameras[cam['name']] = sel_cam

    assert len(cameras) == len(camera_names)

    return cameras


def get_input_data(img_path, cam_file):
    camera_names = sorted([d for d in os.listdir(img_path)
                          if osp.isdir(osp.join(img_path, d))])
    directories = [osp.join(img_path, d) for d in camera_names]
    num_cameras = len(camera_names)
    # load camera parameters
    cameras = get_panoptic_camera_parameters(cam_file, camera_names)
    frames = [sorted(glob(osp.join(d, '*'))) for d in directories]
    frames = list(map(list, zip(*frames)))
    input_data = []
    sample_id = 0
    for frame in frames:
        assert len(frame) == num_cameras
        for i in range(num_cameras):
            single_view_camera = cameras[camera_names[i]].copy()
            image_file = frame[i]
            input_data.append({
                'image_file':
                    image_file,
                'camera':
                    single_view_camera,
                'sample_id':
                    sample_id,
            })
            sample_id += 1

    return input_data, num_cameras


def inference(args):
    config_dict = Config.fromfile(args.config_file)
    assert args.dataset == 'Body3DMviewDirectPanopticDataset'
    cfg = Config.fromfile('configs/_base_/datasets/panoptic_body3d.py')
    dataset_info = cfg._cfg_dict['dataset_info']
    dataset_info = DatasetInfo(dataset_info)

    model = init_pose_model(
            config_dict,
            args.pose_model_checkpoint,
            device=args.device.lower())
    pipeline = [
        dict(
            type='MultiItemProcess',
            pipeline=[
                dict(type='ToTensor'),
                dict(
                    type='NormalizeTensor',
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ]),
        dict(
            type='DiscardDuplicatedItems',
            keys_list=['sample_id']),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=['sample_id', 'camera', 'center', 'scale', 'image_file']),
    ]
    pipeline = Compose(pipeline)

    input_data, num_cameras = get_input_data(args.img_root, args.camera_param_file)
    num_frames = len(input_data) // num_cameras
    prog_bar = mmcv.ProgressBar(num_frames)

    for i in range(num_frames):
        multiview_data = {}
        image_infos = []
        for c in range(num_cameras):
            singleview_data = input_data[i * num_cameras + c]
            image_file = singleview_data['image_file']
            # load image
            file_client = mmcv.FileClient(backend='disk')
            img_bytes = file_client.get(image_file)
            img = mmcv.imfrombytes(
                img_bytes, flag='color', channel_order='rgb')
            # img = img.astype(np.float32)
            # get image scale
            height, width, _ = img.shape
            input_size = config_dict['model']['human_detector']['image_size']
            center = np.array((width / 2, height / 2), dtype=np.float32)
            scale = get_scale(input_size, (width, height))
            mat_input = get_affine_transform(
                center=center,
                scale=scale / 200.0,
                rot=0.0,
                output_size=input_size)
            img = cv2.warpAffine(img, mat_input, (int(input_size[0]), int(input_size[1])))
            image_infos.append(input_data[i * num_cameras + c])

            singleview_data['img'] = img
            singleview_data['center'] = center
            singleview_data['scale'] = scale
            multiview_data[c] = singleview_data

        multiview_data = pipeline(multiview_data)
        # TODO: inference with input_heatmaps/kpts_2d
        multiview_data = collate([multiview_data], samples_per_gpu=1)
        multiview_data = scatter(multiview_data, [args.device])[0]

        model.show_result(**multiview_data, input_heatmaps=None,
                          dataset_info=dataset_info, radius=args.radius,
                          thickness=args.thickness, out_dir=args.out_img_root,
                          show=args.show, visualize_2d=args.visualize_single_view)
        prog_bar.update()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'config_file',
        help='Config file pose model')
    parser.add_argument(
        'pose_model_checkpoint',
        help='Checkpoint file for pose model')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--out-img-root', type=str, default='', help='Output image root')
    parser.add_argument(
        '--camera-param-file',
        type=str,
        default=None,
        help='Camera parameter file for converting 3D pose predictions from '
        ' the camera space to to world space. If None, no conversion will be '
        'applied.')
    parser.add_argument(
        '--dataset',
        type=str,
        default='Body3DMviewDirectPanopticDataset')
    parser.add_argument(
        '--visualize-single-view',
        action='store_true',
        default=False,
        help='whether to visualize single view imgs')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=8,
        help='Link thickness for visualization')

    args = parser.parse_args()
    inference(args)
