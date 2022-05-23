# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import torch
from torchvision.transforms import functional as F

from mmpose.apis import init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo

try:
    import ffmpegcv
except ImportError:
    raise ImportError(
        'Please install the ffmpeg: \n\n    apt install ffmpeg \n\n'
        'And please install ffmpegcv with:\n\n    pip install ffmpegcv')


def box2cs(box, image_size):
    """Encode bbox(x,y,w,h) into (center, scale) without padding.

    Returns:
        tuple: A tuple containing center and scale.
    """
    x, y, w, h = box[:4]

    aspect_ratio = 1. * image_size[0] / image_size[1]
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / 200.0, h * 1.0 / 200.0], dtype=np.float32)
    return center, scale


def prefetch_img_metas(cfg, ori_wh):
    """Pre-fetch the img_metas from config and original image size.

    Return:
        dict: img_metas.
    """
    w, h = ori_wh
    bbox = np.array([0, 0, w, h])
    center, scale = box2cs(bbox, cfg.data_cfg['image_size'])
    dataset_info = cfg.data['test'].get('dataset_info', None)
    assert dataset_info, 'Please set `dataset_info` in the config.'
    img_metas = {
        'img_or_path':
        None,
        'img':
        None,
        'image_file':
        '',
        'center':
        center,
        'scale':
        scale,
        'bbox_score':
        1,
        'bbox_id':
        0,
        'dataset':
        dataset_info.dataset_name,
        'joints_3d':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'joints_3d_visible':
        np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
        'rotation':
        0,
        'ann_info': {
            'image_size': np.array(cfg.data_cfg['image_size']),
            'num_joints': cfg.data_cfg['num_joints'],
        },
        'flip_pairs':
        dataset_info.flip_pairs
    }
    for pipeline in cfg.test_pipeline[1:]:
        if pipeline['type'] == 'NormalizeTensor':
            img_metas['img_norm_cfg'] = {
                'mean': np.array(pipeline['mean']) * 255.0,
                'std': np.array(pipeline['std']) * 255.0
            }
            break
    else:
        raise Exception('NormalizeTensor is not found.')

    return img_metas


def process_img(frame_resize, img_metas, device):
    """Process the image.

    Cast the image to device and do normalization.
    """
    assert frame_resize.shape[1::-1] == tuple(
        img_metas['ann_info']['image_size'])
    frame_cuda = torch.from_numpy(frame_resize).to(device).float()
    frame_cuda = frame_cuda.permute(2, 0, 1)  # HWC to CHW
    mean = torch.from_numpy(img_metas['img_norm_cfg']['mean']).to(device)
    std = torch.from_numpy(img_metas['img_norm_cfg']['std']).to(device)
    frame_cuda = F.normalize(frame_cuda, mean=mean, std=std, inplace=True)
    frame_cuda = frame_cuda[None, :, :, :]  # NCHW
    data = {'img': frame_cuda, 'img_metas': [img_metas]}
    return data


def main():
    """Visualize the demo video with GPU acceleration.

    Using full frame to estimate the keypoints.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--nvdecode', action='store_true', help='Use NVIDIA decoder')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    pose_model.cfg.data['test']['dataset_info'] = dataset_info
    if args.nvdecode:
        VideoCapture = ffmpegcv.VideoCaptureNV
    else:
        VideoCapture = ffmpegcv.VideoCapture
    video_origin = VideoCapture(args.video_path)
    img_metas = prefetch_img_metas(pose_model.cfg,
                                   (video_origin.width, video_origin.height))
    resize_wh = pose_model.cfg.data_cfg['image_size']
    video_resize = VideoCapture(
        args.video_path,
        resize=resize_wh,
        resize_keepratio=True,
        pix_fmt='rgb24')

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (video_origin.width, video_origin.height)
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            video_origin.fps, size)

    with torch.no_grad():
        for frame_resize, frame_origin in zip(
                mmcv.track_iter_progress(video_resize), video_origin):

            # test a single image
            data = process_img(frame_resize, img_metas, args.device)
            pose_results = pose_model(
                return_loss=False, return_heatmap=False, **data)
            pose_results['keypoints'] = pose_results['preds'][0]
            del pose_results['preds']
            pose_results = [pose_results]

            # show the results
            vis_img = vis_pose_result(
                pose_model,
                frame_origin,
                pose_results,
                radius=args.radius,
                thickness=args.thickness,
                dataset=dataset,
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                show=False)

            if args.show:
                cv2.imshow('Image', vis_img)

            if save_out_video:
                videoWriter.write(vis_img)

            if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_origin.release()
    video_resize.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
