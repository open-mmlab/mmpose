# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import (PoseDataSample, merge_data_samples,
                               split_instances)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show result')
    parser.add_argument('--device', default='cpu', help='Device for inference')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()
    return args


def process_one_image(args, img, model, visualizer=None, show_interval=0):
    """Visualize predicted keypoints of one image."""
    # inference a single image
    pose_results = inference_topdown(model, img)
    # post-processing
    pose_results_2d = []
    for idx, res in enumerate(pose_results):
        pred_instances = res.pred_instances
        keypoints = pred_instances.keypoints
        rel_root_depth = pred_instances.rel_root_depth
        scores = pred_instances.keypoint_scores
        hand_type = pred_instances.hand_type

        res_2d = PoseDataSample()
        gt_instances = res.gt_instances.clone()
        pred_instances = pred_instances.clone()
        res_2d.gt_instances = gt_instances
        res_2d.pred_instances = pred_instances

        # add relative root depth to left hand joints
        keypoints[:, 21:, 2] += rel_root_depth

        # set joint scores according to hand type
        scores[:, :21] *= hand_type[:, [0]]
        scores[:, 21:] *= hand_type[:, [1]]
        # normalize kpt score
        if scores.max() > 1:
            scores /= 255

        res_2d.pred_instances.set_field(keypoints[..., :2].copy(), 'keypoints')

        # rotate the keypoint to make z-axis correspondent to height
        # for better visualization
        vis_R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        keypoints[..., :3] = keypoints[..., :3] @ vis_R

        # rebase height (z-axis)
        if not args.disable_rebase_keypoint:
            valid = scores > 0
            keypoints[..., 2] -= np.min(
                keypoints[valid, 2], axis=-1, keepdims=True)

        pose_results[idx].pred_instances.keypoints = keypoints
        pose_results[idx].pred_instances.keypoint_scores = scores
        pose_results_2d.append(res_2d)

    data_samples = merge_data_samples(pose_results)
    data_samples_2d = merge_data_samples(pose_results_2d)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            det_data_sample=data_samples_2d,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            convert_keypoint=False,
            axis_azimuth=-115,
            axis_limit=200,
            axis_elev=15,
            show_kpt_idx=args.show_kpt_idx,
            show=args.show,
            wait_time=show_interval)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    args = parse_args()

    assert args.input != ''
    assert args.show or (args.output_root != '')

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build the model from a config file and a checkpoint file
    model = init_model(
        args.config, args.checkpoint, device=args.device.lower())

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    if args.input == 'webcam':
        input_type = 'webcam'
    else:
        input_type = mimetypes.guess_type(args.input)[0].split('/')[0]

    if input_type == 'image':
        # inference
        pred_instances = process_one_image(args, args.input, model, visualizer)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type in ['webcam', 'video']:

        if args.input == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(args.input)

        video_writer = None
        pred_instances_list = []
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # topdown pose estimation
            pred_instances = process_one_image(args, frame, model, visualizer,
                                               0.001)

            if args.save_predictions:
                # save prediction results
                pred_instances_list.append(
                    dict(
                        frame_id=frame_idx,
                        instances=split_instances(pred_instances)))

            # output videos
            if output_file:
                frame_vis = visualizer.get_image()

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(
                        output_file,
                        fourcc,
                        25,  # saved fps
                        (frame_vis.shape[1], frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

            if args.show:
                # press ESC to exit
                if cv2.waitKey(5) & 0xFF == 27:
                    break

                time.sleep(args.show_interval)

        if video_writer:
            video_writer.release()

        cap.release()

    else:
        args.save_predictions = False
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')

    if args.save_predictions:
        with open(args.pred_save_path, 'w') as f:
            json.dump(
                dict(
                    meta_info=model.dataset_meta,
                    instance_info=pred_instances_list),
                f,
                indent='\t')
        print_log(
            f'predictions have been saved at {args.pred_save_path}',
            logger='current',
            level=logging.INFO)

    if output_file is not None:
        input_type = input_type.replace('webcam', 'video')
        print_log(
            f'the output {input_type} has been saved at {output_file}',
            logger='current',
            level=logging.INFO)


if __name__ == '__main__':
    main()
