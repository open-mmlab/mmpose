# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.utils import track_iter_progress

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import split_instances


def process_one_image(args, img, pose_estimator, visualizer, show_interval):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # inference a single image
    batch_results = inference_bottomup(pose_estimator, img)
    results = batch_results[0]

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=False,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        show=args.show,
        wait_time=show_interval,
        kpt_thr=args.kpt_thr)

    return results.pred_instances


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
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
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
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


def main():
    args = parse_args()
    assert args.show or (args.output_root != '')
    assert args.input != ''
    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
    if input_type == 'image':
        pred_instances = process_one_image(
            args, args.input, model, visualizer, show_interval=0)
        pred_instances_list = split_instances(pred_instances)

        if output_file:
            img_vis = visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

    elif input_type == 'video':
        video_reader = mmcv.VideoReader(args.input)
        video_writer = None

        pred_instances_list = []

        for frame_id, frame in enumerate(track_iter_progress(video_reader)):
            pred_instances = process_one_image(
                args, frame, model, visualizer, show_interval=1)

            pred_instances_list.append(
                dict(
                    frame_id=frame_id,
                    instances=split_instances(pred_instances)))

            if output_file:
                frame_vis = visualizer.get_image()
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # the size of the image with visualization may vary
                    # depending on the presence of heatmaps
                    video_writer = cv2.VideoWriter(output_file, fourcc,
                                                   video_reader.fps,
                                                   (frame_vis.shape[1],
                                                    frame_vis.shape[0]))

                video_writer.write(mmcv.rgb2bgr(frame_vis))

        if video_writer:
            video_writer.release()

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
        print(f'predictions have been saved at {args.pred_save_path}')


if __name__ == '__main__':
    main()
