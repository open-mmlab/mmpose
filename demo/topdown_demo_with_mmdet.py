# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import tempfile
import warnings
from argparse import ArgumentParser

import mmcv
import numpy as np
from mmengine.structures import InstanceData, PixelData

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.utils import register_all_modules as register_mmpose_modules

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules as register_mmdet_modules
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def visualize_img(args, img_path, detector, pose_estimator, visualizer,
                  show_interval):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    register_mmdet_modules()
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu()
    bboxes = pred_instance.bboxes[np.logical_and(
        pred_instance.labels == args.det_cat_id,
        pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes.numpy()

    # predict keypoints
    register_mmpose_modules()
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = pose_results[0]

    # merge predicted bboxes and keypoints
    instances_data = dict(
        bboxes=np.concatenate(
            [data_sample.gt_instances.bboxes for data_sample in pose_results],
            axis=0),
        bbox_scores=np.concatenate([
            data_sample.gt_instances.bbox_scores
            for data_sample in pose_results
        ],
                                   axis=0),
        keypoints=np.concatenate([
            data_sample.pred_instances.keypoints
            for data_sample in pose_results
        ],
                                 axis=0),
        keypoint_scores=np.concatenate([
            data_sample.pred_instances.keypoint_scores
            for data_sample in pose_results
        ],
                                       axis=0))
    pred_instances = InstanceData()
    pred_instances.set_data(instances_data)
    data_samples.pred_instances = pred_instances

    if args.visualize_heatmap:
        if 'pred_fields' not in data_samples:
            warnings.warn('Heatmaps are not returned. Please check '
                          'whether topdown-heatmap model is utilized.')
        else:
            # merge predicted heatmaps
            heatmaps = []
            for data_sample in pose_results:
                heatmap, _ = data_sample.pred_fields.heatmaps.max(axis=0)
                heatmap = visualizer.revert_heatmap(
                    heatmap, data_sample.gt_instances.bbox_centers,
                    data_sample.gt_instances.bbox_scales,
                    data_samples.ori_shape)
                heatmaps.append(heatmap)
            heatmaps = np.stack(heatmaps, axis=0)

            pred_fields = PixelData()
            pred_fields.set_data(dict(heatmaps=heatmaps))
            data_samples.pred_fields = pred_fields

    # show the results
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    out_file = None
    if args.out_img_root:
        out_file = f'{args.out_img_root}/{os.path.basename(img_path)}'

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=args.visualize_heatmap,
        draw_bbox=False,
        show=args.show,
        wait_time=show_interval,
        out_file=out_file,
        kpt_score_thr=args.kpt_thr)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--visualize-heatmap',
        action='store_true',
        default=False,
        help='whether to visualize output heatmap')
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

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build detector
    register_mmdet_modules()
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)

    # build pose estimator
    register_mmpose_modules()
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.visualize_heatmap))))

    # init visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
    if input_type == 'image':
        visualize_img(
            args,
            args.input,
            detector,
            pose_estimator,
            visualizer,
            show_interval=0)
    elif input_type == 'video':
        tmp_folder = tempfile.TemporaryDirectory()
        video = mmcv.VideoReader(args.input)
        video.cvt2frames(tmp_folder.name, show_progress=False)
        for img_fname in os.listdir(tmp_folder.name):
            visualize_img(
                args,
                f'{tmp_folder.name}/{img_fname}',
                detector,
                pose_estimator,
                visualizer,
                show_interval=1)
        tmp_folder.cleanup()
    else:
        raise ValueError(f'file {os.basename(args.input)} has invalid format.')


if __name__ == '__main__':
    main()
