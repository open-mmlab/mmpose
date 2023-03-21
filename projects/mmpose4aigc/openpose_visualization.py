# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
from argparse import ArgumentParser

import mmcv
import numpy as np
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import math

import cv2


def mmpose_to_openpose_visualization(args, img_path, detector, pose_estimator):
    """Visualize predicted keypoints of one image in openpose format."""

    # predict bbox
    init_default_scope(detector.cfg.get('default_scope', 'mmdet'))
    det_result = inference_detector(detector, img_path)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)
    # concatenate scores and keypoints
    keypoints = np.concatenate(
        (data_samples.pred_instances.keypoints,
         data_samples.pred_instances.keypoint_scores.reshape(-1, 17, 1)),
        axis=-1)
    # compute neck joint
    neck = (keypoints[:, 5] + keypoints[:, 6]) / 2
    neck[:, 2] = keypoints[:, 5, 2] * keypoints[:, 6, 2]
    # 17 keypoints to 18 keypoints
    new_keypoints = np.insert(keypoints[:, ], 17, neck, axis=1)
    # mmpose format to openpose format
    new_keypoints[:, [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1
                      ], :] = new_keypoints[:, [
                          1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
                      ], :]
    # out file name .png
    out_file = 'openpose_' + os.path.splitext(
        os.path.basename(img_path))[0] + '.png'
    # show the results
    img = mmcv.imread(img_path, channel_order='rgb')
    # black background
    black_img = np.zeros_like(img)

    # openpose format
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15],
               [15, 17], [1, 16], [16, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
              [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
              [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
              [255, 0, 170], [255, 0, 85]]

    # draw keypoints
    cur_black_img = black_img.copy()
    for i in range(len(new_keypoints)):
        for j in range(18):
            x, y, conf = new_keypoints[i][j]
            if conf == 0:
                continue
            cv2.circle(
                cur_black_img, (int(x), int(y)), 4, colors[j], thickness=-1)
    black_img = cv2.addWeighted(black_img, 0.4, cur_black_img, 0.6, 0)

    # draw links
    cur_black_img = black_img.copy()
    for i in range(len(new_keypoints)):
        for link in range(17):

            Y = new_keypoints[i][np.array(limbSeq[link]) - 1, 0]
            X = new_keypoints[i][np.array(limbSeq[link]) - 1, 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle),
                0, 360, 1)
            cv2.fillConvexPoly(cur_black_img, polygon, colors[link])
    black_img = cv2.addWeighted(black_img, 0.4, cur_black_img, 0.6, 0)
    # save image
    cv2.imwrite(out_file, black_img[:, :, [2, 1, 0]])


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--det_config', help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('--pose_config', help='Config file for pose')
    parser.add_argument('--pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--input', type=str, help='input Image file')
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
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
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

    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # init visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta)

    input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
    if input_type == 'image':
        mmpose_to_openpose_visualization(args, args.input, detector,
                                         pose_estimator)


if __name__ == '__main__':
    main()
