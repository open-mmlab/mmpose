# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    import face_recognition
    has_face_det = True
except (ImportError, ModuleNotFoundError):
    has_face_det = False


def process_face_det_results(face_det_results):
    """Process det results, and return a list of bboxes.

    :param face_det_results: (top, right, bottom and left)
    :return: a list of detected bounding boxes (x,y,x,y)-format
    """

    person_results = []
    for bbox in face_det_results:
        person = {}
        # left, top, right, bottom
        person['bbox'] = [bbox[3], bbox[0], bbox[1], bbox[2]]
        person_results.append(person)

    return person_results


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
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

    assert has_face_det, 'Please install face_recognition to run the demo. '\
                         '"pip install face_recognition", For more details, '\
                         'see https://github.com/ageitgey/face_recognition'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')

    print('Initializing model...')
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

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running inference...')
    for _, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        face_det_results = face_recognition.face_locations(
            cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB))
        face_results = process_face_det_results(face_det_results)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            face_results,
            bbox_thr=None,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Frame', vis_frame)

        if save_out_video:
            videoWriter.write(vis_frame)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
