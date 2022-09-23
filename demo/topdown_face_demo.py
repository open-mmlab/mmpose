# Copyright (c) OpenMMLab. All rights reserved.
import mimetypes
import os
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules as register_mmpose_modules

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
        # left, top, right, bottom
        person_results.append([bbox[3], bbox[0], bbox[1], bbox[2]])
    person_results = np.array(person_results)

    return person_results


def visualize_img(args, img_path, pose_estimator, visualizer, show_interval):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    image = face_recognition.load_image_file(img_path)
    face_det_results = face_recognition.face_locations(image)
    bboxes = process_face_det_results(face_det_results)

    bboxes = np.concatenate((bboxes, np.ones((bboxes.shape[0], 1))), axis=1)
    bboxes = bboxes[nms(bboxes, args.nms_thr)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')

    out_file = None
    if args.output_root:
        out_file = f'{args.output_root}/{os.path.basename(img_path)}'

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=args.draw_heatmap,
        draw_bbox=False,
        show=args.show,
        wait_time=show_interval,
        out_file=out_file,
        kpt_score_thr=args.kpt_thr)


def main():
    """Visualize the demo images.

    Use `face_recognition` to detect the face.
    """
    parser = ArgumentParser()
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
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
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
        help='Whether to draw output heatmap')
    parser.add_argument(
        '--radius',
        type=int,
        default=2,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_face_det, 'Please install face_recognition to run the demo. ' \
                         '"pip install face_recognition", For more details, ' \
                         'see https://github.com/ageitgey/face_recognition'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)

    # build pose estimator
    register_mmpose_modules()
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
    visualizer.kpt_color = 'red'

    input_type = mimetypes.guess_type(args.input)[0].split('/')[0]
    if input_type == 'image':
        visualize_img(
            args, args.input, pose_estimator, visualizer, show_interval=0)
    elif input_type == 'video':
        tmp_folder = tempfile.TemporaryDirectory()
        video = mmcv.VideoReader(args.input)
        progressbar = mmengine.ProgressBar(len(video))
        video.cvt2frames(tmp_folder.name, show_progress=False)
        output_root = args.output_root
        args.output_root = tmp_folder.name
        for img_fname in os.listdir(tmp_folder.name):
            visualize_img(
                args,
                f'{tmp_folder.name}/{img_fname}',
                pose_estimator,
                visualizer,
                show_interval=1)
            progressbar.update()
        if output_root:
            mmcv.frames2video(
                tmp_folder.name,
                f'{output_root}/{os.path.basename(args.input)}',
                fps=video.fps,
                fourcc='mp4v',
                show_progress=False)
        tmp_folder.cleanup()
    else:
        raise ValueError(
            f'file {os.path.basename(args.input)} has invalid format.')


if __name__ == '__main__':
    main()
