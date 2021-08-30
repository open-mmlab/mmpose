# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

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
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument('--img', type=str, default='', help='Image file')
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

    assert has_face_det, 'Please install face_recognition to run the demo. ' \
                         '"pip install face_recognition", For more details, ' \
                         'see https://github.com/ageitgey/face_recognition'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''

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

    image_name = os.path.join(args.img_root, args.img)

    # test a single image, the resulting box is (top, right, bottom and left)
    image = face_recognition.load_image_file(image_name)
    face_det_results = face_recognition.face_locations(image)

    # keep the person class bounding boxes.
    face_results = process_face_det_results(face_det_results)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        image_name,
        face_results,
        bbox_thr=None,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    if args.out_img_root == '':
        out_file = None
    else:
        os.makedirs(args.out_img_root, exist_ok=True)
        out_file = os.path.join(args.out_img_root, f'vis_{args.img}')

    # show the results
    vis_pose_result(
        pose_model,
        image_name,
        pose_results,
        radius=args.radius,
        thickness=args.thickness,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr,
        show=args.show,
        out_file=out_file)


if __name__ == '__main__':
    main()
