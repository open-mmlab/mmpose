import os
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
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
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    image_name = os.path.join(args.img_root, args.img)

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, image_name)

    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

    # test a single image, with a list of bboxes.

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_results,
        bbox_thr=args.bbox_thr,
        format='xyxy',
        dataset=dataset,
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
        dataset=dataset,
        kpt_score_thr=args.kpt_thr,
        show=args.show,
        out_file=out_file)


if __name__ == '__main__':
    main()
