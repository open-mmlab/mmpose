import os
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


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
        help='Bounding bbox score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    dataset = pose_model.cfg.data['test']['type']

    image_name = os.path.join(args.img_root, args.img)

    # test a single image, the resulting box is (x1, y1, x2, y2)
    det_results = inference_detector(det_model, image_name)

    # keep the person class bounding boxes. (FasterRCNN)
    person_bboxes = det_results[0].copy()

    # test a single image, with a list of bboxes.
    pose_results, outputs = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_bboxes,
        bbox_thr=args.bbox_thr,
        format='xyxy',
        dataset=dataset)

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
