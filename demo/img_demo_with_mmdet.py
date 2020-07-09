import os
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

from mmpose.apis import inference_pose_model, init_pose_model, show_pose_result


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
        '--img_prefix', type=str, default='', help='Image prefix')
    parser.add_argument('--img', type=str, default='', help='Image file')

    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--bbox_thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--kpt_thr', type=float, default=0.3, help='keypoint score threshold')
    args = parser.parse_args()

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    image_name = os.path.join(args.img_prefix, args.img)

    # test a single image, the resulting box is (x1, y1, x2, y2)
    det_results = inference_detector(det_model, image_name)

    # keep the person class bounding boxes.
    person_bboxes = det_results[0][0].copy()

    # test a single image, with a list of bboxes.
    pose_results = inference_pose_model(
        pose_model,
        image_name,
        person_bboxes,
        bbox_thr=args.bbox_thr,
        format='xyxy')

    # show the results
    show_pose_result(
        pose_model,
        image_name,
        pose_results,
        skeleton=skeleton,
        kpt_score_thr=args.kpt_thr)


if __name__ == '__main__':
    main()
