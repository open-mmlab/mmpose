import os
from argparse import ArgumentParser

import cv2
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-root', type=str, help='image root')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-img-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert (args.show or args.out_img_root != '')
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    #cap = cv2.VideoCapture(args.video_path)
    #assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    #fps = cap.get(cv2.CAP_PROP_FPS)
    #size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    #if args.out_video_root == '':
    #    save_out_video = False
    #else:
    os.makedirs(args.out_img_root, exist_ok=True)
    #    save_out_video = True

    #if save_out_video:
    #    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #    videoWriter = cv2.VideoWriter(
    #        os.path.join(args.out_video_root,
    #                     f'vis_{os.path.basename(args.video_path)}'), fourcc,
    #        fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # while (cap.isOpened()):
    #     flag, img = cap.read()
    #     if not flag:
    #         break
    assert os.path.exists(args.img_root)
    for img_name in os.listdir(args.img_root):
        img_path = os.path.join(args.img_root, img_name)
        img = cv2.imread(img_path)
        size = img.shape[1], img.shape[0]
        # keep the person class bounding boxes.
        person_results = [{'bbox': np.array([0, 0, size[0], size[1]])}]

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            format='xyxy',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)
        write_path = os.path.join(args.out_img_root, img_name)
        cv2.imwrite(write_path ,vis_img)



if __name__ == '__main__':
    main()
