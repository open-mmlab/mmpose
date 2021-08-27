# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_mesh_model, init_pose_model,
                         vis_3d_mesh_result)


def main():
    """Visualize the demo images.

    Require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    img_keys = list(coco.imgs.keys())

    # process each image
    for i in range(len(img_keys)):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        # test a single image, with a list of bboxes
        pose_results = inference_mesh_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset)

        if args.out_img_root == '':
            out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        vis_3d_mesh_result(
            pose_model,
            pose_results,
            image_name,
            show=args.show,
            out_file=out_file)


if __name__ == '__main__':
    main()
