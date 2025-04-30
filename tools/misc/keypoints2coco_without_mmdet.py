# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
from argparse import ArgumentParser

from mmcv import track_iter_progress
from PIL import Image
from xtcocotools.coco import COCO

from mmpose.apis import inference_top_down_pose_model, init_pose_model


def main():
    """Visualize the demo images.

    pose_keypoints require the json_file containing boxes.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--json-file',
        type=str,
        default='',
        help='Json file containing image person bboxes in COCO format.')
    parser.add_argument(
        '--out-json-file',
        type=str,
        default='',
        help='Output json contains pseudolabeled annotation')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']

    img_keys = list(coco.imgs.keys())

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    categories = [{'id': 1, 'name': 'person'}]
    img_anno_dict = {'images': [], 'annotations': [], 'categories': categories}

    # process each image
    ann_uniq_id = int(0)
    for i in track_iter_progress(range(len(img_keys))):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])

        width, height = Image.open(image_name).size
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # add output of model and bboxes to dict
        for indx, i in enumerate(pose_results):
            pose_results[indx]['keypoints'][
                pose_results[indx]['keypoints'][:, 2] < args.kpt_thr, :3] = 0
            pose_results[indx]['keypoints'][
                pose_results[indx]['keypoints'][:, 2] >= args.kpt_thr, 2] = 2
            x = int(pose_results[indx]['bbox'][0])
            y = int(pose_results[indx]['bbox'][1])
            w = int(pose_results[indx]['bbox'][2] -
                    pose_results[indx]['bbox'][0])
            h = int(pose_results[indx]['bbox'][3] -
                    pose_results[indx]['bbox'][1])
            bbox = [x, y, w, h]
            area = round((w * h), 0)

            images = {
                'file_name': image_name.split('/')[-1],
                'height': height,
                'width': width,
                'id': int(image_id)
            }

            annotations = {
                'keypoints': [
                    int(i) for i in pose_results[indx]['keypoints'].reshape(
                        -1).tolist()
                ],
                'num_keypoints':
                len(pose_results[indx]['keypoints']),
                'area':
                area,
                'iscrowd':
                0,
                'image_id':
                int(image_id),
                'bbox':
                bbox,
                'category_id':
                1,
                'id':
                ann_uniq_id,
            }

            img_anno_dict['annotations'].append(annotations)
            ann_uniq_id += 1

        img_anno_dict['images'].append(images)

    # create json
    with open(args.out_json_file, 'w') as outfile:
        json.dump(img_anno_dict, outfile, indent=2)


if __name__ == '__main__':
    main()
