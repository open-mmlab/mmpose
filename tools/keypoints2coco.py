import os
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
import imagesize
import numpy as np
import json
from tqdm import tqdm
def main():
    """Visualize the demo images.

    pose_keypoints equire the json_file containing boxes.
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
        '--out-json-file',
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
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')

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
    img_anno_dict = {'images': [], 'annotations':[], 'categories':categories}

    # process each image
    ann_uniq_id = int(0)
    for idx, i in enumerate(tqdm(range(len(img_keys)))):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_root, image['file_name'])
        width, height = imagesize.get(image_name)
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_results = []
        for ann_id in ann_ids:
            person = {}
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            person['bbox'] = ann['bbox']
            person_results.append(person)

        #print(ann_id, person_results)
        # test a single image, with a list of bboxes
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=None,
            format='xywh',
            dataset=dataset,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
    
        for indx, i in enumerate(pose_results):
            pose_results[indx]['keypoints'][::, 2] = int(2)
            top = int(pose_results[indx]['bbox'][0])
            left = int(pose_results[indx]['bbox'][1])
            bbox_width = int(pose_results[indx]['bbox'][2] - pose_results[indx]['bbox'][0])
            bbox_height = int(pose_results[indx]['bbox'][3] - pose_results[indx]['bbox'][1])
            bbox = [top, left, bbox_width, bbox_height]
            area = round((bbox_width*bbox_height), 0)
            

            images = {
            'file_name': image_name.split('/')[-1],
            'height': height,
            'width': width,
            'id': int(image_id)
            }   


            annotations = {
            'keypoints': [int(i) for i in pose_results[indx]['keypoints'].reshape(-1).tolist()],
            'num_keypoints': len(pose_results[indx]['keypoints']),
            'area': area,
            'iscrowd': 0,
            'image_id': int(image_id),
            'bbox': bbox,
            'category_id': 1,
            'id': ann_uniq_id,
            }

            
            img_anno_dict['annotations'].append(annotations)
            ann_uniq_id += 1

        img_anno_dict['images'].append(images)

        if args.out_img_root == '':
                out_file = None
        else:
            os.makedirs(args.out_img_root, exist_ok=True)
            out_file = os.path.join(args.out_img_root, image_name.split('/')[-1])
        
        
        #vis_pose_result(
        #    pose_model,
        #    image_name,
        #    pose_results,
        #    dataset=dataset,
        #    kpt_score_thr=args.kpt_thr,
        #    show=args.show,
        #    out_file=out_file)

    with open(args.out_json_file, "w") as outfile:
        json.dump(img_anno_dict, outfile, indent=2)
            

if __name__ == '__main__':
    main()

