import os
from argparse import ArgumentParser

from tqdm import tqdm

from mmpose.apis import inference_pose_model, init_pose_model, show_pose_result


def main():
    """Visualize the demo images."""
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--img_prefix', type=str, default='', help='Image prefix')
    parser.add_argument(
        '--json_file',
        type=str,
        default='',
        help='Json file containing image info.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt_thr', type=float, default=0.3, help='box score threshold')
    args = parser.parse_args()

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    from pycocotools.coco import COCO
    coco = COCO(args.json_file)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    img_keys = list(coco.imgs.keys())

    # process each image
    for i in tqdm(range(len(img_keys))):
        # get bounding box annotations
        image_id = img_keys[i]
        image = coco.loadImgs(image_id)[0]
        image_name = os.path.join(args.img_prefix, image['file_name'])
        ann_ids = coco.getAnnIds(image_id)

        # make person bounding boxes
        person_bboxes = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            # bbox format is 'xywh'
            bbox = ann['bbox']
            person_bboxes.append(bbox)

        # test a single image, with a list of bboxes.
        pose_results = inference_pose_model(
            pose_model, image_name, person_bboxes, format='xywh')

        # show the results
        show_pose_result(
            pose_model,
            image_name,
            pose_results,
            skeleton=skeleton,
            kpt_score_thr=args.kpt_thr)


if __name__ == '__main__':
    main()
