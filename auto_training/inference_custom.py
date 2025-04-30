import argparse
import os
import json
import numpy as np
import cv2
from mmengine.config import Config, DictAction
from mmpose.apis import init_model, inference_topdown
from xtcocotools.coco import COCO
from mmengine.utils import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMPose inference on images")
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('model', help='Path to checkpoint or ONNX file')
    parser.add_argument('--img-dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--bbox-json', type=str, required=True, help='Path to COCO format bounding box JSON')
    parser.add_argument('--out-dir', type=str, help='Directory to save visualized results (optional)')
    parser.add_argument('--predictions-dir', type=str, required=True, help='Directory to save individual prediction files')
    parser.add_argument('--device', default='cuda:0', help='Device to run inference on (e.g., "cuda:0" or "cpu")')
    parser.add_argument('--score-thr', type=float, default=0.1, help='Keypoint score threshold')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='Override some settings in the config file. The key-value pair in '
             'xxx=yyy format will be merged into the config.')
    return parser.parse_args()


def draw_bboxes(image, bboxes):
    """Draw bounding boxes on the image using OpenCV."""
    for bbox in bboxes:
        x, y, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + x2, y + y2), (255, 0, 0), 2)  # Draw rectangle


def draw_keypoints(image, keypoints, scores, score_thr):
    """Draw keypoints on the image using OpenCV."""

            
    labels = ["front_right", "rear_right", "front_left", "rear_left"]
    labels.sort()
    for el1, el2 in zip(keypoints, scores):

        for kp, score, label in zip(el1, el2, labels):
            if score > score_thr:
                x, y = int(kp[0]), int(kp[1])
                color = (0, 255, 0) if "front" in label else (0, 0, 255)
                cv2.circle(image, (x, y), 5, color, -1)  # Draw keypoint
                cv2.putText(image, f"{int(score*100)}% {label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    args = parse_args()

    coco = COCO(args.bbox_json)
    img_ids = list(coco.imgs.keys())

    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    model = init_model(cfg, args.model, device=args.device)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.predictions_dir, exist_ok=True)

    progress_bar = ProgressBar(len(img_ids))

    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(args.img_dir, img_info['file_name'])

        if not os.path.exists(img_path):
            progress_bar.update()
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image: {img_path}")
            progress_bar.update()
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations = coco.loadAnns(ann_ids)
        person_bboxes = np.array([ann['bbox'] for ann in annotations])
        person_bboxes = np.unique(person_bboxes, axis=0)

        #for a in annotations:
        #   kps = np.asarray(a["keypoints"]).reshape((-1, 3))
        #   scores = np.ones_like(kps) 
        #   draw_keypoints(image, [kps], scores, [0])



        pose_results = inference_topdown(
            model,
            image_rgb,
            person_bboxes,
            bbox_format='xywh'  # COCO annotations typically use 'xywh' format
        )

        keypoints_results = []
        for pose in pose_results:
            pred_instances = pose.pred_instances
            if pred_instances is not None:


                keypoints = pred_instances.keypoints
                scores = pred_instances.keypoint_scores
                bbox = pred_instances.bboxes[0]

                # Save the bbox along with the keypoints data
                keypoints_results.append({
                    'keypoints': keypoints.tolist(),
                    'scores': scores.tolist(),
                    'bbox': bbox.tolist()  # Add bbox to the result
                })

                if args.out_dir:
                    draw_keypoints(image, keypoints, scores, args.score_thr)
                    draw_bboxes(image, person_bboxes)

        # Save the visualized image if `out-dir` is provided
        if args.out_dir:
            out_file = os.path.join(args.out_dir, img_info['file_name'])
            cv2.imwrite(out_file, image)

        # Save individual prediction file
#        prediction_file = os.path.join(args.predictions_dir, f"{os.path.splitext(img_info['file_name'])[0]}.json")
#        with open(prediction_file, 'w') as f:
#            json.dump({
#                "result": keypoints_results,
#                "score": 0  # Placeholder for score; replace with actual logic if needed
#            }, f)
#
        progress_bar.update()

    print("Inference completed.")


if __name__ == '__main__':
    main()