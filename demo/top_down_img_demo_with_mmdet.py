# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


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
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
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

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    '''
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    '''
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
    '''
    python3 demo/top_down_img_demo_with_mmdet.py mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth configs/vehicle/2d_kpt_sview_rgb_img/topdown_heatmap/carfusion/hrnet_w32_carfusion_384x288.py ./trained_files/latest.pth --bbox-thr 0.12 --kpt-thr 0.4 --thickness 2 --radius 6 --det-cat-id 3 --img-root data/ --out-img-root run/ --img o_10015.jpg
    python3 demo/top_down_img_demo_with_mmdet.py mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth configs/vehicle/2d_kpt_sview_rgb_img/topdown_heatmap/carfusion/hrnet_w32_carfusion_384x288.py ./trained_files/latest.pth --bbox-thr 0.3 --kpt-thr 0.4 --thickness 2 --radius 6 --det-cat-id 3 --img-root ../CVPR2023/data/shaler3/Cwalt_database/ --out-img-root ../CVPR2023/data/shaler3/ --img 1.jpg
    '''

    import glob
    root = args.img_root
    for folder in glob.glob(root+'/*'):
        print(folder)
        for img_name in glob.glob(folder+'/*.jpg'):
            args.img = img_name
            if '.npy' in img_name:
                continue
            if os.path.exists(img_name + '.npz'):
                continue

            image_name = os.path.join(args.img)

            '''

    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, image_name)

    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
            '''

            person_results_single = []
            import numpy as np
            import imagesize
    

            width, height = imagesize.get(image_name)
            print(image_name, width, height)

            person_results_single.append({'bbox': np.array([0,0,width,height,0.99], dtype='float32')})
            person_results = person_results_single

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
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            np.save(img_name+'.npy',pose_results)

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
                dataset_info=dataset_info,
                kpt_score_thr=args.kpt_thr,
                radius=args.radius,
                thickness=args.thickness,
                show=args.show,
                out_file=out_file)


if __name__ == '__main__':
    main()
