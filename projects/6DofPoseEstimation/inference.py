import os
import numpy as np
from utils import *
import argparse
import torch
import matplotlib.pyplot as plt

import mmengine
import mmcv

from mmengine.registry import init_default_scope
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.structures import merge_data_samples
from mmpose.evaluation.functional import nms

from mmdet.apis import inference_detector, init_detector
from mmengine.visualization import Visualizer


def predict(image_path):
    
    Visualizer.get_instance('visualization_hook')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    detector = init_detector(
        # '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmdet-tiny_milk/rtmdet-tiny_milk.py',
        '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmdet-tiny_ape/rtmdet-tiny_ape.py',
        # '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmdet-tiny_milk/best_coco_bbox_mAP_epoch_197.pth',
        '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmdet-tiny_ape/best_coco_bbox_mAP_epoch_198.pth',
        device=device)
    
    pose_estimator = init_pose_estimator(
        # '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmpose-s_milk/rtmpose-s_milk.py',
        '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmpose-s_ape/rtmpose-s_ape.py',
        # '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmpose-s_milk/best_PCK_epoch_20.pth',
        '/home/liuyoufu/code/mmpose-openmmlab/mmpose/work_dirs/rtmpose-s_ape/best_PCK_epoch_260.pth',
        device=device,
        cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}})
    
    init_default_scope(detector.cfg.get('default_scope', 'mmdet'))
            
    detect_result = inference_detector(detector, image_path)
    CONF_THRES = 0.5

    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4].astype('int')
    
    pose_results = inference_topdown(pose_estimator, image_path, bboxes)
    data_samples = merge_data_samples(pose_results)
    keypoints = data_samples.pred_instances.keypoints.astype('int')
    
    return keypoints


def parse_args():
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--image-path', help='image path')
    parser.add_argument('--root-path', type=str, help='root path')
    parser.add_argument('--id', type=str, help='object id, for example: 01')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    image_path = args.image_path
    obj_id = args.id
    
    # 根据图像文件路径，划分物体id 根目录 与 图像名称
    root_path = args.root_path
    
    # get model_info_dict, obj_id
    model_info_path = root_path + 'models/'
    model_info_dict = mmengine.load(model_info_path + 'models_info.yml')
    object_path = os.path.join(root_path, f'data/{args.id}/')
    info_dict = mmengine.load(object_path + 'info.yml')
    
    # 根据图像名，获取对应图像的内参
    intrinsic = np.array(info_dict[0]['cam_K']).reshape(3,3)
    
    # get corner3D (8*3)
    corners3D = get_3D_corners(model_info_dict, obj_id)
    
    # get prediction
    keypoint_pr = predict(image_path)
    corners2D_pr = keypoint_pr.reshape(-1,2)
    
    # Compute [R|t] by pnp  =====  pred
    R_pr, t_pr = pnp(corners3D,
                     corners2D_pr,
                     np.array(intrinsic, dtype='float32'))
    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
    proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, intrinsic)) 
    
    
    image = mmcv.imread(image_path)
    height = image.shape[0]
    width = image.shape[1]
    
    plt.xlim((0, width))
    plt.ylim((0, height))
    
    save = mmcv.imresize(image, (width, height))[:,:,::-1]
    plt.imshow(save)
    
    filename = os.path.basename(image_path)
    mmcv.imwrite(save, './photo/'+filename)
    
    # Projections
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], 
                     [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    for edge in edges_corners:
        plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='g', linewidth=2.0)
    plt.gca().invert_yaxis()
    plt.show()
    plt.pause(0)

if __name__ == "__main__":
    main()