import argparse
import copy
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from datasets.pipelines import TopDownGenerateTargetFewShot
from mmcv.cnn import fuse_conv_bn
from mmengine.config import Config, DictAction
from mmengine.runner import load_checkpoint
from torchvision import transforms

from mmpose.models import build_pose_estimator
from tools.visualization import COLORS, plot_results


class ResizePad:

    def __init__(self, w=256, h=256):
        self.w = w
        self.h = h

    def __call__(self, image):
        _, w_1, h_1 = image.shape
        ratio_1 = w_1 / h_1
        # check if the original and final aspect ratios are the same within a
        # margin
        if round(ratio_1, 2) != 1:
            # padding to preserve aspect ratio
            if ratio_1 > 1:  # Make the image higher
                hp = int(w_1 - h_1)
                hp = hp // 2
                image = F.pad(image, (hp, 0, hp, 0), 0, 'constant')
                return F.resize(image, [self.h, self.w])
            else:
                wp = int(h_1 - w_1)
                wp = wp // 2
                image = F.pad(image, (0, wp, 0, wp), 0, 'constant')
                return F.resize(image, [self.h, self.w])
        else:
            return F.resize(image, [self.h, self.w])


def parse_args():
    parser = argparse.ArgumentParser(description='Pose Anything Demo')
    parser.add_argument('--support', help='Image file')
    parser.add_argument('--query', help='Image file')
    parser.add_argument(
        '--config', default='configs/demo.py', help='test config file path')
    parser.add_argument(
        '--checkpoint', default='pretrained', help='checkpoint file')
    parser.add_argument('--outdir', default='output', help='checkpoint file')

    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 "
        "model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    support_img = cv2.imread(args.support)
    query_img = cv2.imread(args.query)
    if support_img is None or query_img is None:
        raise ValueError('Fail to read images')

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        ResizePad(cfg.model.encoder_config.img_size,
                  cfg.model.encoder_config.img_size)
    ])

    # frame = copy.deepcopy(support_img)
    padded_support_img = preprocess(support_img).cpu().numpy().transpose(
        1, 2, 0) * 255
    frame = copy.deepcopy(padded_support_img.astype(np.uint8).copy())
    kp_src = []
    skeleton = []
    count = 0
    prev_pt = None
    prev_pt_idx = None
    color_idx = 0

    def selectKP(event, x, y, flags, param):
        nonlocal kp_src, frame
        # if we are in points selection mode, the mouse was clicked,
        # list of  points with the (x, y) location of the click
        # and draw the circle

        if event == cv2.EVENT_LBUTTONDOWN:
            kp_src.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 1)
            cv2.imshow('Source', frame)

        if event == cv2.EVENT_RBUTTONDOWN:
            kp_src = []
            frame = copy.deepcopy(support_img)
            cv2.imshow('Source', frame)

    def draw_line(event, x, y, flags, param):
        nonlocal skeleton, kp_src, frame, count, prev_pt, prev_pt_idx, \
            marked_frame, color_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            closest_point = min(
                kp_src, key=lambda p: (p[0] - x)**2 + (p[1] - y)**2)
            closest_point_index = kp_src.index(closest_point)
            if color_idx < len(COLORS):
                c = COLORS[color_idx]
            else:
                c = random.choices(range(256), k=3)

            cv2.circle(frame, closest_point, 2, c, 1)
            if count == 0:
                prev_pt = closest_point
                prev_pt_idx = closest_point_index
                count = count + 1
                cv2.imshow('Source', frame)
            else:
                cv2.line(frame, prev_pt, closest_point, c, 2)
                cv2.imshow('Source', frame)
                count = 0
                skeleton.append((prev_pt_idx, closest_point_index))
                color_idx = color_idx + 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            frame = copy.deepcopy(marked_frame)
            cv2.imshow('Source', frame)
            count = 0
            color_idx = 0
            skeleton = []
            prev_pt = None

    cv2.namedWindow('Source', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Source', 800, 600)
    cv2.setMouseCallback('Source', selectKP)
    cv2.imshow('Source', frame)

    # keep looping until points have been selected
    print('Press any key when finished marking the points!! ')
    while True:
        if cv2.waitKey(1) > 0:
            break

    marked_frame = copy.deepcopy(frame)
    cv2.setMouseCallback('Source', draw_line)
    print('Press any key when finished creating skeleton!!')
    while True:
        if cv2.waitKey(1) > 0:
            break

    cv2.destroyAllWindows()
    kp_src = torch.tensor(kp_src).float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ResizePad(cfg.model.encoder_config.img_size,
                  cfg.model.encoder_config.img_size)
    ])

    if len(skeleton) == 0:
        skeleton = [(0, 0)]

    support_img = preprocess(support_img).flip(0)[None]
    query_img = preprocess(query_img).flip(0)[None]
    # Create heatmap from keypoints
    genHeatMap = TopDownGenerateTargetFewShot()
    data_cfg = cfg.data_cfg
    data_cfg['image_size'] = np.array(
        [cfg.model.encoder_config.img_size, cfg.model.encoder_config.img_size])
    data_cfg['joint_weights'] = None
    data_cfg['use_different_joint_weights'] = False
    kp_src_3d = torch.cat((kp_src, torch.zeros(kp_src.shape[0], 1)), dim=-1)
    kp_src_3d_weight = torch.cat(
        (torch.ones_like(kp_src), torch.zeros(kp_src.shape[0], 1)), dim=-1)
    target_s, target_weight_s = genHeatMap._msra_generate_target(
        data_cfg, kp_src_3d, kp_src_3d_weight, sigma=2)
    target_s = torch.tensor(target_s).float()[None]
    target_weight_s = torch.tensor(target_weight_s).float()[None]

    data = {
        'img_s': [support_img],
        'img_q':
        query_img,
        'target_s': [target_s],
        'target_weight_s': [target_weight_s],
        'target_q':
        None,
        'target_weight_q':
        None,
        'return_loss':
        False,
        'img_metas': [{
            'sample_skeleton': [skeleton],
            'query_skeleton':
            skeleton,
            'sample_joints_3d': [kp_src_3d],
            'query_joints_3d':
            kp_src_3d,
            'sample_center': [kp_src.mean(dim=0)],
            'query_center':
            kp_src.mean(dim=0),
            'sample_scale': [kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0]],
            'query_scale':
            kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0],
            'sample_rotation': [0],
            'query_rotation':
            0,
            'sample_bbox_score': [1],
            'query_bbox_score':
            1,
            'query_image_file':
            '',
            'sample_image_file': [''],
        }]
    }

    # Load model
    model = build_pose_estimator(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model.eval()

    with torch.no_grad():
        outputs = model(**data)

    # visualize results
    vis_s_weight = target_weight_s[0]
    vis_q_weight = target_weight_s[0]
    vis_s_image = support_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    vis_q_image = query_img[0].detach().cpu().numpy().transpose(1, 2, 0)
    support_kp = kp_src_3d

    plot_results(
        vis_s_image,
        vis_q_image,
        support_kp,
        vis_s_weight,
        None,
        vis_q_weight,
        skeleton,
        None,
        torch.tensor(outputs['points']).squeeze(0),
        out_dir=args.outdir)

    print('Output saved to output dir: {}'.format(args.outdir))


if __name__ == '__main__':
    main()
