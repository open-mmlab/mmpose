# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from argparse import ArgumentParser

import requests

from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.models import AssociativeEmbedding, TopDown


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--out-dir', default='vis_results', help='Visualization output path')
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # Inference single image by native apis.
    model = init_pose_model(args.config, args.checkpoint, device=args.device)
    if isinstance(model, TopDown):
        pytorch_result, _ = inference_top_down_pose_model(
            model, args.img, person_results=None)
    elif isinstance(model, (AssociativeEmbedding, )):
        pytorch_result, _ = inference_bottom_up_pose_model(model, args.img)
    else:
        raise NotImplementedError()

    vis_pose_result(
        model,
        args.img,
        pytorch_result,
        out_file=osp.join(args.out_dir, 'pytorch_result.png'))

    # Inference single image by torchserve engine.
    url = 'http://' + args.inference_addr + '/predictions/' + args.model_name
    with open(args.img, 'rb') as image:
        response = requests.post(url, image)
    server_result = response.json()

    vis_pose_result(
        model,
        args.img,
        server_result,
        out_file=osp.join(args.out_dir, 'torchserve_result.png'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
