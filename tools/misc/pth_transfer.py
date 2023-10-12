# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch


def change_model(args):
    dis_model = torch.load(args.dis_path, map_location='cpu')
    all_name = []
    if args.two_dis:
        for name, v in dis_model['state_dict'].items():
            if name.startswith('teacher.backbone'):
                all_name.append((name[8:], v))
            elif name.startswith('distill_losses.loss_mgd.down'):
                all_name.append(('head.' + name[24:], v))
            elif name.startswith('student.head'):
                all_name.append((name[8:], v))
            else:
                continue
    else:
        for name, v in dis_model['state_dict'].items():
            if name.startswith('student.'):
                all_name.append((name[8:], v))
            else:
                continue
    state_dict = OrderedDict(all_name)
    dis_model['state_dict'] = state_dict

    save_keys = ['meta', 'state_dict']
    ckpt_keys = list(dis_model.keys())
    for k in ckpt_keys:
        if k not in save_keys:
            dis_model.pop(k, None)

    torch.save(dis_model, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer CKPT')
    parser.add_argument('dis_path', help='dis_model path')
    parser.add_argument('output_path', help='output path')
    parser.add_argument(
        '--two_dis', action='store_true', default=False, help='if two dis')
    args = parser.parse_args()
    change_model(args)
