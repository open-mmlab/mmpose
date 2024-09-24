# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import subprocess
from datetime import date

import torch
from mmengine.logging import print_log
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument(
        '--save-keys',
        nargs='+',
        type=str,
        default=['meta', 'state_dict'],
        help='keys to save in published checkpoint (default: meta state_dict)')
    parser.add_argument(
        '--float16',
        action='store_true',
        default=False,
        help='Whether save model as float16')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file,
                       out_file,
                       save_keys=['meta', 'state_dict'],
                       float16=False):
    checkpoint = torch.load(in_file, map_location='cpu')
    checkpoint['meta']['float16'] = float16

    # only keep `meta` and `state_dict` for smaller file size
    ckpt_keys = list(checkpoint.keys())
    for k in ckpt_keys:
        if k not in save_keys:
            print_log(
                f'Key `{k}` will be removed because it is not in '
                f'save_keys. If you want to keep it, '
                f'please set --save-keys.',
                logger='current')
            checkpoint.pop(k, None)

    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.

    if float16:
        print(save_keys)
        if 'meta' not in save_keys:
            raise ValueError(
                'Key `meta` must be in save_keys to save model as float16. '
                'Change float16 to False or add `meta` in save_keys.')
        print_log('Saving model as float16.', logger='current')
        for key in checkpoint['state_dict'].keys():
            checkpoint['state_dict'][key] = checkpoint['state_dict'][key].half(
            )

    if digit_version(TORCH_VERSION) >= digit_version('1.8.0'):
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)
    sha = subprocess.check_output(['sha256sum', out_file]).decode()
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file

    date_now = date.today().strftime('%Y%m%d')
    final_file = out_file_name + f'-{sha[:8]}_{date_now}.pth'
    subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file, args.save_keys,
                       args.float16)


if __name__ == '__main__':
    main()
