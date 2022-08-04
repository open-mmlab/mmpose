# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import time

from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(
        description='Converting the predicted .mat file to .json file.')
    parser.add_argument('pred_mat_file', help='input prediction mat file.')
    parser.add_argument(
        'gt_json_file',
        help='input ground-truth json file to get the image name. '
        'Default: "data/mpii/mpii_val.json" ')
    parser.add_argument('output_json_file', help='output converted json file.')
    args = parser.parse_args()
    return args


def save_json(list_file, path):
    with open(path, 'w') as f:
        json.dump(list_file, f, indent=4)
    return 0


def convert_mat(pred_mat_file, gt_json_file, output_json_file):
    res = loadmat(pred_mat_file)
    preds = res['preds']
    N = preds.shape[0]

    with open(gt_json_file) as anno_file:
        anno = json.load(anno_file)

    assert len(anno) == N

    instance = {}

    for pred, ann in zip(preds, anno):
        ann.pop('joints_vis')
        ann['joints'] = pred.tolist()

    instance['annotations'] = anno
    instance['info'] = {}
    instance['info']['description'] = 'Converted MPII prediction.'
    instance['info']['year'] = time.strftime('%Y', time.localtime())
    instance['info']['date_created'] = time.strftime('%Y/%m/%d',
                                                     time.localtime())

    save_json(instance, output_json_file)


def main():
    args = parse_args()
    convert_mat(args.pred_mat_file, args.gt_json_file, args.output_json_file)


if __name__ == '__main__':
    main()
