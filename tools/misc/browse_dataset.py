# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine import Config, DictAction
from mmengine.registry import build_from_cfg, init_default_scope
from mmengine.structures import InstanceData

from mmpose.registry import DATASETS, VISUALIZERS
from mmpose.structures import PoseDataSample


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--mode',
        default='transformed',
        type=str,
        choices=['original', 'transformed'],
        help='display mode; display original pictures or transformed '
        'pictures. "original" means to show images load from disk'
        '; "transformed" means to show images after transformed;'
        'Defaults to "transformed".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def generate_dup_file_name(out_file):
    """Automatically rename out_file when duplicated file exists.

    This case occurs when there is multiple instances on one image.
    """
    if out_file and osp.exists(out_file):
        img_name, postfix = osp.basename(out_file).rsplit('.', 1)
        exist_files = tuple(
            filter(lambda f: f.startswith(img_name),
                   os.listdir(osp.dirname(out_file))))
        if len(exist_files) > 0:
            img_path = f'{img_name}({len(exist_files)}).{postfix}'
            out_file = osp.join(osp.dirname(out_file), img_path)
    return out_file


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    backend_args = cfg.get('backend_args', dict(backend='local'))

    # register all modules in mmpose into the registries
    scope = cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    if args.mode == 'original':
        cfg[f'{args.phase}_dataloader'].dataset.pipeline = []
    else:
        # pack transformed keypoints for visualization
        cfg[f'{args.phase}_dataloader'].dataset.pipeline[
            -1].pack_transformed = True

    dataset = build_from_cfg(cfg[f'{args.phase}_dataloader'].dataset, DATASETS)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.set_dataset_meta(dataset.metainfo)

    progress_bar = mmengine.ProgressBar(len(dataset))

    idx = 0
    item = dataset[0]

    while idx < len(dataset):
        idx += 1
        next_item = None if idx >= len(dataset) else dataset[idx]

        if args.mode == 'original':
            if next_item is not None and item['img_path'] == next_item[
                    'img_path']:
                # merge annotations for one image
                item['keypoints'] = np.concatenate(
                    (item['keypoints'], next_item['keypoints']))
                item['keypoints_visible'] = np.concatenate(
                    (item['keypoints_visible'],
                     next_item['keypoints_visible']))
                item['bbox'] = np.concatenate(
                    (item['bbox'], next_item['bbox']))
                progress_bar.update()
                continue
            else:
                img_path = item['img_path']
                img_bytes = fileio.get(img_path, backend_args=backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='bgr')

                # forge pseudo data_sample
                gt_instances = InstanceData()
                gt_instances.keypoints = item['keypoints']
                gt_instances.keypoints_visible = item['keypoints_visible']
                gt_instances.bboxes = item['bbox']
                data_sample = PoseDataSample()
                data_sample.gt_instances = gt_instances

                item = next_item
        else:
            img = item['inputs'].permute(1, 2, 0).numpy()
            data_sample = item['data_samples']
            img_path = data_sample.img_path
            item = next_item

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None
        out_file = generate_dup_file_name(out_file)

        img = mmcv.bgr2rgb(img)

        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            draw_pred=False,
            draw_bbox=(args.mode == 'original'),
            draw_heatmap=True,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=out_file)

        progress_bar.update()


if __name__ == '__main__':
    main()
