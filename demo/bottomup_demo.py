# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcv.image import imread

from mmpose.apis import inference_bottomup, init_model
from mmpose.registry import VISUALIZERS
from mmpose.utils import register_all_modules


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.show or (args.out_file != '')

    # register all modules in mmpose into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    # inference a single image
    batch_results = inference_bottomup(model, args.input)
    results = batch_results[0]

    # show the results
    img = imread(args.input, channel_order='rgb')

    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=False,
        show=args.show,
        draw_heatmap=args.draw_heatmap,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
