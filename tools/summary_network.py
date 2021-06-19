import argparse
# import tensorwatch as tw

from mmcv import Config
# from mmcv.cnn import get_model_complexity_info
from torchstat_utils import model_stats
from mmpose.models import build_posenet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    parser.add_argument('--out-file', type=str,
                        help='Output file name') 
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_posenet(cfg.model)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    df = model_stats(model, input_shape)
    print(df)
    if args.out_file:
        df.to_html(args.out_file + '.html')
        df.to_csv(args.out_file + '.csv')

    # flops, params = get_model_complexity_info(model, input_shape)
    # split_line = '=' * 30
    # print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
    #     split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
