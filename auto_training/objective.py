from auto_training.config_factories.mmpose_config_factory import make_mmpose_config
from mmengine.runner import Runner
import argparse
import os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--res', type=int, help='resolution of the model')
    parser.add_argument('--augmentation_index', type=int, help='augmentation index')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--repeat_times', type=int, help='repeat times')
    parser.add_argument('--resnet_depth', type=int, help='resnet_depth')
    parser.add_argument('--backbone_type', type=str, help='backbone_type')

    return parser.parse_args()


def merge_args(cfg):
    """Merge CLI arguments to config."""

    cfg.launcher = 'none'

    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor',
                             cfg.get('preprocess_cfg', {}))

    return cfg


def train(res, augmentation_index, batch_size, repeat_times, resnet_depth, backbone_type):
    timestamp = f"det_res{res}_aug{augmentation_index}_b{batch_size}_rep{repeat_times}_d{resnet_depth}_{backbone_type}"
    # dataset = 'general_dataset_12_12_24'
    dataset = 'wurth_optimization_dataset'
    data_path = f'/data/{dataset}/'
    out_path = f'/data/wurth_optimization/{dataset}'

    # replace the ${key} with the value of cfg.key
    cfg = make_mmpose_config(
        data_path,
        classes=['bottom_left', 'bottom_right', 'top_left', 'top_right'],
        res=(res, res),
        augmentation_index=augmentation_index,
        batch_size=batch_size,
        repeat_times=repeat_times,
        resnet_depth=resnet_depth,
        backbone_type=backbone_type
    )

    # init the logger before other steps
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir = os.path.join(out_path, timestamp)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, f"{timestamp}_config.py"))

    # merge CLI arguments to config
    cfg = merge_args(cfg)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()

    return runner.val_metrics['PCK']


if __name__ == '__main__':
    args = parse_args()
    train(
        res=args.res,
        augmentation_index=args.augmentation_index,
        batch_size=args.batch_size,
        repeat_times=args.repeat_times,
        resnet_depth=args.resnet_depth,
        backbone_type=args.backbone_type,
    )