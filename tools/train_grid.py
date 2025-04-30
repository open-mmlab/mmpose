import argparse
import os
import os.path as osp
import csv
import shutil
import itertools
import time

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

csv_file = 'training_results.csv'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def check_existing_run(work_dir):
    if not os.path.exists(csv_file):
        return False
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['run_name'] == work_dir and row['status'] == "completed":
                return True
    return False


def save_run_result(run_name, status, epe, ap, ar):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['run_name', 'status', 'EPE', 'AP', 'AR'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'run_name': run_name,
            'status': status,
            'EPE': epe if epe is not None else "N/A",
            'AP': ap if ap is not None else "N/A",
            'AR': ar if ar is not None else "N/A"
        })


def run_test(work_dir, config, checkpoint):
    command = f"python tools/test.py {config} {checkpoint} --work-dir {work_dir}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        lines = result.stdout.splitlines()
        epe, ap, ar = None, None, None
        for line in lines:
            if 'EPE' in line:
                epe = float(line.split(': ')[-1])
            elif 'AP' in line:
                ap = float(line.split(': ')[-1])
            elif 'AR' in line:
                ar = float(line.split(': ')[-1])
        return epe, ap, ar
    else:
        print(f"Test failed: {result.stderr}")
        return None, None, None


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > config > default
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if args.amp is True:
        from mmengine.optim import AmpOptimWrapper
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()

    hyperparam_grid = {
        'lr': [1e-5, 1e-4, 1e-3],
        'epochs': [5], # TODO: set valid
        'sigma': [1, 1.5, 2],
        'rot_factor': [40, 60],
        'optimizer': ['Adam', 'SGD'],
        'batch_size': [32, 64, 128],
    }

    step_dict = {
        5: [2, 3],
        10: [7, 9],
        100: [70, 90],
        200: [140, 180],
        300: [210, 270],
    }

    hyperparam_combinations = list(itertools.product(
        hyperparam_grid['lr'],
        hyperparam_grid['epochs'],
        hyperparam_grid['sigma'],
        hyperparam_grid['rot_factor'],
        hyperparam_grid['optimizer'],
        hyperparam_grid['batch_size']
    ))

    print(f"Total combinations: {len(hyperparam_combinations)}")

    for lr, epochs, sigma, rot_factor, optimizer, batch_size in hyperparam_combinations:
        cfg = Config.fromfile(args.config)
        cfg = merge_args(cfg, args)

        work_dir = f'work_dirs/grid_lr_{lr}_epochs_{epochs}_sigma_{sigma}_rot_{rot_factor}_optim_{optimizer}_batch_{batch_size}/'
        if check_existing_run(work_dir):
            print(f"Skipping already completed run: {work_dir}")
            continue

        # Update the config with the hyperparameters
        cfg.optim_wrapper.optimizer = dict(type=optimizer, lr=lr)
        cfg.train_cfg.max_epochs = epochs
        cfg.codec.sigma = sigma  # Modify sigma in loss
        cfg.train_dataloader.batch_size = batch_size
        cfg.param_scheduler[1].milestones = step_dict[epochs]

        # Assign the work directory
        cfg.work_dir = work_dir

        # Create work directory
        mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Dump the updated config to the work directory
        config_dst = osp.join(cfg.work_dir, 'modified_config.py')
        cfg.dump(config_dst)

        # Set up Runner and train
        runner = Runner.from_cfg(cfg)

        try:
            runner.train()

            # Run test and evaluate results
            checkpoint_path = osp.join(work_dir, "latest.pth")
            epe, ap, ar = run_test(work_dir, config_dst, checkpoint_path)
            status = "completed" if epe is not None and ap is not None and ar is not None else "failed"

        except Exception as e:
            print(f"Error during training: {e}")
            epe, ap, ar = None, None, None
            status = "failed"

        save_run_result(work_dir, status, epe, ap, ar)


if __name__ == '__main__':
    main()
