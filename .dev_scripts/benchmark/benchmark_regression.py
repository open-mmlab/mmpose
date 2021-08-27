# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp
import random
import socket
from datetime import datetime

import mmcv


def is_port_available(port, host='127.0.0.1'):
    """check whether a port is in use return True if the port is available else
    False."""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, int(port)))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='running benchmark regression with tmux')
    parser.add_argument(
        '--config',
        help='test config file path',
        default='./.dev_scripts/benchmark/benchmark_regression_cfg.yaml')
    parser.add_argument(
        '--priority',
        nargs=2,
        type=int,
        help='largest priority for test and train tasks respectively',
        default=[3, 3])

    # runtime setting parameters
    parser.add_argument(
        '--root-work-dir', help='the root working directory to store logs')
    parser.add_argument(
        '--session-name', '-s', help='the tmux session name', default='test')
    parser.add_argument(
        '--panes-per-window',
        type=int,
        help='the maximum number of panes in each tmux window',
        default=12)
    parser.add_argument(
        '--env',
        help='the conda environment used to run the tasks',
        default='pt1.6')
    parser.add_argument(
        '--partition', help='the partition name', default='mm_human')
    parser.add_argument('--gpus', help='the total number of GPUs', default=8)
    parser.add_argument(
        '--gpus-per-node',
        default=8,
        help='the number of GPUs used per computing node',
        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument(
        '--cpus-per-task', default=5, help='the number of CPUs used per task')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.root_work_dir is None:
        # get the current time stamp
        now = datetime.now()
        ts = now.strftime('%Y_%m_%d_%H_%M')
        args.root_work_dir = f'work_dirs/benchmark_regression_{ts}'
    mmcv.mkdir_or_exist(osp.abspath(args.root_work_dir))

    cfg = mmcv.load(args.config)

    # priority for test and train tasks respectively
    prio_test, prio_train = args.priority
    prio = max(prio_test, prio_train) + 1

    # number of benchmark regression tasks
    num_task = 0
    for i in range(prio):
        if i <= prio_test:
            num_task += len(cfg['model_list'][f'P{i}'])
        if i <= prio_train:
            num_task += len(cfg['model_list'][f'P{i}'])

    # number of windows need to be created
    num_win = math.ceil(num_task / args.panes_per_window)

    # create a new tmux session
    os.system(f'tmux new -s {args.session_name} -d')

    # tmux select-window -t 0
    os.system('tmux select-window -t 0')

    num_task_tmp = num_task
    # create new windows and panes
    for i in range(num_win):
        # tmux select-window -t 0
        os.system('tmux select-window -t 0')
        # tmux new-window -n win_1
        os.system(f'tmux new-window -n win_{i+1}')

        # decide how many panes will be created in current window
        if num_task_tmp >= args.panes_per_window:
            num_cur_win_task = args.panes_per_window
            num_task_tmp -= args.panes_per_window
        else:
            num_cur_win_task = num_task_tmp

        # split each window into different panes
        for j in range(num_cur_win_task - 1):
            ratio = int(100 - 100 / (num_cur_win_task - j))
            os.system(f'tmux split-window -h -p {ratio}')
        os.system('tmux select-layout tiled')

    # the initial number of task
    cur_task = 1

    # get the hostname
    hostname = socket.gethostname()
    print('hostname: ', hostname)
    # get the host ip
    ip = socket.gethostbyname(hostname)
    print('ip: ', ip)

    # initialize a starting port
    cur_port = 29500

    for i in range(prio):
        models = cfg['model_list'][f'P{i}']

        # modes = ['test','train']
        modes = []
        if i <= prio_test:
            modes.append('test')
        if i <= prio_train:
            modes.append('train')

        for model in models:
            cur_config = model['config']
            cur_checkpoint = model['checkpoint']

            if 'task_name' in model.keys():
                task_name = model['task_name']
            else:
                task_name = osp.splitext(osp.basename(cur_config))[0]

            for mode in modes:
                # select the window and pane
                cur_win = int(math.ceil(cur_task / args.panes_per_window))
                os.system('tmux select-window -t 0')
                os.system(f'tmux select-window -t win_{cur_win}')
                cur_pane = (cur_task - 1) % args.panes_per_window
                os.system(f'tmux select-pane -t {cur_pane}')

                cmd = f'conda activate {args.env}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')
                cmd = f'echo executing task: {cur_task}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')

                cur_partition = model[mode][
                    'partition'] if 'partition' in model[mode].keys(
                    ) else args.partition
                cur_gpus = model[mode]['gpus'] if 'gpus' in model[mode].keys(
                ) else args.gpus
                cur_gpus_per_node = model[mode][
                    'gpus_per_node'] if 'gpus_per_node' in model[mode].keys(
                    ) else args.gpus_per_node
                cur_cpus_per_task = model[mode][
                    'cpus_per_task'] if 'cpus_per_task' in model[mode].keys(
                    ) else args.cpus_per_task

                cur_task_name = mode + '_' + task_name
                cur_work_dir = osp.join(args.root_work_dir, cur_task_name)

                if mode == 'test':
                    # deal with extra python arguments
                    py_cmd = f' --work-dir {cur_work_dir} '
                    if 'py_args' in model[mode].keys():
                        keys = list(model[mode]['py_args'].keys())
                        values = list(model[mode]['py_args'].values())

                        for k in range(len(keys)):
                            if values[k] is None:
                                if keys[k] in ['fuse_conv_bn', 'gpu_collect']:
                                    py_cmd += f' --{keys[k]} '
                            else:
                                py_cmd += f' --{keys[k]} {values[k]} '
                    cmd = f'MASTER_PORT={cur_port} GPUS={cur_gpus} ' + \
                          f'GPUS_PER_NODE={cur_gpus_per_node} ' + \
                          f'CPUS_PER_TASK={cur_cpus_per_task} ' + \
                          f'./tools/slurm_test.sh {cur_partition} ' + \
                          f'{cur_task_name} ' + \
                          f'{cur_config} {cur_checkpoint} ' + \
                          f'{py_cmd}'

                    os.system(f'tmux send-keys "{cmd}" "C-m"')

                else:
                    py_cmd = ' '
                    # deal with extra python arguments
                    if 'py_args' in model[mode].keys():
                        keys = list(model[mode]['py_args'].keys())
                        values = list(model[mode]['py_args'].values())

                        for k in range(len(keys)):
                            if values[k] is None:
                                if keys[k] in [
                                        'no-validate', 'deterministic',
                                        'autoscale-lr'
                                ]:
                                    py_cmd += f' --{keys[k]} '
                            else:
                                py_cmd += f' --{keys[k]} {values[k]} '
                    cmd = f'MASTER_PORT={cur_port} GPUS={cur_gpus} ' + \
                          f'GPUS_PER_NODE={cur_gpus_per_node} ' + \
                          f'CPUS_PER_TASK={cur_cpus_per_task} ' + \
                          f'./tools/slurm_train.sh {cur_partition} ' + \
                          f'{cur_task_name} ' + \
                          f'{cur_config} {cur_work_dir} ' + \
                          f'{py_cmd}'
                    os.system(f'tmux send-keys "{cmd}" "C-m"')

                cur_port += 1
                # if the port is used, use a random number for port
                while not is_port_available(cur_port, ip):
                    cur_port = random.randint(29000, 39000)
                print(f'port used in task {cur_task} is: {cur_port}')
                cur_task += 1

    # close the base window
    os.system('tmux select-window -t 0')
    cmd = 'tmux kill-window -t 0'
    os.system(f'tmux send-keys -t {args.session_name} "{cmd}" "C-m"')

    print('All tasks submitted!')


if __name__ == '__main__':
    main()
