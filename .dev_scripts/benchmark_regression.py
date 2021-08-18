import argparse
import datetime
import math
import os

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='running benchmark regression with tmux')
    parser.add_argument(
        '--config',
        help='test config file path',
        default='./.dev_scripts/benchmark_regression_cfg.yaml')
    parser.add_argument(
        '--priority',
        help='the largest number of model priority that will be run',
        default=2)

    # runtime setting parameters
    parser.add_argument(
        '--root-work-dir', help='the root working directory to store logs')

    parser.add_argument(
        '--session-name', '-s', help='the tmux session name', default='test')

    parser.add_argument(
        '--panes-per-window',
        help='the maximum number of panes in each tmux window',
        default=12)

    parser.add_argument(
        '--env',
        help='the conda environment used to run the tasks',
        default='pt1.6')
    parser.add_argument(
        '--partition', help='the partition name', default='openmmlab')

    parser.add_argument('--gpus', help='the total number of GPUs', default=8)
    parser.add_argument(
        '--gpus-per-node',
        default=8,
        help='the number of GPUs used per computing node',
        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument(
        '--cpus-per-task', default=5, help='the number of CPUs used per task')
    parser.add_argument('--port', default=29666, help='the starting port used')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.root_work_dir is None:
        date = datetime.datetime.now()
        args.root_work_dir = 'work_dirs/benchmark_regression_' + \
                             f'{date.year}{date.month:02d}{date.day:02d}_' + \
                             f'{date.hour:02d}{date.minute:02d}'
    mmcv.mkdir_or_exist(os.path.abspath(args.root_work_dir))

    cfg = mmcv.load(args.config)

    # filter out the models with less priority
    if 'inference' in cfg['model_list'].keys():
        ori_len = len(cfg['model_list']['inference'])
        model_cfg_infer = [
            cfg['model_list']['inference'][i] for i in range(ori_len)
            if cfg['model_list']['inference'][i]['priority'] <= args.priority
        ]
        num_infer_task = len(model_cfg_infer)
        num_infer_win = math.ceil(num_infer_task / args.panes_per_window)
    else:
        num_infer_task = 0
        num_infer_win = 0

    if 'train' in cfg['model_list'].keys():
        ori_len = len(cfg['model_list']['train'])
        model_cfg_train = [
            cfg['model_list']['train'][i] for i in range(ori_len)
            if cfg['model_list']['train'][i]['priority'] <= args.priority
        ]
        num_train_task = len(model_cfg_train)
        num_train_win = math.ceil(num_train_task / args.panes_per_window)
    else:
        num_train_task = 0
        num_train_win = 0

    # create a new tmux session
    os.system(f'tmux new -s {args.session_name} -d')

    # tmux select-window -t 0
    os.system('tmux select-window -t 0')

    num_infer_task_tmp = num_infer_task
    num_train_task_tmp = num_train_task

    # create new windows
    for i in range(num_infer_win):
        # tmux select-window -t 0
        os.system('tmux select-window -t 0')
        # tmux new-window -n infer_1
        os.system(f'tmux new-window -n infer_{i+1}')

        if num_infer_task_tmp >= args.panes_per_window:
            num_cur_task = args.panes_per_window
            num_infer_task_tmp -= args.panes_per_window
        else:
            num_cur_task = num_infer_task_tmp

        # split each window into different panes
        for j in range(num_cur_task - 1):
            ratio = int(100 - 100 / (num_cur_task - j))
            os.system(f'tmux split-window -h -p {ratio:02d}')

        os.system('tmux select-layout tiled')

    for i in range(num_train_win):
        # tmux select-window - t 0
        os.system('tmux select-window -t 0')
        # tmux new-window -n train_1
        os.system(f'tmux new-window -n train_{i+1}')

        if num_train_task_tmp >= args.panes_per_window:
            num_cur_task = args.panes_per_window
            num_train_task_tmp -= args.panes_per_window
        else:
            num_cur_task = num_train_task_tmp

        # split each window into different panes
        for j in range(num_cur_task - 1):
            ratio = int(100 - 100 / (num_cur_task - j))
            os.system(f'tmux split-window -h -p {ratio:02d}')

        os.system('tmux select-layout tiled')

    port = args.port
    # select window and pane to run
    if num_infer_task != 0:
        num_infer_task_tmp = num_infer_task
        for i in range(num_infer_win):
            os.system('tmux select-window -t 0')
            # tmux select-window -t infer_j
            os.system(f'tmux select-window -t infer_{i+1}')

            if num_infer_task_tmp >= args.panes_per_window:
                num_cur_task = args.panes_per_window
                num_infer_task_tmp -= args.panes_per_window
            else:
                num_cur_task = num_infer_task_tmp

            for j in range(num_cur_task):
                os.system(f'tmux select-pane -t {j}')

                cmd = f'conda activate {args.env}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')
                cmd = 'echo executing infer task: ' + \
                      f'{i*args.panes_per_window+j+1}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')

                cur_cfg = model_cfg_infer[i * args.panes_per_window + j]

                if 'port' in cur_cfg.keys():
                    cur_port = cur_cfg['port']
                else:
                    cur_port = port
                    port += 1

                if 'gpus' in cur_cfg.keys():
                    cur_gpus = cur_cfg['gpus']
                else:
                    cur_gpus = args.gpus

                if 'gpus_per_node' in cur_cfg.keys():
                    cur_gpus_per_node = cur_cfg['gpus_per_node']
                else:
                    cur_gpus_per_node = args.gpus_per_node

                if 'cpus_per_task' in cur_cfg.keys():
                    cur_cpus_per_task = cur_cfg['cpus_per_task']
                else:
                    cur_cpus_per_task = args.cpus_per_task

                if 'partition' in cur_cfg.keys():
                    cur_partition = cur_cfg['partition']
                else:
                    cur_partition = args.partition

                if 'task_name' in cur_cfg.keys():
                    cur_task_name = cur_cfg['task_name']
                else:
                    cur_task_name = os.path.basename(
                        cur_cfg['config'].split('.')[-2])

                cur_work_dir = os.path.join(args.root_work_dir, cur_task_name)

                # deal with extra python arguments
                py_cmd = f' --work-dir {cur_work_dir} '
                if 'py_args' in cur_cfg.keys():
                    keys = list(cur_cfg['py_args'].keys())
                    values = list(cur_cfg['py_args'].values())

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
                      f"{cur_cfg['config']} {cur_cfg['checkpoint']} " + \
                      f'{py_cmd}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')

    if num_train_task != 0:
        num_train_task_tmp = num_train_task
        for i in range(num_train_win):
            os.system('tmux select-window -t 0')
            # tmux select-window -t infer_j
            os.system(f'tmux select-window -t train_{i+1}')

            if num_train_task_tmp >= args.panes_per_window:
                num_cur_task = args.panes_per_window
                num_train_task_tmp -= args.panes_per_window
            else:
                num_cur_task = num_train_task_tmp

            for j in range(num_cur_task):
                os.system(f'tmux select-pane -t {j}')

                cmd = f'conda activate {args.env}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')
                cmd = 'echo executing train task: ' + \
                      f'{i*args.panes_per_window+j+1}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')

                cur_cfg = model_cfg_train[i * args.panes_per_window + j]

                if 'port' in cur_cfg.keys():
                    cur_port = cur_cfg['port']
                else:
                    cur_port = port
                    port += 1

                if 'gpus' in cur_cfg.keys():
                    cur_gpus = cur_cfg['gpus']
                else:
                    cur_gpus = args.gpus

                if 'gpus_per_node' in cur_cfg.keys():
                    cur_gpus_per_node = cur_cfg['gpus_per_node']
                else:
                    cur_gpus_per_node = args.gpus_per_node

                if 'cpus_per_task' in cur_cfg.keys():
                    cur_cpus_per_task = cur_cfg['cpus_per_task']
                else:
                    cur_cpus_per_task = args.cpus_per_task

                if 'partition' in cur_cfg.keys():
                    cur_partition = cur_cfg['partition']
                else:
                    cur_partition = args.partition

                if 'task_name' in cur_cfg.keys():
                    cur_task_name = cur_cfg['task_name']
                else:
                    cur_task_name = os.path.basename(
                        cur_cfg['config'].split('.')[-2])

                cur_work_dir = os.path.join(args.root_work_dir, cur_task_name)

                py_cmd = ' '
                # deal with extra python arguments
                if 'py_args' in cur_cfg.keys():
                    keys = list(cur_cfg['py_args'].keys())
                    values = list(cur_cfg['py_args'].values())

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
                      f"{cur_cfg['config']} {cur_work_dir} " + \
                      f'{py_cmd}'
                os.system(f'tmux send-keys "{cmd}" "C-m"')

    # close the base window
    os.system('tmux select-window -t 0')
    cmd = 'tmux kill-window -t 0'
    os.system(f'tmux send-keys -t {args.session_name} "{cmd}" "C-m"')


if __name__ == '__main__':
    main()
