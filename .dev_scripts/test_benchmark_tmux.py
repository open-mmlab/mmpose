import argparse
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description='running benchmark regression with tmux')
    parser.add_argument(
        '--config',
        help='test config file path',
        default='./.dev_scripts/config_list.yaml')

    # runtime setting parameters
    parser.add_argument(
        '--session_name', '-s', help='the tmux session name', default='test')
    parser.add_argument(
        '--path',
        help='the running path, e.g., $mmpose',
        default='~/open_mmlab/mmpose')
    parser.add_argument('--env', help='the conda environment', default='pt1.6')
    parser.add_argument(
        '--partition', help='the partition name', default='openmmlab')

    parser.add_argument('--gpus', help='the total number of GPUs', default=2)
    parser.add_argument(
        '--gpus_per_node',
        default=2,
        help='the number of GPUs used per computing node',
        choices=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument(
        '--cpus_per_task', default=2, help='the number of CPUs used per task')
    parser.add_argument('--port', default=29666, help='the starting port used')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # cd $mmpose
    os.system('cd {}'.format(args.path))

    # create a new tmux session
    os.system('tmux new -s {} -d'.format(args.session_name))

    # tmux select-window - t 0
    os.system('tmux select-window -t 0')
    # pwd
    os.system('tmux send-keys "%s" "C-m"' % 'pwd')
    # ls
    os.system('tmux send-keys "%s" "C-m"' % 'ls')

    with open(args.config) as config_file:
        config_list = yaml.load(config_file.read())

    port = args.port
    window_index = 1

    for model_name, model in config_list.items():
        task_name = model_name
        config = model['config']
        ckpt = model['checkpoint']

        # tmux select-window - t 0
        os.system('tmux select-window -t 0')
        # tmux new-window -n hrnet_w32_coco_256x192
        os.system('tmux new-window -n {}'.format(task_name))

        # tmux select-window - t window_index
        os.system('tmux select-window -t {}'.format(window_index))

        cmd = 'conda activate {}'.format(args.env)
        os.system('tmux send-keys "%s" "C-m"' % cmd)

        cmd = 'MASTER_PORT={} GPUS={} GPUS_PER_NODE={} CPUS_PER_TASK={} \
            ./tools/slurm_test.sh {} {} {} {}'.format(port, args.gpus,
                                                      args.gpus_per_node,
                                                      args.cpus_per_task,
                                                      args.partition,
                                                      task_name, config, ckpt)
        os.system('tmux send-keys "%s" "C-m"' % cmd)

        window_index += 1
        port += 1


if __name__ == '__main__':
    main()
