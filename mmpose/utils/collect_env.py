from mmcv.utils import collect_env as collect_basic_env

import mmpose


def collect_env():
    env_info = collect_basic_env()
    env_info['MMPose'] = mmpose.__version__
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
