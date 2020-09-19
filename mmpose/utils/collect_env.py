from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash

import mmpose


def get_short_git_hash(num_hashes=7):
    return get_git_hash()[:num_hashes]


def collect_env():
    env_info = collect_basic_env()
    env_info['MMPose'] = (mmpose.__version__ + '+' + get_short_git_hash())
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
