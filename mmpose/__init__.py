import mmcv
from mmcv import digit_version, parse_version_info

from .version import __version__, short_version

mmcv_minimum_version = '1.1.3'
mmcv_maximum_version = '1.3'
mmcv_version = digit_version(mmcv.__version__)
version_info = parse_version_info(__version__)

assert digit_version(mmcv_minimum_version) <= mmcv_version, \
        f'MMCV=={mmcv.__version__} is used but incompatible. ' \
        f'Please install mmcv>={mmcv_minimum_version}.'

assert digit_version(mmcv_maximum_version) > mmcv_version, \
        f'MMCV=={mmcv.__version__} is used but incompatible. ' \
        f'Please install mmcv<{mmcv_maximum_version}.'

__all__ = ['__version__', 'short_version', 'version_info']
