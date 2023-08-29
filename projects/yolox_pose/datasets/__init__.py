import mmengine
import mmyolo

compatible_version = '0.5.0'
if mmengine.digit_version(mmyolo.__version__)[1] > \
        mmengine.digit_version(compatible_version)[1]:
    print(f'This project is only compatible with mmyolo {compatible_version} '
          f'or lower. Please install the required version via:'
          f'pip install mmyolo=={compatible_version}')

from .bbox_keypoint_structure import *  # noqa
from .coco_dataset import *  # noqa
from .transforms import *  # noqa
