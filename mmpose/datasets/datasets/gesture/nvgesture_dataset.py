# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from collections import defaultdict

from mmcv import Config

from ...builder import DATASETS
from .gesture_base_dataset import GestureBaseDataset


@DATASETS.register_module()
class NVGestureDataset(GestureBaseDataset):
    """NVGesture dataset for gesture recognition."""

    def __init__(self,
                 ann_file,
                 vid_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/nvgesture.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            vid_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)

        self.db = self._get_db()
        self.vid_ids = list(range(len(self.db)))
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        """Load dataset."""
        db = []
        with open(self.ann_file, 'r') as f:
            samples = f.readlines()
        for sample in samples:
            sample = sample.strip().split()
            sample = {
                item.split(':', 1)[0]: item.split(':', 1)[1]
                for item in sample
            }

            for key in ('depth', 'color'):
                fname, start, end = sample[key].split(':')
                sample[key] = {
                    'path': os.path.join(sample['path'][2:], fname + '.avi'),
                    'valid_frames': (eval(start), eval(end))
                }
            sample['flow'] = {
                'path': sample['color']['path'].replace('color', 'flow'),
                'valid_frames': sample['color']['valid_frames']
            }
            sample['rgb'] = sample['color']
            sample['label'] = eval(sample['label']) - 1

            del sample['path'], sample['duo_left'], sample['color']
            db.append(sample)

        return db

    def _get_single(self, idx):
        """Get anno for a single video."""
        anno = defaultdict(list)
        sample = self.db[self.vid_ids[idx]]

        anno['label'] = sample['label']
        anno['modalities'] = self.modalities

        for modality in self.modalities:
            anno['video_file'].append(
                os.path.join(self.vid_prefix, sample[modality]['path']))
            anno['valid_frames'].append(sample[modality]['valid_frames'])

        return anno
