# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from collections import defaultdict

import json_tricks as json
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
        anno['modality'] = self.modality

        for modal in self.modality:
            anno['video_file'].append(
                os.path.join(self.vid_prefix, sample[modal]['path']))
            anno['valid_frames'].append(sample[modal]['valid_frames'])

        return anno

    def evaluate(self, results, res_folder=None, metric='AP', **kwargs):
        if metric != 'AP':
            raise ValueError(f'Metric {metric} is invalid. Pls use \'AP\'.')

        accuracy_buffer = defaultdict(list)
        num_samples = 0
        for result in results:
            for modal in result['logits']:
                logit = result['logits'][modal].mean(dim=2)
                acc = (logit.argmax(dim=1) == result['label']).int().sum()
                accuracy_buffer[modal].append(acc.item())
            num_samples += len(result['label'])

        accuracy = dict()
        for modal in accuracy_buffer:
            correct = sum(accuracy_buffer[modal])
            accuracy[f'AP_{modal}'] = correct / num_samples
        accuracy['mAP'] = sum(accuracy.values()) / len(accuracy)

        if res_folder is not None:
            with open(osp.join(res_folder, 'accuracy.json'), 'w') as f:
                json.dump(accuracy, f, indent=4)

        return accuracy
