# Copyright (c) OpenMMLab. All rights reserved.
import base64
import os

import mmcv
import torch

from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model)
from mmpose.models.detectors import AssociativeEmbedding, TopDown

try:
    from ts.torch_handler.base_handler import BaseHandler
except ImportError:
    raise ImportError('Please install torchserve.')


class MMPoseHandler(BaseHandler):

    def initialize(self, context):
        properties = context.system_properties
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.map_location + ':' +
                                   str(properties.get('gpu_id')) if torch.cuda.
                                   is_available() else self.map_location)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint = os.path.join(model_dir, serialized_file)
        self.config_file = os.path.join(model_dir, 'config.py')

        self.model = init_pose_model(self.config_file, checkpoint, self.device)
        self.initialized = True

    def preprocess(self, data):
        images = []

        for row in data:
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            images.append(image)

        return images

    def inference(self, data, *args, **kwargs):
        if isinstance(self.model, TopDown):
            results = self._inference_top_down_pose_model(data)
        elif isinstance(self.model, (AssociativeEmbedding, )):
            results = self._inference_bottom_up_pose_model(data)
        else:
            raise NotImplementedError(
                f'Model type {type(self.model)} is not supported.')

        return results

    def _inference_top_down_pose_model(self, data):
        results = []
        for image in data:
            # use dummy person bounding box
            preds, _ = inference_top_down_pose_model(
                self.model, image, person_results=None)
            results.append(preds)
        return results

    def _inference_bottom_up_pose_model(self, data):
        results = []
        for image in data:
            preds, _ = inference_bottom_up_pose_model(self.model, image)
            results.append(preds)
        return results

    def postprocess(self, data):
        output = [[{
            'keypoints': pred['keypoints'].tolist()
        } for pred in preds] for preds in data]

        return output
