# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch
from mmcv.image import imread, imwrite
from mmengine.utils import is_list_of
from parameterized import parameterized

from mmpose.apis import inference_bottomup, inference_topdown, init_model
from mmpose.structures import PoseDataSample
from mmpose.testing._utils import _rand_bboxes, get_config_file, get_repo_dir
from mmpose.utils import register_all_modules


class TestInference(TestCase):

    def setUp(self) -> None:
        register_all_modules()

    @parameterized.expand([(('configs/body_2d_keypoint/topdown_heatmap/coco/'
                             'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'),
                            ('cpu', 'cuda'))])
    def test_init_model(self, config, devices):
        config_file = get_config_file(config)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue

            # test init_model with str path
            _ = init_model(config_file, device=device)

            # test init_model with :obj:`Path`
            _ = init_model(Path(config_file), device=device)

            # test init_detector with undesirable type
            with self.assertRaisesRegex(
                    TypeError, 'config must be a filename or Config object'):
                config_list = [config_file]
                _ = init_model(config_list)

    @parameterized.expand([(('configs/body_2d_keypoint/topdown_heatmap/coco/'
                             'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'),
                            ('cpu', 'cuda'))])
    def test_inference_topdown(self, config, devices):
        project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        project_dir = osp.join(project_dir, '..')
        config_file = osp.join(project_dir, config)

        rng = np.random.RandomState(0)
        img_w = img_h = 100
        img = rng.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        bboxes = _rand_bboxes(rng, 2, img_w, img_h)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue
            model = init_model(config_file, device=device)

            # test inference with bboxes
            results = inference_topdown(model, img, bboxes, bbox_format='xywh')
            self.assertTrue(is_list_of(results, PoseDataSample))
            self.assertEqual(len(results), 2)
            self.assertTrue(results[0].pred_instances.keypoints.shape,
                            (1, 17, 2))

            # test inference without bbox
            results = inference_topdown(model, img)
            self.assertTrue(is_list_of(results, PoseDataSample))
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].pred_instances.keypoints.shape,
                            (1, 17, 2))

            # test inference from image file
            with TemporaryDirectory() as tmp_dir:
                img_path = osp.join(tmp_dir, 'img.jpg')
                imwrite(img, img_path)

                results = inference_topdown(model, img_path)
                self.assertTrue(is_list_of(results, PoseDataSample))
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].pred_instances.keypoints.shape,
                                (1, 17, 2))

    @parameterized.expand([(('configs/body_2d_keypoint/'
                             'associative_embedding/coco/'
                             'ae_hrnet-w32_8xb24-300e_coco-512x512.py'),
                            ('cpu', 'cuda'))])
    def test_inference_bottomup(self, config, devices):
        config_file = get_config_file(config)
        img = osp.join(get_repo_dir(), 'tests/data/coco/000000000785.jpg')

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue
            model = init_model(config_file, device=device)

            # test inference from image
            results = inference_bottomup(model, img=imread(img))
            self.assertTrue(is_list_of(results, PoseDataSample))
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].pred_instances.keypoints.shape,
                            (1, 17, 2))

            # test inference from file
            results = inference_bottomup(model, img=img)
            self.assertTrue(is_list_of(results, PoseDataSample))
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].pred_instances.keypoints.shape,
                            (1, 17, 2))
