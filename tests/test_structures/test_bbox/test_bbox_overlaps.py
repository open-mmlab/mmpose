# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.structures.bbox import bbox_overlaps  # Import your function here


class TestBBoxOverlaps(TestCase):

    def test_bbox_overlaps_iou(self):
        bboxes1 = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [32, 32, 38, 42],
        ])
        bboxes2 = torch.FloatTensor([
            [0, 0, 10, 20],
            [0, 10, 10, 19],
            [10, 10, 20, 20],
        ])
        overlaps = bbox_overlaps(bboxes1, bboxes2)

        expected_overlaps = torch.FloatTensor([
            [0.5000, 0.0000, 0.0000],
            [0.0000, 0.0000, 1.0000],
            [0.0000, 0.0000, 0.0000],
        ])

        self.assertTrue(
            torch.allclose(overlaps, expected_overlaps, rtol=1e-4, atol=1e-4))

    def test_bbox_overlaps_iof(self):
        bboxes1 = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [32, 32, 38, 42],
        ])
        bboxes2 = torch.FloatTensor([
            [0, 0, 10, 20],
            [0, 10, 10, 19],
            [10, 10, 20, 20],
        ])
        overlaps = bbox_overlaps(bboxes1, bboxes2, mode='iof')

        expected_overlaps = torch.FloatTensor([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])

        self.assertTrue(
            torch.allclose(overlaps, expected_overlaps, rtol=1e-4, atol=1e-4))

    def test_bbox_overlaps_giou(self):
        bboxes1 = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [32, 32, 38, 42],
        ])
        bboxes2 = torch.FloatTensor([
            [0, 0, 10, 20],
            [0, 10, 10, 19],
            [10, 10, 20, 20],
        ])
        overlaps = bbox_overlaps(bboxes1, bboxes2, mode='giou')

        expected_overlaps = torch.FloatTensor([
            [0.5000, 0.0000, -0.5000],
            [-0.2500, -0.0500, 1.0000],
            [-0.8371, -0.8766, -0.8214],
        ])

        self.assertTrue(
            torch.allclose(overlaps, expected_overlaps, rtol=1e-4, atol=1e-4))
