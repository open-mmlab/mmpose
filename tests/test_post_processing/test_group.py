# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.core.post_processing.group import HeatmapParser


def test_group():
    cfg = {}
    cfg['num_joints'] = 17
    cfg['detection_threshold'] = 0.1
    cfg['tag_threshold'] = 1
    cfg['use_detection_val'] = True
    cfg['ignore_too_much'] = False
    cfg['nms_kernel'] = 5
    cfg['nms_padding'] = 2
    cfg['tag_per_joint'] = True
    cfg['max_num_people'] = 1
    parser = HeatmapParser(cfg)
    fake_heatmap = torch.zeros(1, 1, 5, 5)
    fake_heatmap[0, 0, 3, 3] = 1
    fake_heatmap[0, 0, 3, 2] = 0.8
    assert parser.nms(fake_heatmap)[0, 0, 3, 2] == 0
    fake_heatmap = torch.zeros(1, 17, 32, 32)
    fake_tag = torch.zeros(1, 17, 32, 32, 1)
    fake_heatmap[0, 0, 10, 10] = 0.8
    fake_heatmap[0, 1, 12, 12] = 0.9
    fake_heatmap[0, 4, 8, 8] = 0.8
    fake_heatmap[0, 8, 6, 6] = 0.9
    fake_tag[0, 0, 10, 10] = 0.8
    fake_tag[0, 1, 12, 12] = 0.9
    fake_tag[0, 4, 8, 8] = 0.8
    fake_tag[0, 8, 6, 6] = 0.9
    grouped, scores = parser.parse(fake_heatmap, fake_tag, True, True)
    assert grouped[0][0, 0, 0] == 10.25
    assert abs(scores[0] - 0.2) < 0.001
    cfg['tag_per_joint'] = False
    parser = HeatmapParser(cfg)
    grouped, scores = parser.parse(fake_heatmap, fake_tag, False, False)
    assert grouped[0][0, 0, 0] == 10.
    grouped, scores = parser.parse(fake_heatmap, fake_tag, False, True)
    assert grouped[0][0, 0, 0] == 10.


def test_group_score_per_joint():
    cfg = {}
    cfg['num_joints'] = 17
    cfg['detection_threshold'] = 0.1
    cfg['tag_threshold'] = 1
    cfg['use_detection_val'] = True
    cfg['ignore_too_much'] = False
    cfg['nms_kernel'] = 5
    cfg['nms_padding'] = 2
    cfg['tag_per_joint'] = True
    cfg['max_num_people'] = 1
    cfg['score_per_joint'] = True
    parser = HeatmapParser(cfg)
    fake_heatmap = torch.zeros(1, 1, 5, 5)
    fake_heatmap[0, 0, 3, 3] = 1
    fake_heatmap[0, 0, 3, 2] = 0.8
    assert parser.nms(fake_heatmap)[0, 0, 3, 2] == 0
    fake_heatmap = torch.zeros(1, 17, 32, 32)
    fake_tag = torch.zeros(1, 17, 32, 32, 1)
    fake_heatmap[0, 0, 10, 10] = 0.8
    fake_heatmap[0, 1, 12, 12] = 0.9
    fake_heatmap[0, 4, 8, 8] = 0.8
    fake_heatmap[0, 8, 6, 6] = 0.9
    fake_tag[0, 0, 10, 10] = 0.8
    fake_tag[0, 1, 12, 12] = 0.9
    fake_tag[0, 4, 8, 8] = 0.8
    fake_tag[0, 8, 6, 6] = 0.9
    grouped, scores = parser.parse(fake_heatmap, fake_tag, True, True)
    assert len(scores[0]) == 17


def test_group_ignore_too_much():
    cfg = {}
    cfg['num_joints'] = 17
    cfg['detection_threshold'] = 0.1
    cfg['tag_threshold'] = 1
    cfg['use_detection_val'] = True
    cfg['ignore_too_much'] = True
    cfg['nms_kernel'] = 5
    cfg['nms_padding'] = 2
    cfg['tag_per_joint'] = True
    cfg['max_num_people'] = 1
    cfg['score_per_joint'] = True
    parser = HeatmapParser(cfg)
    fake_heatmap = torch.zeros(1, 1, 5, 5)
    fake_heatmap[0, 0, 3, 3] = 1
    fake_heatmap[0, 0, 3, 2] = 0.8
    assert parser.nms(fake_heatmap)[0, 0, 3, 2] == 0
    fake_heatmap = torch.zeros(1, 17, 32, 32)
    fake_tag = torch.zeros(1, 17, 32, 32, 1)
    fake_heatmap[0, 0, 10, 10] = 0.8
    fake_heatmap[0, 1, 12, 12] = 0.9
    fake_heatmap[0, 4, 8, 8] = 0.8
    fake_heatmap[0, 8, 6, 6] = 0.9
    fake_tag[0, 0, 10, 10] = 0.8
    fake_tag[0, 1, 12, 12] = 0.9
    fake_tag[0, 4, 8, 8] = 0.8
    fake_tag[0, 8, 6, 6] = 2.0
    grouped, sc = parser.parse(fake_heatmap, fake_tag, True, True)
    assert len(grouped[0]) == 1

    cfg['ignore_too_much'] = False
    parser_noignore = HeatmapParser(cfg)
    grouped, sc = parser_noignore.parse(fake_heatmap, fake_tag, True, True)
    assert len(grouped[0]) == 2
