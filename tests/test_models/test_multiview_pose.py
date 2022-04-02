# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import torch
from mmcv import Config

from mmpose.datasets import DATASETS, build_dataloader
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.models import builder


def test_voxelpose_forward():
    dataset = 'Body3DMviewDirectPanopticDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/panoptic_body3d.py').dataset_info
    space_size = [8000, 8000, 2000]
    space_center = [0, -500, 800]
    cube_size = [20, 20, 8]
    data_cfg = dict(
        image_size=[960, 512],
        heatmap_size=[[240, 128]],
        space_size=space_size,
        space_center=space_center,
        cube_size=cube_size,
        num_joints=15,
        seq_list=['160906_band1'],
        cam_list=[(0, 12), (0, 6)],
        num_cameras=2,
        seq_frame_interval=1,
        subset='train',
        need_2d_label=True,
        need_camera_param=True,
        root_id=2)

    pipeline_heatmap = [
        dict(
            type='MultiItemProcess',
            pipeline=[
                dict(
                    type='BottomUpGenerateTarget', sigma=3, max_num_people=20)
            ]),
        dict(
            type='DiscardDuplicatedItems',
            keys_list=[
                'joints_3d', 'joints_3d_visible', 'ann_info', 'roots_3d',
                'num_persons', 'sample_id'
            ]),
        dict(
            type='GenerateVoxel3DHeatmapTarget',
            sigma=200.0,
            joint_indices=[2]),
        dict(type='RenameKeys', key_pairs=[('targets', 'input_heatmaps')]),
        dict(
            type='Collect',
            keys=['targets_3d', 'input_heatmaps'],
            meta_keys=[
                'camera', 'center', 'scale', 'joints_3d', 'num_persons',
                'joints_3d_visible', 'roots_3d', 'sample_id'
            ]),
    ]

    model_cfg = dict(
        type='DetectAndRegress',
        backbone=None,
        human_detector=dict(
            type='VoxelCenterDetector',
            image_size=[960, 512],
            heatmap_size=[240, 128],
            space_size=space_size,
            cube_size=cube_size,
            space_center=space_center,
            center_net=dict(
                type='V2VNet', input_channels=15, output_channels=1),
            center_head=dict(
                type='CuboidCenterHead',
                space_size=space_size,
                space_center=space_center,
                cube_size=cube_size,
                max_num=3,
                max_pool_kernel=3),
            train_cfg=dict(dist_threshold=500000000.0),
            test_cfg=dict(center_threshold=0.0),
        ),
        pose_regressor=dict(
            type='VoxelSinglePose',
            image_size=[960, 512],
            heatmap_size=[240, 128],
            sub_space_size=[2000, 2000, 2000],
            sub_cube_size=[20, 20, 8],
            num_joints=15,
            pose_net=dict(
                type='V2VNet', input_channels=15, output_channels=15),
            pose_head=dict(type='CuboidPoseHead', beta=100.0),
            train_cfg=None,
            test_cfg=None))

    model = builder.build_posenet(model_cfg)
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dataset_class(
            ann_file=tmpdir + '/tmp_train.pkl',
            img_prefix='tests/data/panoptic_body3d/',
            data_cfg=data_cfg,
            pipeline=pipeline_heatmap,
            dataset_info=dataset_info,
            test_mode=False)

    data_loader = build_dataloader(
        dataset,
        seed=None,
        dist=False,
        shuffle=False,
        drop_last=False,
        workers_per_gpu=1,
        samples_per_gpu=1)

    with torch.no_grad():
        for data in data_loader:
            # test forward_train
            _ = model(
                img=None,
                img_metas=data['img_metas'].data[0],
                return_loss=True,
                targets_3d=data['targets_3d'],
                input_heatmaps=data['input_heatmaps'])

            # test forward_test
            _ = model(
                img=None,
                img_metas=data['img_metas'].data[0],
                return_loss=False,
                input_heatmaps=data['input_heatmaps'])

            with tempfile.TemporaryDirectory() as tmpdir:
                model.show_result(
                    img=None,
                    img_metas=data['img_metas'].data[0],
                    input_heatmaps=data['input_heatmaps'],
                    dataset_info=DatasetInfo(dataset_info),
                    out_dir=tmpdir,
                    visualize_2d=True)
