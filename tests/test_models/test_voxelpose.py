# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

from mmcv import Config

from mmpose.datasets import DATASETS, build_dataloader
from mmpose.models import builder
from mmpose.models.detectors.voxelpose import (CuboidProposalNet,
                                               PoseRegressionNet, ProjectLayer)


def test_voxelpose_forward():
    dataset = 'Body3DMviewDirectPanopticDataset'
    dataset_class = DATASETS.get(dataset)
    dataset_info = Config.fromfile(
        'configs/_base_/datasets/panoptic_body3d.py').dataset_info
    space_size = [8000, 8000, 2000]
    space_center = [0, -500, 800]
    cube_size = [20, 20, 8]
    train_data_cfg = dict(
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

    pipeline = [
        dict(
            type='MultiItemProcess',
            pipeline=[
                dict(
                    type='BottomUpGenerateTarget', sigma=3, max_num_people=30)
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
            meta_keys=['camera', 'center', 'scale', 'joints_3d']),
    ]

    project_layer = ProjectLayer(
        dict(image_size=[960, 512], heatmap_size=[240, 128]))
    root_net = CuboidProposalNet(
        dict(type='V2VNet', input_channels=15, output_channels=1))
    center_head = builder.build_head(
        dict(
            type='CuboidCenterHead',
            cfg=dict(
                space_size=space_size,
                space_center=space_center,
                cube_size=cube_size,
                max_num=10,
                max_pool_kernel=3)))
    pose_net = PoseRegressionNet(
        dict(type='V2VNet', input_channels=15, output_channels=15))
    pose_head = builder.build_head(dict(type='CuboidPoseHead', beta=100.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = dataset_class(
            ann_file=tmpdir + '/tmp_train.pkl',
            img_prefix='tests/data/panoptic_body3d/',
            data_cfg=train_data_cfg,
            pipeline=pipeline,
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

    for data in data_loader:
        initial_cubes, _ = project_layer(
            [htm[0] for htm in data['input_heatmaps']],
            data['img_metas'].data[0], space_size, [space_center], cube_size)
        _ = root_net(initial_cubes)
        center_candidates = center_head(data['targets_3d'])
        center_candidates[..., 3] = \
            (center_candidates[..., 4] > 0.5).float() - 1.0

        batch_size, num_candidates, _ = center_candidates.shape

        for n in range(num_candidates):
            index = center_candidates[:, n, 3] >= 0
            num_valid = index.sum()
            if num_valid > 0:
                pose_input_cube, coordinates \
                    = project_layer([htm[0] for htm in data['input_heatmaps']],
                                    data['img_metas'].data[0],
                                    [800, 800, 800],
                                    center_candidates[:, n, :3],
                                    [8, 8, 8])
                pose_heatmaps_3d = pose_net(pose_input_cube)
                _ = pose_head(pose_heatmaps_3d[index], coordinates[index])
