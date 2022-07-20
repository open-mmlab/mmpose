# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmpose.core import SimpleCamera
from mmpose.datasets.pipelines import Compose

H36M_JOINT_IDX = [14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]


def get_data_sample():

    def _parse_h36m_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        A typical h36m image filename is like:
        S1_Directions_1.54138969_000001.jpg
        """
        subj, rest = osp.basename(imgname).split('_', 1)
        action, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        return subj, action, camera

    ann_flle = 'tests/data/h36m/test_h36m.npz'
    camera_param_file = 'tests/data/h36m/cameras.pkl'

    data = np.load(ann_flle)
    cameras = mmcv.load(camera_param_file)

    _imgnames = data['imgname']
    _joints_2d = data['part'][:, H36M_JOINT_IDX].astype(np.float32)
    _joints_3d = data['S'][:, H36M_JOINT_IDX].astype(np.float32)
    _centers = data['center'].astype(np.float32)
    _scales = data['scale'].astype(np.float32)

    frame_ids = [0]
    target_frame_id = 0

    results = {
        'frame_ids': frame_ids,
        'target_frame_id': target_frame_id,
        'input_2d': _joints_2d[frame_ids, :, :2],
        'input_2d_visible': _joints_2d[frame_ids, :, -1:],
        'input_3d': _joints_3d[frame_ids, :, :3],
        'input_3d_visible': _joints_3d[frame_ids, :, -1:],
        'target': _joints_3d[target_frame_id, :, :3],
        'target_visible': _joints_3d[target_frame_id, :, -1:],
        'imgnames': _imgnames[frame_ids],
        'scales': _scales[frame_ids],
        'centers': _centers[frame_ids],
    }

    # add camera parameters
    subj, _, camera = _parse_h36m_imgname(_imgnames[frame_ids[0]])
    results['camera_param'] = cameras[(subj, camera)]

    # add image size
    results['image_width'] = results['camera_param']['w']
    results['image_height'] = results['camera_param']['h']

    # add ann_info
    ann_info = {}
    ann_info['num_joints'] = 17
    ann_info['joint_weights'] = np.full(17, 1.0, dtype=np.float32)
    ann_info['flip_pairs'] = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15],
                              [13, 16]]
    ann_info['upper_body_ids'] = (0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    ann_info['lower_body_ids'] = (1, 2, 3, 4, 5, 6)
    ann_info['use_different_joint_weights'] = False

    results['ann_info'] = ann_info

    return results


def test_joint_transforms():
    results = get_data_sample()

    mean = np.random.rand(16, 3).astype(np.float32)
    std = np.random.rand(16, 3).astype(np.float32) + 1e-6

    pipeline = [
        dict(
            type='RelativeJointRandomFlip',
            item='target',
            flip_cfg=dict(center_mode='root', center_index=0),
            visible_item='target_visible',
            flip_prob=1.,
            flip_camera=True),
        dict(
            type='GetRootCenteredPose',
            item='target',
            root_index=0,
            root_name='global_position',
            remove_root=True),
        dict(
            type='NormalizeJointCoordinate', item='target', mean=mean,
            std=std),
        dict(type='PoseSequenceToTensor', item='target'),
        dict(
            type='ImageCoordinateNormalization',
            item='input_2d',
            norm_camera=True),
        dict(type='CollectCameraIntrinsics'),
        dict(
            type='Collect',
            keys=[('input_2d', 'input'), ('target', 'output'), 'flip_pairs',
                  'intrinsics'],
            meta_name='metas',
            meta_keys=['camera_param'])
    ]

    pipeline = Compose(pipeline)
    output = pipeline(copy.deepcopy(results))

    # test transformation of target
    joints_0 = results['target']
    joints_1 = output['output'].numpy()
    # manually do transformations
    flip_pairs = output['flip_pairs']
    _joints_0_flipped = joints_0.copy()
    for _l, _r in flip_pairs:
        _joints_0_flipped[..., _l, :] = joints_0[..., _r, :]
        _joints_0_flipped[..., _r, :] = joints_0[..., _l, :]
    _joints_0_flipped[...,
                      0] = 2 * joints_0[..., 0:1, 0] - _joints_0_flipped[...,
                                                                         0]
    joints_0 = _joints_0_flipped
    joints_0 = (joints_0[..., 1:, :] - joints_0[..., 0:1, :] - mean) / std
    # convert to [K*C, T]
    joints_0 = joints_0.reshape(-1)[..., None]
    np.testing.assert_array_almost_equal(joints_0, joints_1)

    # test transformation of input
    joints_0 = results['input_2d']
    joints_1 = output['input']
    # manually do transformations
    center = np.array(
        [0.5 * results['image_width'], 0.5 * results['image_height']],
        dtype=np.float32)
    scale = np.array(0.5 * results['image_width'], dtype=np.float32)
    joints_0 = (joints_0 - center) / scale
    np.testing.assert_array_almost_equal(joints_0, joints_1)

    # test transformation of camera parameters
    camera_param_0 = results['camera_param']
    camera_param_1 = output['metas'].data['camera_param']
    # manually flip and normalization
    camera_param_0['c'][0] *= -1
    camera_param_0['p'][0] *= -1
    camera_param_0['c'] = (camera_param_0['c'] -
                           np.array(center)[:, None]) / scale
    camera_param_0['f'] = camera_param_0['f'] / scale
    np.testing.assert_array_almost_equal(camera_param_0['c'],
                                         camera_param_1['c'])
    np.testing.assert_array_almost_equal(camera_param_0['f'],
                                         camera_param_1['f'])

    # test CollectCameraIntrinsics
    intrinsics_0 = np.concatenate([
        results['camera_param']['f'].reshape(2),
        results['camera_param']['c'].reshape(2),
        results['camera_param']['k'].reshape(3),
        results['camera_param']['p'].reshape(2)
    ])
    intrinsics_1 = output['intrinsics']
    np.testing.assert_array_almost_equal(intrinsics_0, intrinsics_1)

    # test load mean/std from file
    with tempfile.TemporaryDirectory() as tmpdir:
        norm_param = {'mean': mean, 'std': std}
        norm_param_file = osp.join(tmpdir, 'norm_param.pkl')
        mmcv.dump(norm_param, norm_param_file)

        pipeline = [
            dict(
                type='NormalizeJointCoordinate',
                item='target',
                norm_param_file=norm_param_file),
        ]
        pipeline = Compose(pipeline)


def test_camera_projection():
    results = get_data_sample()
    pipeline_1 = [
        dict(
            type='CameraProjection',
            item='input_3d',
            output_name='input_3d_w',
            camera_type='SimpleCamera',
            mode='camera_to_world'),
        dict(
            type='CameraProjection',
            item='input_3d_w',
            output_name='input_3d_wp',
            camera_type='SimpleCamera',
            mode='world_to_pixel'),
        dict(
            type='CameraProjection',
            item='input_3d',
            output_name='input_3d_p',
            camera_type='SimpleCamera',
            mode='camera_to_pixel'),
        dict(type='Collect', keys=['input_3d_wp', 'input_3d_p'], meta_keys=[])
    ]
    camera_param = results['camera_param'].copy()
    camera_param['K'] = np.concatenate(
        (np.diagflat(camera_param['f']), camera_param['c']), axis=-1)
    pipeline_2 = [
        dict(
            type='CameraProjection',
            item='input_3d',
            output_name='input_3d_w',
            camera_type='SimpleCamera',
            camera_param=camera_param,
            mode='camera_to_world'),
        dict(
            type='CameraProjection',
            item='input_3d_w',
            output_name='input_3d_wp',
            camera_type='SimpleCamera',
            camera_param=camera_param,
            mode='world_to_pixel'),
        dict(
            type='CameraProjection',
            item='input_3d',
            output_name='input_3d_p',
            camera_type='SimpleCamera',
            camera_param=camera_param,
            mode='camera_to_pixel'),
        dict(
            type='CameraProjection',
            item='input_3d_w',
            output_name='input_3d_wc',
            camera_type='SimpleCamera',
            camera_param=camera_param,
            mode='world_to_camera'),
        dict(
            type='Collect',
            keys=['input_3d_wp', 'input_3d_p', 'input_2d'],
            meta_keys=[])
    ]

    output1 = Compose(pipeline_1)(results)
    output2 = Compose(pipeline_2)(results)

    np.testing.assert_allclose(
        output1['input_3d_wp'], output1['input_3d_p'], rtol=1e-6)

    np.testing.assert_allclose(
        output2['input_3d_wp'], output2['input_3d_p'], rtol=1e-6)

    np.testing.assert_allclose(
        output2['input_3d_p'], output2['input_2d'], rtol=1e-3, atol=1e-1)

    # test invalid camera parameters
    with pytest.raises(ValueError):
        # missing intrinsic parameters
        camera_param_wo_intrinsic = camera_param.copy()
        camera_param_wo_intrinsic.pop('K')
        camera_param_wo_intrinsic.pop('f')
        camera_param_wo_intrinsic.pop('c')
        _ = Compose([
            dict(
                type='CameraProjection',
                item='input_3d',
                camera_type='SimpleCamera',
                camera_param=camera_param_wo_intrinsic,
                mode='camera_to_pixel')
        ])

    with pytest.raises(ValueError):
        # invalid mode
        _ = Compose([
            dict(
                type='CameraProjection',
                item='input_3d',
                camera_type='SimpleCamera',
                camera_param=camera_param,
                mode='dummy')
        ])

    # test camera without undistortion
    camera_param_wo_undistortion = camera_param.copy()
    camera_param_wo_undistortion.pop('k')
    camera_param_wo_undistortion.pop('p')
    _ = Compose([
        dict(
            type='CameraProjection',
            item='input_3d',
            camera_type='SimpleCamera',
            camera_param=camera_param_wo_undistortion,
            mode='camera_to_pixel')
    ])

    # test pixel to camera transformation
    camera = SimpleCamera(camera_param_wo_undistortion)
    kpt_camera = np.random.rand(14, 3)
    kpt_pixel = camera.camera_to_pixel(kpt_camera)
    _kpt_camera = camera.pixel_to_camera(
        np.concatenate([kpt_pixel, kpt_camera[:, [2]]], -1))
    assert_array_almost_equal(_kpt_camera, kpt_camera, decimal=4)


def test_3d_heatmap_generation():
    ann_info = dict(
        image_size=np.array([256, 256]),
        heatmap_size=np.array([64, 64, 64]),
        heatmap3d_depth_bound=400.0,
        num_joints=17,
        joint_weights=np.ones((17, 1), dtype=np.float32),
        use_different_joint_weights=False)

    results = dict(
        joints_3d=np.zeros([17, 3]),
        joints_3d_visible=np.ones([17, 3]),
        ann_info=ann_info)

    pipeline = Compose([dict(type='Generate3DHeatmapTarget')])
    results_3d = pipeline(results)
    assert results_3d['target'].shape == (17, 64, 64, 64)
    assert results_3d['target_weight'].shape == (17, 1)

    # test joint_indices
    pipeline = Compose(
        [dict(type='Generate3DHeatmapTarget', joint_indices=[0, 8, 16])])
    results_3d = pipeline(results)
    assert results_3d['target'].shape == (3, 64, 64, 64)
    assert results_3d['target_weight'].shape == (3, 1)


def test_voxel3D_heatmap_generation():
    heatmap_size = [200, 160]
    cube_size = [8, 8, 2]
    ann_info = dict(
        image_size=np.array([800, 640]),
        heatmap_size=np.array([heatmap_size]),
        num_joints=17,
        num_scales=1,
        space_size=[12000.0, 12000.0, 2000.0],
        space_center=[3000.0, 4500.0, 1000.0],
        cube_size=cube_size)

    results = dict(
        joints_3d=np.ones([2, 17, 3]),
        joints_3d_visible=np.ones([2, 17, 3]),
        ann_info=ann_info)

    # test single joint index
    joint_indices = [[11, 12]]
    pipeline = Compose([
        dict(
            type='GenerateVoxel3DHeatmapTarget',
            sigma=200.0,
            joint_indices=joint_indices,
        ),
    ])
    results_ = pipeline(results)
    assert results_['targets_3d'].shape == (8, 8, 2)

    # test multiple joint indices
    joint_indices = [0, 8, 6]
    pipeline = Compose([
        dict(
            type='GenerateVoxel3DHeatmapTarget',
            sigma=200.0,
            joint_indices=joint_indices,
        ),
    ])
    results_ = pipeline(results)
    assert results_['targets_3d'].shape == (3, 8, 8, 2)


def test_input_heatmap_generation():
    heatmap_size = [200, 160]
    ann_info = dict(
        image_size=np.array([800, 640]),
        heatmap_size=np.array([heatmap_size]),
        num_joints=17,
        num_scales=1,
    )

    results = dict(
        joints=np.zeros([2, 17, 3]),
        joints_visible=np.ones([2, 17, 3]),
        ann_info=ann_info)

    pipeline = dict(
        type='GenerateInputHeatmaps',
        item='joints',
        visible_item='joints_visible',
        obscured=0.0,
        from_pred=False,
        sigma=3,
        scale=1.0,
        base_size=96,
        target_type='gaussian',
        heatmap_cfg=dict(
            base_scale=0.9,
            offset=0.03,
            threshold=0.6,
            extra=[
                dict(joint_ids=[7, 8], scale_factor=0.5, threshold=0.1),
                dict(
                    joint_ids=[9, 10],
                    scale_factor=0.2,
                    threshold=0.1,
                ),
                dict(
                    joint_ids=[0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],
                    scale_factor=0.5,
                    threshold=0.05)
            ]))

    pipelines = Compose([pipeline])
    results_ = pipelines(results)
    assert results_['input_heatmaps'][0].shape == (17, heatmap_size[1],
                                                   heatmap_size[0])

    # test `obscured`
    pipeline_copy = copy.deepcopy(pipeline)
    pipeline_copy['obscured'] = 0.5
    pipelines = Compose([pipeline])
    results_ = pipelines(results)
    assert results_['input_heatmaps'][0].shape == (17, heatmap_size[1],
                                                   heatmap_size[0])

    # test `heatmap_cfg`
    pipeline_copy = copy.deepcopy(pipeline)
    pipeline_copy['heatmap_cfg'] = None
    pipelines = Compose([pipeline])
    results_ = pipelines(results)
    assert results_['input_heatmaps'][0].shape == (17, heatmap_size[1],
                                                   heatmap_size[0])

    # test `from_pred`
    pipeline_copy = copy.deepcopy(pipeline)
    pipeline_copy['from_pred'] = True
    pipelines = Compose([pipeline])
    results_ = pipelines(results)
    assert results_['input_heatmaps'][0].shape == (17, heatmap_size[1],
                                                   heatmap_size[0])
    # test `from_pred` & `scale`
    pipeline_copy = copy.deepcopy(pipeline)
    pipeline_copy['from_pred'] = True
    pipeline_copy['scale'] = None
    pipelines = Compose([pipeline])
    results_ = pipelines(results)
    assert results_['input_heatmaps'][0].shape == (17, heatmap_size[1],
                                                   heatmap_size[0])


def test_affine_joints():
    ann_info = dict(image_size=np.array([800, 640]))

    results = dict(
        center=np.array([180, 144]),
        scale=np.array([360, 288], dtype=np.float32),
        rotation=0.0,
        joints=np.ones((3, 17, 2)),
        joints_visible=np.ones((3, 17, 2)),
        ann_info=ann_info)

    pipeline = Compose([
        dict(
            type='AffineJoints', item='joints', visible_item='joints_visible')
    ])
    results_ = pipeline(results)
    assert results_['joints'].shape == (3, 17, 2)
    assert results_['joints_visible'].shape == (3, 17, 2)

    # test `joints_visible` is zero
    results_copy = copy.deepcopy(results)
    results_copy['joints_visible'] = np.zeros((3, 17, 2))
    results_ = pipeline(results)
    assert results_['joints'].shape == (3, 17, 2)
    assert results_['joints_visible'].shape == (3, 17, 2)
