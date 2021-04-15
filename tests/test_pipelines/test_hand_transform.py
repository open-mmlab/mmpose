import copy
import os.path as osp

import json_tricks as json
import numpy as np
from xtcocotools.coco import COCO

from mmpose.datasets.pipelines import Compose


def get_mapping_id_name(imgs):
    """
    Args:
        imgs (dict): dict of image info.

    Returns:
        tuple: Image name & id mapping dicts.

        - id2name (dict): Mapping image id to name.
        - name2id (dict): Mapping image name to id.
    """
    id2name = {}
    name2id = {}
    for image_id, image in imgs.items():
        file_name = image['file_name']
        id2name[image_id] = file_name
        name2id[file_name] = image_id
    return id2name, name2id


def _cam2pixel(cam_coord, f, c):
    """Transform the joints from their camera coordinates to their pixel
    coordinates.

    Note:
        N: number of joints

    Args:
        cam_coord (ndarray[N, 3]): 3D joints coordinates
            in the camera coordinate system
        f (ndarray[2]): focal length of x and y axis
        c (ndarray[2]): principal point of x and y axis

    Returns:
        img_coord (ndarray[N, 3]): the coordinates (x, y, 0)
            in the image plane.
    """
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = np.zeros_like(x)
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def _world2cam(world_coord, R, T):
    """Transform the joints from their world coordinates to their camera
    coordinates.

    Note:
        N: number of joints

    Args:
        world_coord (ndarray[3, N]): 3D joints coordinates
            in the world coordinate system
        R (ndarray[3, 3]): camera rotation matrix
        T (ndarray[3, 1]): camera position (x, y, z)

    Returns:
        cam_coord (ndarray[3, N]): 3D joints coordinates
            in the camera coordinate system
    """
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def _encode_handtype(hand_type):
    if hand_type == 'right':
        return np.array([1, 0], dtype=int)
    elif hand_type == 'left':
        return np.array([0, 1], dtype=int)
    elif hand_type == 'interacting':
        return np.array([1, 1], dtype=int)
    else:
        assert 0, f'Not support hand type: {hand_type}'


def _xywh2cs(x, y, w, h, ann_info, padding=1.25):
    """This encodes bbox(x,y,w,w) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        center (np.ndarray[float32](2,)): center of the bbox (x, y).
        scale (np.ndarray[float32](2,)): scale of the bbox w & h.
    """
    aspect_ratio = ann_info['image_size'][0] / ann_info['image_size'][1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    # padding to include proper amount of context
    scale = scale * padding
    return center, scale


def get_sample_data():
    ann_info = {}
    ann_info['image_size'] = np.array([256, 256])
    ann_info['heatmap_size'] = np.array([64, 64, 64])
    ann_info['bbox_depth_size'] = 400.0
    ann_info['heatmap_size_root'] = 64
    ann_info['bbox_depth_size_root'] = 400.0
    ann_info['num_joints'] = 42
    ann_info['joint_weights'] = np.ones((ann_info['num_joints'], 1),
                                        dtype=np.float32)
    ann_info['use_different_joint_weights'] = False
    ann_info['flip_pairs'] = [[i, 21 + i] for i in range(21)]
    ann_info['inference_channel'] = list(range(42))
    ann_info['num_output_channels'] = 42
    ann_info['dataset_channel'] = list(range(42))

    ann_file = 'tests/data/interhand2.6m/test_interhand2.6m_data.json'
    camera_file = 'tests/data/interhand2.6m/test_interhand2.6m_camera.json'
    joint_file = 'tests/data/interhand2.6m/test_interhand2.6m_joint_3d.json'
    img_prefix = 'tests/data/interhand2.6m'

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    id2name, name2id = get_mapping_id_name(coco.imgs)

    with open(camera_file, 'r') as f:
        cameras = json.load(f)
    with open(joint_file, 'r') as f:
        joints = json.load(f)

    bbox_id = 0
    img_id = img_ids[0]
    num_joints = ann_info['num_joints']
    ann_id = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    ann = coco.loadAnns(ann_id)[0]
    img = coco.loadImgs(img_id)[0]

    capture_id = str(img['capture'])
    camera_name = img['camera']
    frame_idx = str(img['frame_idx'])
    image_file = osp.join(img_prefix, id2name[img_id])

    camera_pos = np.array(
        cameras[capture_id]['campos'][camera_name], dtype=np.float32)
    camera_rot = np.array(
        cameras[capture_id]['camrot'][camera_name], dtype=np.float32)
    focal = np.array(
        cameras[capture_id]['focal'][camera_name], dtype=np.float32)
    principal_pt = np.array(
        cameras[capture_id]['princpt'][camera_name], dtype=np.float32)
    joint_world = np.array(
        joints[capture_id][frame_idx]['world_coord'], dtype=np.float32)
    joint_cam = _world2cam(
        joint_world.transpose(1, 0), camera_rot,
        camera_pos.reshape(3, 1)).transpose(1, 0)
    joint_img = _cam2pixel(joint_cam, focal, principal_pt)[:, :2]

    joint_valid = np.array(ann['joint_valid'], dtype=np.float32).flatten()
    hand_type = _encode_handtype(ann['hand_type'])
    hand_type_valid = ann['hand_type_valid']

    bbox = np.array(ann['bbox'], dtype=np.float32)
    # extend the bbox to include some context
    center, scale = _xywh2cs(*bbox, ann_info, 1.25)
    abs_depth = [joint_cam[20, 2], joint_cam[41, 2]]

    rel_root_depth = joint_cam[41, 2] - joint_cam[20, 2]
    # if root is not valid, root-relative 3D depth is also invalid.
    rel_root_valid = joint_valid[20] * joint_valid[41]

    # if root is not valid -> root-relative 3D pose is also not valid.
    # Therefore, mark all joints as invalid
    joint_valid[:20] *= joint_valid[20]
    joint_valid[21:] *= joint_valid[41]

    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
    joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
    joints_3d[:, :2] = joint_img
    joints_3d[:21, 2] = joint_cam[:21, 2] - joint_cam[20, 2]
    joints_3d[21:, 2] = joint_cam[21:, 2] - joint_cam[41, 2]
    joints_3d_visible[...] = np.minimum(1, joint_valid.reshape(-1, 1))

    results = {
        'image_file': image_file,
        'center': center,
        'scale': scale,
        'rotation': 0,
        'joints_3d': joints_3d,
        'joints_3d_visible': joints_3d_visible,
        'hand_type': hand_type,
        'hand_type_valid': hand_type_valid,
        'rel_root_depth': rel_root_depth,
        'rel_root_valid': rel_root_valid,
        'abs_depth': abs_depth,
        'joints_cam': joint_cam,
        'focal': focal,
        'princpt': principal_pt,
        'dataset': 'interhand3d',
        'bbox': bbox,
        'bbox_score': 1,
        'bbox_id': bbox_id,
        'ann_info': ann_info
    }

    return results


def _check_flip(origin_imgs, result_imgs):
    """Check if the origin_imgs are flipped correctly."""
    h, w, c = origin_imgs.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if result_imgs[i, j, k] != origin_imgs[i, w - 1 - j, k]:
                    return False
    return True


def test_hand_transforms():
    results = get_sample_data()

    # load image
    pipeline = Compose([dict(type='LoadImageFromFile')])
    results = pipeline(results)

    # test random flip
    pipeline = Compose([dict(type='HandRandomFlip', flip_prob=1)])
    results_flip = pipeline(copy.deepcopy(results))
    assert _check_flip(results['img'], results_flip['img'])

    # test random translation
    pipeline = Compose(
        [dict(type='HandGetRandomTranslation', trans_factor=0.1)])
    results_trans = pipeline(copy.deepcopy(results))
    assert results_trans['center'].shape == (2, )

    # test 3D heatmap target generation
    pipeline = Compose([dict(type='HandGenerate3DHeatmapTarget', sigma=2)])
    results_3d = pipeline(copy.deepcopy(results))
    assert results_3d['target'].shape == (42, 64, 64, 64)
    assert results_3d['target_weight'].shape == (42, 1)

    # test root depth target generation
    pipeline = Compose([dict(type='HandGenerateDepthTarget')])
    results_depth = pipeline(copy.deepcopy(results))
    assert results_depth['target'].shape == (1, )
    assert results_depth['target_weight'].shape == (1, )

    # test hand type label target generation
    pipeline = Compose([dict(type='HandGenerateLabelTarget')])
    results_label = pipeline(copy.deepcopy(results))
    assert results_label['target'].shape == (2, )
    assert results_label['target_weight'].shape == (2, )

    # test multitask taget gather
    pipeline_list = [[dict(type='HandGenerate3DHeatmapTarget', sigma=2)],
                     [dict(type='HandGenerateDepthTarget')],
                     [dict(type='HandGenerateLabelTarget')]]
    pipeline = Compose(
        [dict(type='MultitaskGatherTarget', pipeline_list=pipeline_list)])
    target_multitask = pipeline(copy.deepcopy(results))
    target = target_multitask['target']
    target_weight = target_multitask['target_weight']
    assert isinstance(target, list)
    assert isinstance(target_weight, list)
    assert target[0].shape == (42, 64, 64, 64)
    assert target_weight[0].shape == (42, 1)
    assert target[1].shape == (1, )
    assert target_weight[1].shape == (1, )
    assert target[2].shape == (2, )
    assert target_weight[2].shape == (2, )
