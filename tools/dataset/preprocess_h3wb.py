import argparse
import os
import pickle
import numpy as np
from os.path import join
import json


# coco-wholebody:
# 1 ~ 17: body
# 18 ~ 23: feets
# 24 ~ 91: face
# 92 ~ 133: hands


# train_subjects = ['2Dto3D_train_part1.json', '2Dto3D_train_part2.json', '2Dto3D_train_part3.json', '2Dto3D_train_part4.json']
# test_subjects = ['2Dto3D_test_2d.json']
train_subjects = ['2Dto3D_train_part1.json', '2Dto3D_train_part2.json', '2Dto3D_train_part3.json']
test_subjects = ['2Dto3D_train_part4.json']
train_img_size = (1002, 1002)


# todo
root_index = 12
num_joints = 133


def get_pose_stats(kps):
    """Get statistic information `mean` and `std` of pose data.

    Args:
        kps (ndarray): keypoints in shape [..., K, C] where K and C is
            the keypoint category number and dimension.
    Returns:
        mean (ndarray): [K, C]
    """
    assert kps.ndim > 2
    K, C = kps.shape[-2:]
    kps = kps.reshape(-1, K, C)
    mean = kps.mean(axis=0)
    std = kps.std(axis=0)
    return mean, std


def get_annotations(joints_2d, joints_3d, scale_factor=1.2):
    """Get annotations, including centers, scales, joints_2d and joints_3d.

    Args:
        joints_2d: 2D joint coordinates in shape [N, K, 2], where N is the
            frame number, K is the joint number.
        joints_3d: 3D joint coordinates in shape [N, K, 3], where N is the
            frame number, K is the joint number.
        scale_factor: Scale factor of bounding box. Default: 1.2.
    Returns:
        centers (ndarray): [N, 2]
        scales (ndarray): [N,]
        joints_2d (ndarray): [N, K, 3]
        joints_3d (ndarray): [N, K, 4]
    """
    # calculate joint visibility
    visibility = (joints_2d[:, :, 0] >= 0) * \
                 (joints_2d[:, :, 0] < train_img_size[0]) * \
                 (joints_2d[:, :, 1] >= 0) * \
                 (joints_2d[:, :, 1] < train_img_size[1])
    visibility = np.array(visibility, dtype=np.float32)[:, :, None]
    joints_2d = np.concatenate([joints_2d, visibility], axis=-1)
    joints_3d = np.concatenate([joints_3d, visibility], axis=-1)

    # calculate bounding boxes
    bboxes = np.stack([
        np.min(joints_2d[:, :, 0], axis=1),
        np.min(joints_2d[:, :, 1], axis=1),
        np.max(joints_2d[:, :, 0], axis=1),
        np.max(joints_2d[:, :, 1], axis=1)
    ],
                      axis=1)
    centers = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2,
                        (bboxes[:, 1] + bboxes[:, 3]) / 2],
                       axis=1)
    scales = scale_factor * np.max(bboxes[:, 2:] - bboxes[:, :2], axis=1) / 200

    return centers, scales, joints_2d, joints_3d


def load_trainset(data_root, out_dir):
    _imgnames = []
    _centers = []
    _scales = []
    _joints_2d = []
    _joints_3d = []
    # cameras = {}
    
    annot_dir = join(out_dir, 'annotations')
    os.makedirs(annot_dir, exist_ok=True)
    
    for subj in train_subjects:
        file_path = join(data_root, subj)
        with open(file_path, 'rb') as fh:
            annos = json.load(fh)
        
        kpts_2d_list = []
        kpts_3d_list = []
        
        for _, kpts in annos.items():
            kpts_2d_per_img = []
            kpts_3d_per_img = []
            for idx in range(num_joints):
                kpts_2d_dict = kpts['keypoints_2d']
                kpts_3d_dict = kpts['keypoints_3d']
                kpts_2d_per_img.append([kpts_2d_dict[str(idx)]['x'], kpts_2d_dict[str(idx)]['y']])
                kpts_3d_per_img.append([kpts_3d_dict[str(idx)]['x'], kpts_3d_dict[str(idx)]['y'], kpts_3d_dict[str(idx)]['z']])
            kpts_2d = np.array(kpts_2d_per_img)
            kpts_3d = np.array(kpts_3d_per_img)
            
            kpts_2d_list.append(kpts_2d[np.newaxis, ...])
            kpts_3d_list.append(kpts_3d[np.newaxis, ...])
        
        joints_2d = np.concatenate(kpts_2d_list)
        joints_3d = np.concatenate(kpts_3d_list)
        joints_3d = joints_3d * 0.001
        centers, scales, joints_2d, joints_3d = get_annotations(
                    joints_2d, joints_3d)
        _centers.append(centers)
        _scales.append(scales)
        _joints_2d.append(joints_2d)
        _joints_3d.append(joints_3d)

    # _imgnames = np.array(_imgnames)
    _centers = np.concatenate(_centers)
    _scales = np.concatenate(_scales)
    _joints_2d = np.concatenate(_joints_2d)
    _joints_3d = np.concatenate(_joints_3d)
    _imgnames = ['none.jpg'] * len(_joints_2d)
    _imgnames = np.array(_imgnames)
    
    out_file = join(annot_dir, 'h3wb_train.npz')
    np.savez(
        out_file,
        imgname=_imgnames,
        center=_centers,
        scale=_scales,
        part=_joints_2d,
        S=_joints_3d)
    
    print(f'Create annotation file for trainset: {out_file}. '
          f'{len(_imgnames)} samples in total.')
    
    # get `mean` and `std` of pose data
    _joints_3d = _joints_3d[..., :3]
    mean_3d, std_3d = get_pose_stats(_joints_3d)
    
    _joints_2d = _joints_2d[..., :2]
    mean_2d, std_2d = get_pose_stats(_joints_2d)
    
    # centered around root
    root_mask = np.array(list(range(num_joints))) == root_index
    
    _joints_3d_rel = _joints_3d - _joints_3d[..., root_index: root_index + 1, :]
    _joints_3d_rel = _joints_3d_rel[..., ~root_mask, :]
    mean_3d_rel, std_3d_rel = get_pose_stats(_joints_3d_rel)
    # mean_3d_rel[root_index] = mean_3d[root_index]
    # std_3d_rel[root_index] = std_3d[root_index]
    
    _joints_2d_rel = _joints_2d - _joints_2d[..., root_index:root_index + 1, :]
    _joints_2d_rel = _joints_2d_rel[..., ~root_mask, :]
    mean_2d_rel, std_2d_rel = get_pose_stats(_joints_2d_rel)
    # mean_2d_rel[root_index] = mean_2d[root_index]
    # std_2d_rel[root_index] = std_2d[root_index]
    
    print(mean_3d_rel.shape, std_3d_rel.shape)
    print(mean_2d_rel.shape, std_2d_rel.shape)
    
    stats = {
        'joint3d_stats': {
            'mean': mean_3d,
            'std': std_3d,
        },
        'joint2d_stats': {
            'mean': mean_2d,
            'std': std_2d,
        },
        'joint3d_rel_stats': {
            'mean': mean_3d_rel,
            'std': std_3d_rel
        },
        'joint2d_rel_stats': {
            'mean': mean_2d_rel,
            'std': std_2d_rel
        }
    }
    
    for name, stat_dict in stats.items():
        out_file = join(annot_dir, f'{name}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump(stat_dict, f)
        print(f'Create statistic data file: {out_file}')
    

def load_testset(data_root, out_dir, valid_only):
    _imgnames = []
    _centers = []
    _scales = []
    _joints_2d = []
    _joints_3d = []
    cameras = {}
    
    annot_dir = join(out_dir, 'annotations')
    os.makedirs(annot_dir, exist_ok=True)
    
    for subj in test_subjects:
        subj_path = join(data_root, subj)
        with open(subj_path, 'rb') as fh:
            annos = json.load(fh)
        
        kpts_2d_list = []
        kpts_3d_list = []
        
        for _, kpts in annos.items():
            kpts_2d_per_img = []
            kpts_3d_per_img = []
            for idx in range(num_joints):
                kpts_2d_dict = kpts['keypoints_2d']
                kpts_3d_dict = kpts['keypoints_3d']
                kpts_2d_per_img.append([kpts_2d_dict[str(idx)]['x'], kpts_2d_dict[str(idx)]['y']])
                kpts_3d_per_img.append([kpts_3d_dict[str(idx)]['x'], kpts_3d_dict[str(idx)]['y'], kpts_3d_dict[str(idx)]['z']])
            
            kpts_2d = np.array(kpts_2d_per_img)
            kpts_3d = np.array(kpts_3d_per_img)
            
            kpts_2d_list.append(kpts_2d[np.newaxis, ...])
            kpts_3d_list.append(kpts_3d[np.newaxis, ...])
        
        joints_2d = np.concatenate(kpts_2d_list)
        joints_3d = np.concatenate(kpts_3d_list)
        joints_3d = joints_3d * 0.001
        centers, scales, joints_2d, joints_3d = get_annotations(
                    joints_2d, joints_3d)
        _centers.append(centers)
        _scales.append(scales)
        _joints_2d.append(joints_2d)
        _joints_3d.append(joints_3d)
        
    # _imgnames = np.array(_imgnames)
    _centers = np.concatenate(_centers)
    _scales = np.concatenate(_scales)
    _joints_2d = np.concatenate(_joints_2d)
    _joints_3d = np.concatenate(_joints_3d)
    _imgnames = ['none.jpg'] * len(_joints_2d)
    _imgnames = np.array(_imgnames)
    
    if valid_only:
        out_file = join(annot_dir, 'h3wb_test_valid.npz')
    else:
        out_file = join(annot_dir, 'h3wb_test_all.npz')
        
    np.savez(
        out_file,
        imgname=_imgnames,
        center=_centers,
        scale=_scales,
        part=_joints_2d,
        S=_joints_3d)
    print(f'Create annotation file for testset: {out_file}. '
          f'{len(_imgnames)} samples in total.')
    
    out_file = join(annot_dir, 'cameras_test.pkl')
    with open(out_file, 'wb') as fout:
        pickle.dump(cameras, fout)
    print(f'Create camera file for testset: {out_file}.')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', 
                        default='/home/jiwei/workspace/h3wb/',
                        type=str, help='data root')
    parser.add_argument('--out_dir', 
                        default='/home/jiwei/workspace/h3wb_processed/',
                        type=str, help='directory to save annotation files.')
    args = parser.parse_args()
    data_root = args.data_root
    out_dir = args.out_dir
    
    load_trainset(data_root, out_dir)
    load_testset(data_root, out_dir, valid_only=True)