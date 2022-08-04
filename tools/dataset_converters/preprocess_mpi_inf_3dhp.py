# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import pickle
import shutil
from os.path import join

import cv2
import h5py
import mmcv
import numpy as np
from scipy.io import loadmat

train_subjects = [i for i in range(1, 9)]
test_subjects = [i for i in range(1, 7)]
train_seqs = [1, 2]
train_cams = [0, 1, 2, 4, 5, 6, 7, 8]
train_frame_nums = {
    (1, 1): 6416,
    (1, 2): 12430,
    (2, 1): 6502,
    (2, 2): 6081,
    (3, 1): 12488,
    (3, 2): 12283,
    (4, 1): 6171,
    (4, 2): 6675,
    (5, 1): 12820,
    (5, 2): 12312,
    (6, 1): 6188,
    (6, 2): 6145,
    (7, 1): 6239,
    (7, 2): 6320,
    (8, 1): 6468,
    (8, 2): 6054
}
test_frame_nums = {1: 6151, 2: 6080, 3: 5838, 4: 6007, 5: 320, 6: 492}
train_img_size = (2048, 2048)
root_index = 14
joints_17 = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]


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
    """Load training data, create annotation file and camera file.
    Args:
        data_root: Directory of dataset, which is organized in the following
            hierarchy:
                data_root
                |-- train
                    |-- S1
                        |-- Seq1
                        |-- Seq2
                    |-- S2
                    |-- ...
                |-- test
                    |-- TS1
                    |-- TS2
                    |-- ...
        out_dir: Directory to save annotation file.
    """
    _imgnames = []
    _centers = []
    _scales = []
    _joints_2d = []
    _joints_3d = []
    cameras = {}

    img_dir = join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    annot_dir = join(out_dir, 'annotations')
    os.makedirs(annot_dir, exist_ok=True)

    for subj in train_subjects:
        for seq in train_seqs:
            seq_path = join(data_root, 'train', f'S{subj}', f'Seq{seq}')
            num_frames = train_frame_nums[(subj, seq)]

            # load camera parametres
            camera_file = join(seq_path, 'camera.calibration')
            with open(camera_file, 'r') as fin:
                lines = fin.readlines()
                for cam in train_cams:
                    K = [float(s) for s in lines[cam * 7 + 5][11:-2].split()]
                    f = np.array([[K[0]], [K[5]]])
                    c = np.array([[K[2]], [K[6]]])
                    RT = np.array(
                        [float(s) for s in lines[cam * 7 + 6][11:-2].split()])
                    RT = np.reshape(RT, (4, 4))
                    R = RT[:3, :3]
                    # convert unit from millimeter to meter
                    T = RT[:3, 3:] * 0.001
                    size = [int(s) for s in lines[cam * 7 + 3][14:].split()]
                    w, h = size
                    cam_param = dict(
                        R=R, T=T, c=c, f=f, w=w, h=h, name=f'train_cam_{cam}')
                    cameras[f'S{subj}_Seq{seq}_Cam{cam}'] = cam_param

            # load annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = loadmat(annot_file)['annot2']
            annot3 = loadmat(annot_file)['annot3']
            for cam in train_cams:
                # load 2D and 3D annotations
                joints_2d = np.reshape(annot2[cam][0][:num_frames],
                                       (num_frames, 28, 2))[:, joints_17]
                joints_3d = np.reshape(annot3[cam][0][:num_frames],
                                       (num_frames, 28, 3))[:, joints_17]
                joints_3d = joints_3d * 0.001
                centers, scales, joints_2d, joints_3d = get_annotations(
                    joints_2d, joints_3d)
                _centers.append(centers)
                _scales.append(scales)
                _joints_2d.append(joints_2d)
                _joints_3d.append(joints_3d)

                # extract frames from video
                video_path = join(seq_path, 'imageSequence',
                                  f'video_{cam}.avi')
                video = mmcv.VideoReader(video_path)
                for i in mmcv.track_iter_progress(range(num_frames)):
                    img = video.read()
                    if img is None:
                        break
                    imgname = f'S{subj}_Seq{seq}_Cam{cam}_{i+1:06d}.jpg'
                    _imgnames.append(imgname)
                    cv2.imwrite(join(img_dir, imgname), img)

    _imgnames = np.array(_imgnames)
    _centers = np.concatenate(_centers)
    _scales = np.concatenate(_scales)
    _joints_2d = np.concatenate(_joints_2d)
    _joints_3d = np.concatenate(_joints_3d)

    out_file = join(annot_dir, 'mpi_inf_3dhp_train.npz')
    np.savez(
        out_file,
        imgname=_imgnames,
        center=_centers,
        scale=_scales,
        part=_joints_2d,
        S=_joints_3d)
    print(f'Create annotation file for trainset: {out_file}. '
          f'{len(_imgnames)} samples in total.')

    out_file = join(annot_dir, 'cameras_train.pkl')
    with open(out_file, 'wb') as fout:
        pickle.dump(cameras, fout)
    print(f'Create camera file for trainset: {out_file}.')

    # get `mean` and `std` of pose data
    _joints_3d = _joints_3d[..., :3]  # remove visibility
    mean_3d, std_3d = get_pose_stats(_joints_3d)

    _joints_2d = _joints_2d[..., :2]  # remove visibility
    mean_2d, std_2d = get_pose_stats(_joints_2d)

    # centered around root
    _joints_3d_rel = _joints_3d - _joints_3d[..., root_index:root_index + 1, :]
    mean_3d_rel, std_3d_rel = get_pose_stats(_joints_3d_rel)
    mean_3d_rel[root_index] = mean_3d[root_index]
    std_3d_rel[root_index] = std_3d[root_index]

    _joints_2d_rel = _joints_2d - _joints_2d[..., root_index:root_index + 1, :]
    mean_2d_rel, std_2d_rel = get_pose_stats(_joints_2d_rel)
    mean_2d_rel[root_index] = mean_2d[root_index]
    std_2d_rel[root_index] = std_2d[root_index]

    stats = {
        'joint3d_stats': {
            'mean': mean_3d,
            'std': std_3d
        },
        'joint2d_stats': {
            'mean': mean_2d,
            'std': std_2d
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


def load_testset(data_root, out_dir, valid_only=True):
    """Load testing data, create annotation file and camera file.

    Args:
        data_root: Directory of dataset.
        out_dir: Directory to save annotation file.
        valid_only: Only keep frames with valid_label == 1.
    """
    _imgnames = []
    _centers = []
    _scales = []
    _joints_2d = []
    _joints_3d = []
    cameras = {}

    img_dir = join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    annot_dir = join(out_dir, 'annotations')
    os.makedirs(annot_dir, exist_ok=True)

    for subj in test_subjects:
        subj_path = join(data_root, 'test', f'TS{subj}')
        num_frames = test_frame_nums[subj]

        # load annotations
        annot_file = os.path.join(subj_path, 'annot_data.mat')
        with h5py.File(annot_file, 'r') as fin:
            annot2 = np.array(fin['annot2']).reshape((-1, 17, 2))
            annot3 = np.array(fin['annot3']).reshape((-1, 17, 3))
            valid = np.array(fin['valid_frame']).reshape(-1)

        # manually estimate camera intrinsics
        fx, cx = np.linalg.lstsq(
            annot3[:, :, [0, 2]].reshape((-1, 2)),
            (annot2[:, :, 0] * annot3[:, :, 2]).reshape(-1, 1),
            rcond=None)[0].flatten()
        fy, cy = np.linalg.lstsq(
            annot3[:, :, [1, 2]].reshape((-1, 2)),
            (annot2[:, :, 1] * annot3[:, :, 2]).reshape(-1, 1),
            rcond=None)[0].flatten()
        if subj <= 4:
            w, h = 2048, 2048
        else:
            w, h = 1920, 1080
        cameras[f'TS{subj}'] = dict(
            c=np.array([[cx], [cy]]),
            f=np.array([[fx], [fy]]),
            w=w,
            h=h,
            name=f'test_cam_{subj}')

        # get annotations
        if valid_only:
            valid_frames = np.nonzero(valid)[0]
        else:
            valid_frames = np.arange(num_frames)
        joints_2d = annot2[valid_frames, :, :]
        joints_3d = annot3[valid_frames, :, :] * 0.001

        centers, scales, joints_2d, joints_3d = get_annotations(
            joints_2d, joints_3d)
        _centers.append(centers)
        _scales.append(scales)
        _joints_2d.append(joints_2d)
        _joints_3d.append(joints_3d)

        # copy and rename images
        for i in valid_frames:
            imgname = f'TS{subj}_{i+1:06d}.jpg'
            shutil.copyfile(
                join(subj_path, 'imageSequence', f'img_{i+1:06d}.jpg'),
                join(img_dir, imgname))
            _imgnames.append(imgname)

    _imgnames = np.array(_imgnames)
    _centers = np.concatenate(_centers)
    _scales = np.concatenate(_scales)
    _joints_2d = np.concatenate(_joints_2d)
    _joints_3d = np.concatenate(_joints_3d)

    if valid_only:
        out_file = join(annot_dir, 'mpi_inf_3dhp_test_valid.npz')
    else:
        out_file = join(annot_dir, 'mpi_inf_3dhp_test_all.npz')
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
    parser.add_argument('data_root', type=str, help='data root')
    parser.add_argument(
        'out_dir', type=str, help='directory to save annotation files.')
    args = parser.parse_args()
    data_root = args.data_root
    out_dir = args.out_dir

    load_trainset(data_root, out_dir)
    load_testset(data_root, out_dir, valid_only=True)
