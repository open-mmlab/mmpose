# -----------------------------------------------------------------------------
# Adapted from https://github.com/anibali/h36m-fetch
# Original license: Copyright (c) Aiden Nibali, under the Apache License.
# -----------------------------------------------------------------------------

import argparse
import os
import pickle
import tarfile
import xml.etree.ElementTree as ET
from os.path import join

import cv2
import numpy as np
from spacepy import pycdf


class PreprocessH36m:
    """Preprocess Human3.6M dataset.

    Args:
        metadata (str): Path to metadata.xml.
        original_dir (str): Directory of the original dataset with all files
            compressed. Specifically, .tgz files belonging to subject 1
            should be placed under the subdirectory 's1'.
        extracted_dir (str): Directory of the extracted files. If not given, it
            will be placed under the same parent directory as original_dir.
        processed_der (str): Directory of the processed files. If not given, it
            will be placed under the same parent directory as original_dir.
        sample_rate (int): Downsample FPS to `1 / sample_rate`. Default: 5.
    """

    def __init__(self,
                 metadata,
                 original_dir,
                 extracted_dir=None,
                 processed_dir=None,
                 sample_rate=5):
        self.metadata = metadata
        self.original_dir = original_dir
        self.sample_rate = sample_rate

        if extracted_dir is None:
            self.extracted_dir = join(
                os.path.dirname(os.path.abspath(self.original_dir)),
                'extracted')
        else:
            self.extracted_dir = extracted_dir

        if processed_dir is None:
            self.processed_dir = join(
                os.path.dirname(os.path.abspath(self.original_dir)),
                'processed')
        else:
            self.processed_dir = processed_dir

        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.camera_ids = []
        self._load_metadata()

        self.subjects_annot = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        self.subjects_splits = {
            'train': ['S1', 'S5', 'S6', 'S7', 'S8'],
            'test': ['S9', 'S11']
        }
        self.extract_files = ['Videos', 'D2_Positions', 'D3_Positions_mono']
        self.movable_joints = [
            0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27
        ]
        self.scale_factor = 1.2
        self.image_sizes = {
            '54138969': {
                'width': 1000,
                'height': 1002
            },
            '55011271': {
                'width': 1000,
                'height': 1000
            },
            '58860488': {
                'width': 1000,
                'height': 1000
            },
            '60457274': {
                'width': 1000,
                'height': 1002
            }
        }

    def extract_tgz(self):
        """Extract files from self.extrct_files."""
        os.makedirs(self.extracted_dir, exist_ok=True)
        for subject in self.subjects_annot:
            cur_dir = join(self.original_dir, subject.lower())
            for file in self.extract_files:
                filename = join(cur_dir, file + '.tgz')
                print(f'Extracting {filename} ...')
                with tarfile.open(filename) as tar:
                    tar.extractall(self.extracted_dir)
        print('Extraction done.\n')

    def generate_cameras_file(self):
        """Generate cameras.pkl which contains camera parameters for 11
        subjects each with 4 cameras."""
        cameras = {}
        for subject in range(1, 12):
            for camera in range(4):
                key = (f'S{subject}', self.camera_ids[camera])
                cameras[key] = self._get_camera_params(camera, subject)

        out_file = join(self.processed_dir, 'annotation_body3d', 'cameras.pkl')
        with open(out_file, 'wb') as fout:
            pickle.dump(cameras, fout)
        print(f'Camera parameters have been written to "{out_file}".\n')

    def generate_annotations(self):
        """Generate annotations for training and testing data."""
        output_dir = join(self.processed_dir, 'annotation_body3d',
                          f'fps{50 // self.sample_rate}')
        os.makedirs(output_dir, exist_ok=True)

        for data_split in ('train', 'test'):
            imgnames_all = []
            centers_all = []
            scales_all = []
            kps2d_all = []
            kps3d_all = []
            for subject in self.subjects_splits[data_split]:
                for action, subaction in self.sequence_mappings[subject].keys(
                ):
                    if action == '1':
                        # exclude action "_ALL"
                        continue
                    for camera in self.camera_ids:
                        imgnames, centers, scales, kps2d, kps3d\
                         = self._load_annotations(
                            subject, action, subaction, camera)
                        imgnames_all.append(imgnames)
                        centers_all.append(centers)
                        scales_all.append(scales)
                        kps2d_all.append(kps2d)
                        kps3d_all.append(kps3d)

            imgnames_all = np.concatenate(imgnames_all)
            centers_all = np.concatenate(centers_all)
            scales_all = np.concatenate(scales_all)
            kps2d_all = np.concatenate(kps2d_all)
            kps3d_all = np.concatenate(kps3d_all)

            out_file = join(output_dir, f'h36m_{data_split}.npz')
            np.savez(
                out_file,
                imgname=imgnames_all,
                center=centers_all,
                scale=scales_all,
                part=kps2d_all,
                S=kps3d_all)

            print(
                f'All annotations of {data_split}ing data have been written to'
                f' "{out_file}". {len(imgnames_all)} samples in total.\n')

            if data_split == 'train':
                kps_3d_all = kps3d_all[..., :3]  # remove visibility
                mean_3d, std_3d = self._get_pose_stats(kps_3d_all)

                kps_2d_all = kps2d_all[..., :2]  # remove visibility
                mean_2d, std_2d = self._get_pose_stats(kps_2d_all)

                # centered around root
                # the root keypoint is 0-index
                kps_3d_rel = kps_3d_all[..., 1:, :] - kps_3d_all[..., :1, :]
                mean_3d_rel, std_3d_rel = self._get_pose_stats(kps_3d_rel)

                kps_2d_rel = kps_2d_all[..., 1:, :] - kps_2d_all[..., :1, :]
                mean_2d_rel, std_2d_rel = self._get_pose_stats(kps_2d_rel)

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
                    out_file = join(output_dir, f'{name}.pkl')
                    with open(out_file, 'wb') as f:
                        pickle.dump(stat_dict, f)
                    print(f'Create statistic data file: {out_file}')

    @staticmethod
    def _get_pose_stats(kps):
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

    def _load_metadata(self):
        """Load meta data from metadata.xml."""

        assert os.path.exists(self.metadata)

        tree = ET.parse(self.metadata)
        root = tree.getroot()

        for i, tr in enumerate(root.find('mapping')):
            if i == 0:
                _, _, *self.subjects = [td.text for td in tr]
                self.sequence_mappings \
                    = {subject: {} for subject in self.subjects}
            elif i < 33:
                action_id, subaction_id, *prefixes = [td.text for td in tr]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id, subaction_id)]\
                        = prefix

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        self.camera_ids \
            = [elem.text for elem in root.find('dbcameras/index2id')]

        w0 = root.find('w0')
        self.cameras_raw = [float(num) for num in w0.text[1:-1].split()]

    def _get_base_filename(self, subject, action, subaction, camera):
        """Get base filename given subject, action, subaction and camera."""
        return f'{self.sequence_mappings[subject][(action, subaction)]}' + \
            f'.{camera}'

    def _get_camera_params(self, camera, subject):
        """Get camera parameters given camera id and subject id."""
        metadata_slice = np.zeros(15)
        start = 6 * (camera * 11 + (subject - 1))

        metadata_slice[:6] = self.cameras_raw[start:start + 6]
        metadata_slice[6:] = self.cameras_raw[265 + camera * 9 - 1:265 +
                                              (camera + 1) * 9 - 1]

        # extrinsics
        x, y, z = -metadata_slice[0], metadata_slice[1], -metadata_slice[2]

        R_x = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)],
                        [0, -np.sin(x), np.cos(x)]])
        R_y = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                        [-np.sin(y), 0, np.cos(y)]])
        R_z = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z),
                                                    np.cos(z), 0], [0, 0, 1]])
        R = (R_x @ R_y @ R_z).T
        T = metadata_slice[3:6].reshape(-1, 1)
        # convert unit from millimeter to meter
        T *= 0.001

        # intrinsics
        c = metadata_slice[8:10, None]
        f = metadata_slice[6:8, None]

        # distortion
        k = metadata_slice[10:13, None]
        p = metadata_slice[13:15, None]

        return {
            'R': R,
            'T': T,
            'c': c,
            'f': f,
            'k': k,
            'p': p,
            'w': self.image_sizes[self.camera_ids[camera]]['width'],
            'h': self.image_sizes[self.camera_ids[camera]]['height'],
            'name': f'camera{camera + 1}',
            'id': self.camera_ids[camera]
        }

    def _load_annotations(self, subject, action, subaction, camera):
        """Load annotations for a sequence."""
        subj_dir = join(self.extracted_dir, subject)
        basename = self._get_base_filename(subject, action, subaction, camera)

        # load 2D keypoints
        with pycdf.CDF(
                join(subj_dir, 'MyPoseFeatures', 'D2_Positions',
                     basename + '.cdf')) as cdf:
            kps_2d = np.array(cdf['Pose'])

        num_frames = kps_2d.shape[1]
        kps_2d = kps_2d.reshape((num_frames, 32, 2))[::self.sample_rate,
                                                     self.movable_joints]
        kps_2d = np.concatenate([kps_2d, np.ones((len(kps_2d), 17, 1))],
                                axis=2)

        # load 3D keypoints
        with pycdf.CDF(
                join(subj_dir, 'MyPoseFeatures', 'D3_Positions_mono',
                     basename + '.cdf')) as cdf:
            kps_3d = np.array(cdf['Pose'])

        kps_3d = kps_3d.reshape(
            (num_frames, 32, 3))[::self.sample_rate,
                                 self.movable_joints] / 1000.
        kps_3d = np.concatenate([kps_3d, np.ones((len(kps_3d), 17, 1))],
                                axis=2)

        # calculate bounding boxes
        bboxes = np.stack([
            np.min(kps_2d[:, :, 0], axis=1),
            np.min(kps_2d[:, :, 1], axis=1),
            np.max(kps_2d[:, :, 0], axis=1),
            np.max(kps_2d[:, :, 1], axis=1)
        ],
                          axis=1)
        centers = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2,
                            (bboxes[:, 1] + bboxes[:, 3]) / 2],
                           axis=1)
        scales = self.scale_factor * np.max(
            bboxes[:, 2:] - bboxes[:, :2], axis=1) / 200

        # extract frames and save imgnames
        imgnames = []
        video_path = join(subj_dir, 'Videos', basename + '.mp4')
        sub_base = subject + '_' + basename.replace(' ', '_')
        img_dir = join(self.processed_dir, 'images', subject, sub_base)
        os.makedirs(img_dir, exist_ok=True)
        prefix = join(subject, sub_base, sub_base)

        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            success, img = cap.read()
            if not success:
                break
            if i % self.sample_rate == 0:
                imgname = f'{prefix}_{i + 1:06d}.jpg'
                imgnames.append(imgname)
                dest_path = join(self.processed_dir, 'images', imgname)
                if not os.path.exists(dest_path):
                    cv2.imwrite(dest_path, img)
                if len(imgnames) == len(centers):
                    break
            i += 1
        cap.release()
        imgnames = np.array(imgnames)

        print(f'Annoatations for sequence "{subject} {basename}" are loaded. '
              f'{len(imgnames)} samples in total.')

        return imgnames, centers, scales, kps_2d, kps_3d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--metadata', type=str, required=True, help='Path to metadata.xml')
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Directory of the original dataset with all files compressed. '
        'Specifically, .tgz files belonging to subject 1 should be placed '
        'under the subdirectory \"s1\".')
    parser.add_argument(
        '--extracted',
        type=str,
        default=None,
        help='Directory of the extracted files. If not given, it will be '
        'placed under the same parent directory as original_dir.')
    parser.add_argument(
        '--processed',
        type=str,
        default=None,
        help='Directory of the processed files. If not given, it will be '
        'placed under the same parent directory as original_dir.')
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=5,
        help='Downsample FPS to `1 / sample_rate`. Default: 5.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    h36m = PreprocessH36m(
        metadata=args.metadata,
        original_dir=args.original,
        extracted_dir=args.extracted,
        processed_dir=args.processed,
        sample_rate=args.sample_rate)
    h36m.extract_tgz()
    h36m.generate_cameras_file()
    h36m.generate_annotations()
