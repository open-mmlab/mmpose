# -----------------------------------------------------------------------------
# Adapted from https://github.com/anibali/h36m-fetch
# Original license: Copyright (c) Aiden Nibali, under the Apache License.
# -----------------------------------------------------------------------------

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
        processed_der (str): Directory of the process files. If not given, it
            will be placed under the same parent directory as original_dir.
        sample_rate (int): Downsample FPS to `1 / sample_rate`. Default: 5.
        smpl_train (str): Path to .pkl file containing SMPL pose and shape for
            training data. If not given, pose and shape parameters will not be
            saved to the annotation file.
        smpl_train (str): Path to .pkl file containing SMPL pose and shape for
            training data. If not given, pose and shape parameters will not be
            saved to the annotation file.
    """

    def __init__(self,
                 metadata,
                 original_dir,
                 extracted_dir=None,
                 processed_dir=None,
                 sample_rate=5,
                 smpl_train=None,
                 smpl_test=None):
        self.metadata = metadata
        self.original_dir = original_dir
        self.sample_rate = sample_rate
        self.smpl_train = smpl_train
        self.smpl_test = smpl_test

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
        self.joints_mapping = [
            14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6
        ]
        self.scale_factor = 1.2

    def extract_tgz(self):
        """Extract files from self.extrct_files."""
        os.makedirs(self.extracted_dir, exist_ok=True)
        for subject in self.subjects_annot:
            cur_dir = join(self.original_dir, subject.lower())
            for file in self.extract_files:
                filename = join(cur_dir, file + '.tgz')
                print('Extracting %s ...' % filename)
                with tarfile.open(filename) as tar:
                    tar.extractall(self.extracted_dir)
        print('Extraction done.\n')

    def generate_cameras_file(self):
        """Generate cameras.pkl which contains camera parameters for 11
        subjects each with 4 cameras."""
        cameras = {}
        for subject in range(1, 12):
            for camera in range(4):
                key = ('S%d' % subject, self.camera_ids[camera])
                cameras[key] = self._get_camera_params(camera, subject)

        out_file = self.processed_dir + '/cameras.pkl'
        with open(out_file, 'wb') as fout:
            pickle.dump(cameras, fout)
        print('Camera parameters have been written to \"%s\".\n' % out_file)

    def generate_annotations(self):
        """Generate annotations for training and testing data."""
        output_dir = join(self.processed_dir, 'annotations')
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

            out_file = join(output_dir, 'h36m_annot_%s.npz' % data_split)

            if eval('self.smpl_' + data_split) is not None:
                with open(eval('self.smpl_' + data_split), 'rb') as fin:
                    smpl = pickle.load(fin)

                poses_all = []
                shapes_all = []
                for i in range(len(imgnames_all)):
                    imgname = imgnames_all[i]
                    pose = smpl[imgname][:72]
                    shape = smpl[imgname][72:]
                    poses_all.append(pose)
                    shapes_all.append(shape)
                poses_all = np.stack(poses_all, axis=0)
                shapes_all = np.stack(shapes_all, axis=0)

                np.savez(
                    out_file,
                    imgname=imgnames_all,
                    center=centers_all,
                    scale=scales_all,
                    part=kps2d_all,
                    pose=poses_all,
                    shape=shapes_all,
                    S=kps3d_all)

            else:
                np.savez(
                    out_file,
                    imgname=imgnames_all,
                    center=centers_all,
                    scale=scales_all,
                    part=kps2d_all,
                    S=kps3d_all)

            print('All annotations of %sing data have been written to \"%s\". '
                  '%d samples in total.\n' %
                  (data_split, out_file, len(imgnames_all)))

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
        return '{}.{}'.format(
            self.sequence_mappings[subject][(action, subaction)], camera)

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

        # intrinsics
        c = metadata_slice[8:10, np.newaxis]
        f = metadata_slice[6:8, np.newaxis]

        # distortion
        k = metadata_slice[10:13, np.newaxis]
        p = metadata_slice[13:15, np.newaxis]

        return {
            'R': R,
            'T': T,
            'c': c,
            'f': f,
            'k': k,
            'p': p,
            'name': 'camera%d' % (camera + 1),
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
            kps_2d_17 = np.array(cdf['Pose'])

        num_frames = kps_2d_17.shape[1]
        kps_2d_17 = kps_2d_17.reshape((num_frames, 32, 2))[::self.sample_rate,
                                                           self.movable_joints]
        kps_2d_17 = np.concatenate(
            [kps_2d_17, np.ones((len(kps_2d_17), 17, 1))], axis=2)
        kps_2d = np.zeros((len(kps_2d_17), 24, 3))
        kps_2d[:, self.joints_mapping, :] = kps_2d_17[:, :, :]

        # load 3D keypoints
        with pycdf.CDF(
                join(subj_dir, 'MyPoseFeatures', 'D3_Positions_mono',
                     basename + '.cdf')) as cdf:
            kps_3d_17 = np.array(cdf['Pose'])

        kps_3d_17 = kps_3d_17.reshape(
            (num_frames, 32, 3))[::self.sample_rate,
                                 self.movable_joints] / 1000.
        kps_3d_17 = np.concatenate(
            [kps_3d_17, np.ones((len(kps_3d_17), 17, 1))], axis=2)
        kps_3d = np.zeros((len(kps_3d_17), 24, 4))
        kps_3d[:, self.joints_mapping, :] = kps_3d_17[:, :, :]

        # calculate bounding boxes
        bboxes = np.stack([
            np.min(kps_2d_17[:, :, 0], axis=1),
            np.min(kps_2d_17[:, :, 1], axis=1),
            np.max(kps_2d_17[:, :, 0], axis=1),
            np.max(kps_2d_17[:, :, 1], axis=1)
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
        img_dir = join(self.processed_dir, 'imgs', subject, sub_base)
        os.makedirs(img_dir, exist_ok=True)
        prefix = join(subject, sub_base, sub_base)

        cap = cv2.VideoCapture(video_path)
        for i in range(0, num_frames, self.sample_rate):
            imgname = prefix + '_%06d.jpg' % (i + 1)
            imgnames.append(imgname)
            dest_path = join(self.processed_dir, 'imgs', imgname)
            if os.path.exists(dest_path):
                continue
            cap.set(1, i)
            success, img = cap.read()
            if success:
                cv2.imwrite(dest_path, img)
        cap.release()
        imgnames = np.array(imgnames)
        assert len(centers) == len(imgnames)

        print('Annoatations for sequence \"%s %s\" are loaded.'
              '%d samples in total.' % (subject, basename, len(imgnames)))

        return imgnames, centers, scales, kps_2d, kps_3d


if __name__ == '__main__':
    METADATA = '/data/Human3.6M/metadata.xml'
    ORIGINAL_DIR = '/data/Human3.6M/original'
    EXTRACTED_DIR = '/data/Human3.6M/extracted'

    h36m = PreprocessH36m(
        METADATA,
        ORIGINAL_DIR,
        EXTRACTED_DIR,
        smpl_train='/data/Human3.6M/process/h36m_smpl_train.pkl')
    h36m.extract_tgz()
    h36m.generate_cameras_file()
    h36m.generate_annotations()
