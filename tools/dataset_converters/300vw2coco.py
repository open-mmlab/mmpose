# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import multiprocessing
import os
import subprocess
from glob import glob
from os.path import join

import numpy as np
from PIL import Image


def extract_frames(video_path):
    # Get the base path and video name
    base_path, video_name = os.path.split(video_path)
    # Remove the extension from the video name to get the folder name
    folder_name = 'imgs'
    # Create the new folder path
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist;
        os.makedirs(folder_path)

        # Create the output file pattern
        output_pattern = os.path.join(folder_path, '%06d.png')

        # Call ffmpeg to extract the frames
        subprocess.call([
            'ffmpeg', '-i', video_path, '-q:v', '0', '-start_number', '1',
            output_pattern
        ])
    else:
        # Skip this video if the frame folder already exist!
        print(f'{folder_path} already exist. Skip {video_path}!')
        return


class Base300VW:

    def __init__(self):
        extra_path = './tests/data/300vw/broken_frames.npy'
        self.broken_frames = np.load(extra_path, allow_pickle=True).item()
        self.videos_full = [
            '001', '002', '003', '004', '007', '009', '010', '011', '013',
            '015', '016', '017', '018', '019', '020', '022', '025', '027',
            '028', '029', '031', '033', '034', '035', '037', '039', '041',
            '043', '044', '046', '047', '048', '049', '053', '057', '059',
            '112', '113', '114', '115', '119', '120', '123', '124', '125',
            '126', '138', '143', '144', '150', '158', '160', '203', '204',
            '205', '208', '211', '212', '213', '214', '218', '223', '224',
            '225', '401', '402', '403', '404', '405', '406', '407', '408',
            '409', '410', '411', '412', '505', '506', '507', '508', '509',
            '510', '511', '514', '515', '516', '517', '518', '519', '520',
            '521', '522', '524', '525', '526', '528', '529', '530', '531',
            '533', '537', '538', '540', '541', '546', '547', '548', '550',
            '551', '553', '557', '558', '559', '562'
        ]

        # Category 1 in laboratory and naturalistic well-lit conditions
        self.videos_test_1 = [
            '114', '124', '125', '126', '150', '158', '401', '402', '505',
            '506', '507', '508', '509', '510', '511', '514', '515', '518',
            '519', '520', '521', '522', '524', '525', '537', '538', '540',
            '541', '546', '547', '548'
        ]
        # Category 2 in real-world human-computer interaction applications
        self.videos_test_2 = [
            '203', '208', '211', '212', '213', '214', '218', '224', '403',
            '404', '405', '406', '407', '408', '409', '412', '550', '551',
            '553'
        ]
        # Category 3 in arbitrary conditions
        self.videos_test_3 = [
            '410', '411', '516', '517', '526', '528', '529', '530', '531',
            '533', '557', '558', '559', '562'
        ]

        self.videos_test = \
            self.videos_test_1 + self.videos_test_2 + self.videos_test_3
        self.videos_train = [
            i for i in self.videos_full if i not in self.videos_test
        ]

        self.videos_part = ['001', '401']

        self.point_num = 68


class Preprocess300VW(Base300VW):

    def __init__(self, dataset_root):
        super().__init__()
        self.dataset_root = dataset_root
        self._extract_frames()
        self.json_data = self._init_json_data()

    def _init_json_data(self):
        """Initialize JSON data structure."""
        return {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 1,
                'name': 'person'
            }]
        }

    def _extract_frames(self):
        """Extract frames from videos."""
        all_video_paths = glob(os.path.join(self.dataset_root, '*/vid.avi'))
        with multiprocessing.Pool() as pool:
            pool.map(extract_frames, all_video_paths)

    def _extract_keypoints_from_pts(self, file_path):
        """Extract keypoints from .pts files."""
        keypoints = []
        with open(file_path, 'r') as file:
            file_content = file.read()
        start_index = file_content.find('{')
        end_index = file_content.rfind('}')
        if start_index != -1 and end_index != -1:
            data_inside_braces = file_content[start_index + 1:end_index]
            lines = data_inside_braces.split('\n')
            for line in lines:
                if line.strip():
                    x, y = map(float, line.split())
                    keypoints.extend([x, y])
        else:
            print('No data found inside braces.')
        return keypoints

    def _get_video_list(self, video_list):
        """Get video list based on input type."""
        if isinstance(video_list, list):
            return self.videos_part
        elif isinstance(video_list, str):
            if hasattr(self, video_list):
                return getattr(self, video_list)
            else:
                raise KeyError
        elif video_list is None:
            return self.videos_part
        else:
            raise ValueError

    def _process_image(self, img_path):
        """Process image and return image dictionary."""
        image_dict = {}
        image_dict['file_name'] = os.path.relpath(img_path, self.dataset_root)
        image_pic = Image.open(img_path)
        pic_width, pic_height = image_pic.size
        image_dict['height'] = pic_height
        image_dict['width'] = pic_width
        image_pic.close()
        return image_dict

    def _process_annotation(self, annot_path, image_id, anno_id):
        """Process annotation and return annotation dictionary."""
        annotation = {
            'segmentation': [],
            'num_keypoints': self.point_num,
            'iscrowd': 0,
            'category_id': 1,
        }
        keypoints = self._extract_keypoints_from_pts(annot_path)
        keypoints3 = []
        for kp_i in range(1, 68 * 2 + 1):
            keypoints3.append(keypoints[kp_i - 1])
            if kp_i % 2 == 0:
                keypoints3.append(1)
        annotation['keypoints'] = keypoints3
        annotation = self._calculate_annotation_properties(
            annotation, keypoints)
        annotation['image_id'] = image_id
        annotation['id'] = anno_id
        return annotation

    def _calculate_annotation_properties(self, annotation, keypoints):
        """Calculate properties for annotation."""
        keypoints_x = []
        keypoints_y = []
        for j in range(self.point_num * 2):
            if j % 2 == 0:
                keypoints_x.append(keypoints[j])
            else:
                keypoints_y.append(keypoints[j])
        x_left = min(keypoints_x)
        x_right = max(keypoints_x)
        y_low = min(keypoints_y)
        y_high = max(keypoints_y)
        w = x_right - x_left
        h = y_high - y_low
        annotation['scale'] = math.ceil(max(w, h)) / 200
        annotation['area'] = w * h
        annotation['center'] = [(x_left + x_right) / 2, (y_low + y_high) / 2]
        return annotation

    def convert_annotations(self,
                            video_list=None,
                            json_save_name='anno_300vw.json'):
        """Convert 300vw original annotations to coco format."""
        video_list = self._get_video_list(video_list)
        image_id = 0
        anno_id = 0
        for video_id in video_list:
            annot_root = join(self.dataset_root, video_id, 'annot')
            img_dir = join(self.dataset_root, video_id, 'imgs')
            if not (os.path.isdir(annot_root) and os.path.isdir(img_dir)):
                print(f'{annot_root} or {img_dir} not found. skip {video_id}!')
                continue
            annots = sorted(os.listdir(annot_root))
            for annot in annots:
                frame_num = int(annot.split('.')[0])
                if video_id in self.broken_frames and \
                        frame_num in self.broken_frames[video_id]:
                    print(f'skip broken frames: {frame_num} in {video_id}')
                    continue

                img_path = os.path.join(img_dir, f'{frame_num:06d}.png')
                if not os.path.exists(img_path):
                    print(f'{img_path} not found. skip!')
                    continue

                # Process image and add to JSON data
                image_dict = self._process_image(img_path)
                image_dict['id'] = image_id

                # Construct annotation path
                annot_path = os.path.join(annot_root, annot)
                annotation = self._process_annotation(annot_path, image_id,
                                                      anno_id)

                # Process annotation and add to JSON data
                self.json_data['images'].append(image_dict)
                self.json_data['annotations'].append(annotation)

                image_id += 1
                anno_id += 1

            print(f'Annotations from "{annot_root}" have been converted.')

        self._save_json_data(json_save_name)

    def _save_json_data(self, json_save_name):
        json_save_path = os.path.join(self.dataset_root, json_save_name)
        with open(json_save_path, 'w') as json_file:
            json.dump(self.json_data, json_file, indent=4)


if __name__ == '__main__':
    convert300vw = Preprocess300VW(dataset_root='./tests/data/300vw')
    convert300vw.convert_annotations()
