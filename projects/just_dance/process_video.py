# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from typing import Optional

import cv2
import mmcv
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress

from mmpose.apis import Pose2DInferencer
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.visualization import PoseLocalVisualizer

try:
    from .calculate_similarity import (calculate_similarity,
                                       select_piece_from_similarity)
    from .utils import (blend_images, convert_video_fps, get_smoothed_kpt,
                        resize_image_to_fixed_height)
except ImportError:
    from calculate_similarity import (calculate_similarity,
                                      select_piece_from_similarity)
    from utils import (blend_images, convert_video_fps, get_smoothed_kpt,
                       resize_image_to_fixed_height)

model_cfg = dict(
    human=dict(
        model='rtmpose-t_8xb256-420e_aic-coco-256x192',
        det_model=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'configs/rtmdet-nano_one-person.py'),
        det_weights='https://download.openmmlab.com/mmpose/v1/projects/'
        'rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth',
    ),
    bear=dict(
        model='rtmpose-l_8xb256-420e_humanart-256x192',
        det_model='rtmdet-m',
        det_cat_ids=77,
    ),
)


class VideoProcessor:
    """A class to process videos for pose estimation and visualization."""

    def __init__(self):
        self.category = 'human'

    def _set_category(self, category):
        assert category in model_cfg
        self.category = category

    @property
    def pose_estimator(self) -> Pose2DInferencer:
        if not hasattr(self, '_pose_estimator'):
            self._pose_estimator = dict()
        if self.category not in self._pose_estimator:
            self._pose_estimator[self.category] = Pose2DInferencer(
                **(model_cfg[self.category]))
            self._pose_estimator[
                self.category].model.test_cfg['flip_test'] = False
        return self._pose_estimator[self.category]

    @property
    def visualizer(self) -> PoseLocalVisualizer:
        if hasattr(self, '_visualizer'):
            return self._visualizer
        elif hasattr(self, '_pose_estimator'):
            return self.pose_estimator.visualizer

        # init visualizer
        self._visualizer = PoseLocalVisualizer()
        metainfo_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)).rsplit(os.sep, 1)[0],
            'configs/_base_/datasets/coco.py')
        metainfo = parse_pose_metainfo(dict(from_file=metainfo_file))
        self._visualizer.set_dataset_meta(metainfo)
        return self._visualizer

    @torch.no_grad()
    def get_keypoints_from_frame(self, image: np.ndarray) -> np.ndarray:
        """Extract keypoints from a single video frame."""

        det_results = self.pose_estimator.detector(
            image, return_datasamples=True)['predictions']
        pred_instance = det_results[0].pred_instances

        if len(pred_instance) == 0:
            return np.zeros((1, 17, 3), dtype=np.float32)

        # only select the most significant person
        data_info = dict(
            img=image,
            bbox=pred_instance.bboxes.cpu().numpy()[:1],
            bbox_score=pred_instance.scores.cpu().numpy()[:1])

        if data_info['bbox_score'] < 0.2:
            return np.zeros((1, 17, 3), dtype=np.float32)

        data_info.update(self.pose_estimator.model.dataset_meta)
        data = self.pose_estimator.collate_fn(
            [self.pose_estimator.pipeline(data_info)])

        # custom forward
        data = self.pose_estimator.model.data_preprocessor(data, False)
        feats = self.pose_estimator.model.extract_feat(data['inputs'])
        pred_instances = self.pose_estimator.model.head.predict(
            feats,
            data['data_samples'],
            test_cfg=self.pose_estimator.model.test_cfg)[0]
        keypoints = np.concatenate(
            (pred_instances.keypoints, pred_instances.keypoint_scores[...,
                                                                      None]),
            axis=-1)

        return keypoints

    @torch.no_grad()
    def get_keypoints_from_video(self, video: str) -> np.ndarray:
        """Extract keypoints from a video."""

        video_fname = video.rsplit('.', 1)[0]
        if os.path.exists(f'{video_fname}_kpts.pth'):
            keypoints = torch.load(f'{video_fname}_kpts.pth')
            return keypoints

        video_reader = mmcv.VideoReader(video)

        if abs(video_reader.fps - 30) > 0.1:
            video_reader = mmcv.VideoReader(convert_video_fps(video))

        assert abs(video_reader.fps - 30) < 0.1, f'only support videos with ' \
            f'30 FPS, but the video {video_fname} has {video_reader.fps} fps'

        if os.path.basename(video_fname).startswith('bear'):
            self._set_category('bear')
        else:
            self._set_category('human')
        keypoints_list = []
        for i, frame in enumerate(video_reader):
            keypoints = self.get_keypoints_from_frame(frame)
            keypoints_list.append(keypoints)
        keypoints = np.concatenate(keypoints_list)
        torch.save(keypoints, f'{video_fname}_kpts.pth')
        return keypoints

    @torch.no_grad()
    def run(self,
            tch_video: str,
            stu_video: str,
            output_file: Optional[str] = None):
        # extract human poses
        tch_kpts = self.get_keypoints_from_video(tch_video)
        stu_kpts = self.get_keypoints_from_video(stu_video)

        # compute similarity
        similarity = calculate_similarity(tch_kpts, stu_kpts)

        # select piece
        piece_info = select_piece_from_similarity(similarity)

        # output
        tch_name = os.path.basename(tch_video).rsplit('.', 1)[0]
        stu_name = os.path.basename(stu_video).rsplit('.', 1)[0]
        if output_file is None:
            fname = f'{tch_name}-{stu_name}.mp4'
            output_file = os.path.join(tempfile.mkdtemp(), fname)
        return self.generate_output_video(tch_video, stu_video, output_file,
                                          tch_kpts, stu_kpts, piece_info)

    def generate_output_video(self, tch_video: str, stu_video: str,
                              output_file: str, tch_kpts: np.ndarray,
                              stu_kpts: np.ndarray, piece_info: dict) -> str:
        """Generate an output video with keypoints overlay."""

        tch_video_reader = mmcv.VideoReader(tch_video)
        stu_video_reader = mmcv.VideoReader(stu_video)
        for _ in range(piece_info['tch_start']):
            _ = next(tch_video_reader)
        for _ in range(piece_info['stu_start']):
            _ = next(stu_video_reader)

        score, last_vis_score = 0, 0
        video_writer = None
        for i in track_iter_progress(range(piece_info['length'])):
            tch_frame = mmcv.bgr2rgb(next(tch_video_reader))
            stu_frame = mmcv.bgr2rgb(next(stu_video_reader))
            tch_frame = resize_image_to_fixed_height(tch_frame, 300)
            stu_frame = resize_image_to_fixed_height(stu_frame, 300)

            stu_kpt = get_smoothed_kpt(stu_kpts, piece_info['stu_start'] + i,
                                       5)
            tch_kpt = get_smoothed_kpt(tch_kpts, piece_info['tch_start'] + i,
                                       5)

            # draw pose
            stu_kpt[..., 1] += (300 - 256)
            tch_kpt[..., 0] += (256 - 192)
            tch_kpt[..., 1] += (300 - 256)
            stu_inst = InstanceData(
                keypoints=stu_kpt[None, :, :2],
                keypoint_scores=stu_kpt[None, :, 2])
            tch_inst = InstanceData(
                keypoints=tch_kpt[None, :, :2],
                keypoint_scores=tch_kpt[None, :, 2])

            stu_out_img = self.visualizer._draw_instances_kpts(
                np.zeros((300, 256, 3)), stu_inst)
            tch_out_img = self.visualizer._draw_instances_kpts(
                np.zeros((300, 256, 3)), tch_inst)
            out_img = blend_images(
                stu_out_img, tch_out_img, blend_ratios=(1, 0.3))

            # draw score
            score_frame = piece_info['similarity'][i]
            score += score_frame * 1000
            if score - last_vis_score > 1500:
                last_vis_score = score
            self.visualizer.set_image(out_img)
            self.visualizer.draw_texts(
                'score: ', (60, 30),
                font_sizes=15,
                colors=(255, 255, 255),
                vertical_alignments='bottom')
            self.visualizer.draw_texts(
                f'{int(last_vis_score)}', (115, 30),
                font_sizes=30 * max(0.4, score_frame),
                colors=(255, 255, 255),
                vertical_alignments='bottom')
            out_img = self.visualizer.get_image()

            # concatenate
            concatenated_image = np.hstack((stu_frame, out_img, tch_frame))
            if video_writer is None:
                video_writer = cv2.VideoWriter(output_file,
                                               cv2.VideoWriter_fourcc(*'mp4v'),
                                               30,
                                               (concatenated_image.shape[1],
                                                concatenated_image.shape[0]))
            video_writer.write(mmcv.rgb2bgr(concatenated_image))

        if video_writer is not None:
            video_writer.release()
        return output_file


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('teacher_video', help='Path to the Teacher Video')
    parser.add_argument('student_video', help='Path to the Student Video')
    parser.add_argument(
        '--output-file', help='Path to save the output Video', default=None)
    args = parser.parse_args()

    processor = VideoProcessor()
    processor.run(args.teacher_video, args.student_video, args.output_file)
