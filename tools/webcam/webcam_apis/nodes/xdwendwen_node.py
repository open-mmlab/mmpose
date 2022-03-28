# Copyright (c) OpenMMLab. All rights reserved.
import json
from dataclasses import dataclass
from typing import List, Tuple, Union

import cv2
import numpy as np

from mmpose.datasets.dataset_info import DatasetInfo
from ..utils import load_image_from_disk_or_url
from .builder import NODES
from .frame_drawing_node import FrameDrawingNode


@dataclass
class DynamicInfo:
    pos_curr: Tuple[int, int] = (0, 0)
    pos_step: Tuple[int, int] = (0, 0)
    step_curr: int = 0


@NODES.register_module()
class XDwenDwenNode(FrameDrawingNode):
    """An effect drawing node that captures the face of a cat or dog and blend
    it into a Bing-Dwen-Dwen (the mascot of 2022 Beijing Winter Olympics).

    Parameters:
        name (str, optional): The node name (also thread name).
        frame_buffer (str): The name of the input buffer.
        output_buffer (str | list): The name(s) of the output buffer(s).
        mode_key (str | int): A hot key to switch the background image.
        resource_file (str): The annotation file of resource images, which
            should be in Labelbee format and contain both facial keypoint and
            region annotations.
        out_shape (tuple): The shape of output frame in (width, height).
    """

    dynamic_scale = 0.15
    dynamic_max_step = 15

    def __init__(
        self,
        name: str,
        frame_buffer: str,
        output_buffer: Union[str, List[str]],
        mode_key: Union[str, int],
        resource_file: str,
        out_shape: Tuple[int, int] = (480, 480),
        rigid_transform: bool = True,
    ):
        super().__init__(name, frame_buffer, output_buffer, enable=True)

        self.mode_key = mode_key
        self.mode_index = 0
        self.out_shape = out_shape
        self.rigid = rigid_transform

        self.latest_pred = None

        self.dynamic_info = DynamicInfo()

        self.register_event(
            self.mode_key, is_keyboard=True, handler_func=self.switch_mode)

        self._init_resource(resource_file)

    def _init_resource(self, resource_file):

        # The resource_file is a JSON file that contains the facial
        # keypoint and mask annotation information of the resource files.
        # The annotations should follow the label-bee standard format.
        # See https://github.com/open-mmlab/labelbee-client for details.
        with open(resource_file) as f:
            anns = json.load(f)
        resource_infos = []

        for ann in anns:
            # Load image
            img = load_image_from_disk_or_url(ann['url'])
            # Load result
            rst = json.loads(ann['result'])

            # Check facial keypoint information
            assert rst['step_1']['toolName'] == 'pointTool'
            assert len(rst['step_1']['result']) == 3

            keypoints = sorted(
                rst['step_1']['result'], key=lambda x: x['order'])
            keypoints = np.array([[pt['x'], pt['y']] for pt in keypoints])

            # Check facial mask
            assert rst['step_2']['toolName'] == 'polygonTool'
            assert len(rst['step_2']['result']) == 1
            assert len(rst['step_2']['result'][0]['pointList']) > 2

            mask_pts = np.array(
                [[pt['x'], pt['y']]
                 for pt in rst['step_2']['result'][0]['pointList']])

            mul = 1.0 + self.dynamic_scale

            w_scale = self.out_shape[0] / img.shape[1] * mul
            h_scale = self.out_shape[1] / img.shape[0] * mul

            img = cv2.resize(
                img,
                dsize=None,
                fx=w_scale,
                fy=h_scale,
                interpolation=cv2.INTER_CUBIC)

            keypoints *= [w_scale, h_scale]
            mask_pts *= [w_scale, h_scale]

            mask = cv2.fillPoly(
                np.zeros(img.shape[:2], dtype=np.uint8),
                [mask_pts.astype(np.int32)],
                color=1)

            res = {
                'img': img,
                'keypoints': keypoints,
                'mask': mask,
            }
            resource_infos.append(res)

        self.resource_infos = resource_infos

        self._reset_dynamic()

    def switch_mode(self):
        self.mode_index = (self.mode_index + 1) % len(self.resource_infos)

    def _reset_dynamic(self):
        x_tar = np.random.randint(int(self.out_shape[0] * self.dynamic_scale))
        y_tar = np.random.randint(int(self.out_shape[1] * self.dynamic_scale))

        x_step = (x_tar -
                  self.dynamic_info.pos_curr[0]) / self.dynamic_max_step
        y_step = (y_tar -
                  self.dynamic_info.pos_curr[1]) / self.dynamic_max_step

        self.dynamic_info.pos_step = (x_step, y_step)
        self.dynamic_info.step_curr = 0

    def draw(self, frame_msg):

        full_pose_results = frame_msg.get_pose_results()

        pred = None
        if full_pose_results:
            for pose_results in full_pose_results:
                if not pose_results['preds']:
                    continue

                pred = pose_results['preds'][0].copy()
                pred['dataset'] = DatasetInfo(pose_results['model_cfg'].data.
                                              test.dataset_info).dataset_name

                self.latest_pred = pred
                break

        # Use the latest pose result if there is none available in
        # the current frame.
        if pred is None:
            pred = self.latest_pred

        # Get the background image and facial annotations
        res = self.resource_infos[self.mode_index]
        img = frame_msg.get_image()
        canvas = res['img'].copy()
        mask = res['mask']
        kpts_tar = res['keypoints']

        if pred is not None:
            if pred['dataset'] == 'ap10k':
                # left eye: 0, right eye: 1, nose: 2
                kpts_src = pred['keypoints'][[0, 1, 2], :2]
            elif pred['dataset'] == 'coco_wholebody':
                # left eye: 1, right eye 2, nose: 0
                kpts_src = pred['keypoints'][[1, 2, 0], :2]
            else:
                raise ValueError('Can not obtain face landmark information'
                                 f'from dataset: {pred["type"]}')

            trans_mat = self._get_transform(kpts_src, kpts_tar)

            warp = cv2.warpAffine(img, trans_mat, dsize=canvas.shape[:2])
            cv2.copyTo(warp, mask, canvas)

        # Add random movement to the background
        xc, yc = self.dynamic_info.pos_curr
        xs, ys = self.dynamic_info.pos_step
        w, h = self.out_shape

        x = min(max(int(xc), 0), canvas.shape[1] - w + 1)
        y = min(max(int(yc), 0), canvas.shape[0] - h + 1)

        canvas = canvas[y:y + h, x:x + w]

        self.dynamic_info.pos_curr = (xc + xs, yc + ys)
        self.dynamic_info.step_curr += 1

        if self.dynamic_info.step_curr == self.dynamic_max_step:
            self._reset_dynamic()

        return canvas

    def _get_transform(self, kpts_src, kpts_tar):
        if self.rigid:
            # rigid transform
            n = kpts_src.shape[0]
            X = np.zeros((n * 2, 4), dtype=np.float32)
            U = np.zeros((n * 2, 1), dtype=np.float32)
            X[:n, :2] = kpts_src
            X[:n, 2] = 1
            X[n:, 0] = kpts_src[:, 1]
            X[n:, 1] = -kpts_src[:, 0]
            X[n:, 3] = 1

            U[:n, 0] = kpts_tar[:, 0]
            U[n:, 0] = kpts_tar[:, 1]

            M = np.linalg.pinv(X).dot(U).flatten()

            trans_mat = np.array([[M[0], M[1], M[2]], [-M[1], M[0], M[3]]],
                                 dtype=np.float32)

        else:
            # normal affine transform
            # adaptive horizontal flipping
            if (np.linalg.norm(kpts_tar[0] - kpts_tar[2]) -
                    np.linalg.norm(kpts_tar[1] - kpts_tar[2])) * (
                        np.linalg.norm(kpts_src[0] - kpts_src[2]) -
                        np.linalg.norm(kpts_src[1] - kpts_src[2])) < 0:
                kpts_src = kpts_src[[1, 0, 2], :]
            trans_mat, _ = cv2.estimateAffine2D(
                kpts_src.astype(np.float32), kpts_tar.astype(np.float32))

        return trans_mat
