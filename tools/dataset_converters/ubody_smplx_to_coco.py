# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
from functools import partial
from typing import Dict, List

import mmengine
import numpy as np
import smplx
import torch
from pycocotools.coco import COCO


class SMPLX(object):

    def __init__(self, human_model_path):
        self.human_model_path = human_model_path
        self.layer_args = {
            'create_global_orient': False,
            'create_body_pose': False,
            'create_left_hand_pose': False,
            'create_right_hand_pose': False,
            'create_jaw_pose': False,
            'create_leye_pose': False,
            'create_reye_pose': False,
            'create_betas': False,
            'create_expression': False,
            'create_transl': False,
        }

        self.neutral_model = smplx.create(
            self.human_model_path,
            'smplx',
            gender='NEUTRAL',
            use_pca=False,
            use_face_contour=True,
            **self.layer_args)
        if torch.cuda.is_available():
            self.neutral_model = self.neutral_model.to('cuda:0')

        self.vertex_num = 10475
        self.face = self.neutral_model.faces
        self.shape_param_dim = 10
        self.expr_code_dim = 10
        # 22 (body joints) + 30 (hand joints) + 1 (face jaw joint)
        self.orig_joint_num = 53

        # yapf: disable
        self.orig_joints_name = (
            # 22 body joints
            'Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee',
            'Spine2', 'L_Ankle', 'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot',
            'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder',
            'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
            # left hand joints
            'L_Index_1', 'L_Index_2', 'L_Index_3', 'L_Middle_1', 'L_Middle_2',
            'L_Middle_3', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Ring_1',
            'L_Ring_2', 'L_Ring_3', 'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3',
            # right hand joints
            'R_Index_1', 'R_Index_2', 'R_Index_3', 'R_Middle_1', 'R_Middle_2',
            'R_Middle_3', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Ring_1',
            'R_Ring_2', 'R_Ring_3', 'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3',
            # 1 face jaw joint
            'Jaw',
        )
        self.orig_flip_pairs = (
            # body joints
            (1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19),
            (20, 21),
            # hand joints
            (22, 37), (23, 38), (24, 39), (25, 40), (26, 41), (27, 42),
            (28, 43), (29, 44), (30, 45), (31, 46), (32, 47), (33, 48),
            (34, 49), (35, 50), (36, 51),
        )
        # yapf: enable
        self.orig_root_joint_idx = self.orig_joints_name.index('Pelvis')
        self.orig_joint_part = {
            'body':
            range(
                self.orig_joints_name.index('Pelvis'),
                self.orig_joints_name.index('R_Wrist') + 1),
            'lhand':
            range(
                self.orig_joints_name.index('L_Index_1'),
                self.orig_joints_name.index('L_Thumb_3') + 1),
            'rhand':
            range(
                self.orig_joints_name.index('R_Index_1'),
                self.orig_joints_name.index('R_Thumb_3') + 1),
            'face':
            range(
                self.orig_joints_name.index('Jaw'),
                self.orig_joints_name.index('Jaw') + 1)
        }

        # changed SMPLX joint set for the supervision
        self.joint_num = (
            137  # 25 (body joints) + 40 (hand joints) + 72 (face keypoints)
        )
        # yapf: disable
        self.joints_name = (
            # 25 body joints
            'Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle',
            'R_Ankle', 'Neck', 'L_Shoulder', 'R_Shoulder', 'L_Elbow',
            'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Big_toe', 'L_Small_toe',
            'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Ear', 'R_Ear',
            'L_Eye', 'R_Eye', 'Nose',
            # left hand joints
            'L_Thumb_1', 'L_Thumb_2', 'L_Thumb_3', 'L_Thumb4', 'L_Index_1',
            'L_Index_2', 'L_Index_3', 'L_Index_4', 'L_Middle_1', 'L_Middle_2',
            'L_Middle_3', 'L_Middle_4', 'L_Ring_1', 'L_Ring_2', 'L_Ring_3',
            'L_Ring_4', 'L_Pinky_1', 'L_Pinky_2', 'L_Pinky_3', 'L_Pinky_4',
            # right hand joints
            'R_Thumb_1', 'R_Thumb_2', 'R_Thumb_3', 'R_Thumb_4', 'R_Index_1',
            'R_Index_2', 'R_Index_3', 'R_Index_4', 'R_Middle_1', 'R_Middle_2',
            'R_Middle_3', 'R_Middle_4', 'R_Ring_1', 'R_Ring_2', 'R_Ring_3',
            'R_Ring_4', 'R_Pinky_1', 'R_Pinky_2', 'R_Pinky_3', 'R_Pinky_4',
            # 72 face keypoints
            *[
                f'Face_{i}' for i in range(1, 73)
            ],
        )

        self.root_joint_idx = self.joints_name.index('Pelvis')
        self.lwrist_idx = self.joints_name.index('L_Wrist')
        self.rwrist_idx = self.joints_name.index('R_Wrist')
        self.neck_idx = self.joints_name.index('Neck')
        self.flip_pairs = (
            # body joints
            (1, 2), (3, 4), (5, 6), (8, 9), (10, 11), (12, 13), (14, 17),
            (15, 18), (16, 19), (20, 21), (22, 23),
            # hand joints
            (25, 45), (26, 46), (27, 47), (28, 48), (29, 49), (30, 50),
            (31, 51), (32, 52), (33, 53), (34, 54), (35, 55), (36, 56),
            (37, 57), (38, 58), (39, 59), (40, 60), (41, 61), (42, 62),
            (43, 63), (44, 64),
            # face eyebrow
            (67, 68), (69, 78), (70, 77), (71, 76), (72, 75), (73, 74),
            # face below nose
            (83, 87), (84, 86),
            # face eyes
            (88, 97), (89, 96), (90, 95), (91, 94), (92, 99), (93, 98),
            # face mouse
            (100, 106), (101, 105), (102, 104), (107, 111), (108, 110),
            # face lip
            (112, 116), (113, 115), (117, 119),
            # face contours
            (120, 136), (121, 135), (122, 134), (123, 133), (124, 132),
            (125, 131), (126, 130), (127, 129)
        )
        self.joint_idx = (
            0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21, 60, 61, 62, 63,
            64, 65, 59, 58, 57, 56, 55,  # body joints
            37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31,
            32, 33, 70,  # left hand joints
            52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46,
            47, 48, 75,  # right hand joints
            22, 15,  # jaw, head
            57, 56,  # eyeballs
            76, 77, 78, 79, 80, 81, 82, 83, 84, 85,  # eyebrow
            86, 87, 88, 89,  # nose
            90, 91, 92, 93, 94,  # below nose
            95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,  # eyes
            107,  # right mouth
            108, 109, 110, 111, 112,  # upper mouth
            113,  # left mouth
            114, 115, 116, 117, 118,  # lower mouth
            119,  # right lip
            120, 121, 122,  # upper lip
            123,  # left lip
            124, 125, 126,  # lower lip
            127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143,  # face contour
        )
        # yapf: enable

        self.joint_part = {
            'body':
            range(
                self.joints_name.index('Pelvis'),
                self.joints_name.index('Nose') + 1),
            'lhand':
            range(
                self.joints_name.index('L_Thumb_1'),
                self.joints_name.index('L_Pinky_4') + 1),
            'rhand':
            range(
                self.joints_name.index('R_Thumb_1'),
                self.joints_name.index('R_Pinky_4') + 1),
            'hand':
            range(
                self.joints_name.index('L_Thumb_1'),
                self.joints_name.index('R_Pinky_4') + 1),
            'face':
            range(
                self.joints_name.index('Face_1'),
                self.joints_name.index('Face_72') + 1)
        }


def read_annotation_file(annotation_file: str) -> List[Dict]:
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def process_scene_anno(scene: str, annotation_root: str, splits: np.array,
                       human_model_path: str):
    annos = read_annotation_file(
        osp.join(annotation_root, scene, 'smplx_annotation.json'))
    keypoint_annos = COCO(
        osp.join(annotation_root, scene, 'keypoint_annotation.json'))
    human_model = SMPLX(human_model_path)

    train_annos = []
    val_annos = []
    train_imgs = []
    val_imgs = []

    progress_bar = mmengine.ProgressBar(len(keypoint_annos.anns.keys()))
    for aid in keypoint_annos.anns.keys():
        ann = keypoint_annos.anns[aid]
        img = keypoint_annos.loadImgs(ann['image_id'])[0]
        if img['file_name'].startswith('/'):
            file_name = img['file_name'][1:]
        else:
            file_name = img['file_name']

        video_name = file_name.split('/')[-2]
        if 'Trim' in video_name:
            video_name = video_name.split('_Trim')[0]

        img_path = os.path.join(
            annotation_root.replace('annotations', 'images'), scene, file_name)
        if not os.path.exists(img_path):
            progress_bar.update()
            continue
        if str(aid) not in annos:
            progress_bar.update()
            continue

        smplx_param = annos[str(aid)]
        human_model_param = smplx_param['smplx_param']
        cam_param = smplx_param['cam_param']
        if 'lhand_valid' not in human_model_param:
            human_model_param['lhand_valid'] = ann['lefthand_valid']
            human_model_param['rhand_valid'] = ann['righthand_valid']
            human_model_param['face_valid'] = ann['face_valid']

        rotation_valid = np.ones((human_model.orig_joint_num),
                                 dtype=np.float32)
        coord_valid = np.ones((human_model.joint_num), dtype=np.float32)

        root_pose = human_model_param['root_pose']
        body_pose = human_model_param['body_pose']
        shape = human_model_param['shape']
        trans = human_model_param['trans']

        if 'lhand_pose' in human_model_param and human_model_param.get(
                'lhand_valid', False):
            lhand_pose = human_model_param['lhand_pose']
        else:
            lhand_pose = np.zeros(
                (3 * len(human_model.orig_joint_part['lhand'])),
                dtype=np.float32)
            rotation_valid[human_model.orig_joint_part['lhand']] = 0
            coord_valid[human_model.orig_joint_part['lhand']] = 0

        if 'rhand_pose' in human_model_param and human_model_param.get(
                'rhand_valid', False):
            rhand_pose = human_model_param['rhand_pose']
        else:
            rhand_pose = np.zeros(
                (3 * len(human_model.orig_joint_part['rhand'])),
                dtype=np.float32)
            rotation_valid[human_model.orig_joint_part['rhand']] = 0
            coord_valid[human_model.orig_joint_part['rhand']] = 0

        if 'jaw_pose' in human_model_param and \
            'expr' in human_model_param and \
                human_model_param.get('face_valid', False):
            jaw_pose = human_model_param['jaw_pose']
            expr = human_model_param['expr']
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
            expr = np.zeros((human_model.expr_code_dim), dtype=np.float32)
            rotation_valid[human_model.orig_joint_part['face']] = 0
            coord_valid[human_model.orig_joint_part['face']] = 0

        # init human model inputs
        device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        root_pose = torch.FloatTensor(root_pose).to(device).view(1, 3)
        body_pose = torch.FloatTensor(body_pose).to(device).view(-1, 3)
        lhand_pose = torch.FloatTensor(lhand_pose).to(device).view(-1, 3)
        rhand_pose = torch.FloatTensor(rhand_pose).to(device).view(-1, 3)
        jaw_pose = torch.FloatTensor(jaw_pose).to(device).view(-1, 3)
        shape = torch.FloatTensor(shape).to(device).view(1, -1)
        expr = torch.FloatTensor(expr).to(device).view(1, -1)
        trans = torch.FloatTensor(trans).to(device).view(1, -1)
        zero_pose = torch.zeros((1, 3), dtype=torch.float32, device=device)

        with torch.no_grad():
            output = human_model.neutral_model(
                betas=shape,
                body_pose=body_pose.view(1, -1),
                global_orient=root_pose,
                transl=trans,
                left_hand_pose=lhand_pose.view(1, -1),
                right_hand_pose=rhand_pose.view(1, -1),
                jaw_pose=jaw_pose.view(1, -1),
                leye_pose=zero_pose,
                reye_pose=zero_pose,
                expression=expr)

        joint_cam = output.joints[0].cpu().numpy()[human_model.joint_idx, :]
        joint_img = cam2pixel(joint_cam, cam_param['focal'],
                              cam_param['princpt'])

        joint_cam = (joint_cam - joint_cam[human_model.root_joint_idx, None, :]
                     )  # root-relative
        joint_cam[human_model.joint_part['lhand'], :] = (
            joint_cam[human_model.joint_part['lhand'], :] -
            joint_cam[human_model.lwrist_idx, None, :]
        )  # left hand root-relative
        joint_cam[human_model.joint_part['rhand'], :] = (
            joint_cam[human_model.joint_part['rhand'], :] -
            joint_cam[human_model.rwrist_idx, None, :]
        )  # right hand root-relative
        joint_cam[human_model.joint_part['face'], :] = (
            joint_cam[human_model.joint_part['face'], :] -
            joint_cam[human_model.neck_idx, None, :])  # face root-relative

        body_3d_size = 2
        output_hm_shape = (16, 16, 12)
        joint_img[human_model.joint_part['body'],
                  2] = ((joint_cam[human_model.joint_part['body'], 2].copy() /
                         (body_3d_size / 2) + 1) / 2.0 * output_hm_shape[0])
        joint_img[human_model.joint_part['lhand'],
                  2] = ((joint_cam[human_model.joint_part['lhand'], 2].copy() /
                         (body_3d_size / 2) + 1) / 2.0 * output_hm_shape[0])
        joint_img[human_model.joint_part['rhand'],
                  2] = ((joint_cam[human_model.joint_part['rhand'], 2].copy() /
                         (body_3d_size / 2) + 1) / 2.0 * output_hm_shape[0])
        joint_img[human_model.joint_part['face'],
                  2] = ((joint_cam[human_model.joint_part['face'], 2].copy() /
                         (body_3d_size / 2) + 1) / 2.0 * output_hm_shape[0])

        keypoints_2d = joint_img[:, :2].copy()
        keypoints_3d = joint_img.copy()
        keypoints_valid = coord_valid.reshape((-1, 1))

        ann['keypoints'] = keypoints_2d.tolist()
        ann['keypoints_3d'] = keypoints_3d.tolist()
        ann['keypoints_valid'] = keypoints_valid.tolist()
        ann['camera_param'] = cam_param
        img['file_name'] = os.path.join(scene, file_name)
        if video_name in splits:
            val_annos.append(ann)
            val_imgs.append(img)
        else:
            train_annos.append(ann)
            train_imgs.append(img)
        progress_bar.update()

    categories = [{
        'supercategory': 'person',
        'id': 1,
        'name': 'person',
        'keypoints': human_model.joints_name,
        'skeleton': human_model.flip_pairs
    }]
    train_data = {
        'images': train_imgs,
        'annotations': train_annos,
        'categories': categories
    }
    val_data = {
        'images': val_imgs,
        'annotations': val_annos,
        'categories': categories
    }

    mmengine.dump(
        train_data,
        osp.join(annotation_root, scene, 'train_3dkeypoint_annotation.json'))
    mmengine.dump(
        val_data,
        osp.join(annotation_root, scene, 'val_3dkeypoint_annotation.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data/UBody')
    parser.add_argument('--human-model-path', type=str, default='data/SMPLX')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()

    split_path = f'{args.data_root}/splits/intra_scene_test_list.npy'
    annotation_path = f'{args.data_root}/annotations'

    folders = os.listdir(annotation_path)
    folders = [f for f in folders if osp.isdir(osp.join(annotation_path, f))]
    human_model_path = args.human_model_path
    splits = np.load(split_path)

    if args.nproc > 1:
        mmengine.track_parallel_progress(
            partial(
                process_scene_anno,
                annotation_root=annotation_path,
                splits=splits,
                human_model_path=human_model_path), folders, args.nproc)
    else:
        mmengine.track_progress(
            partial(
                process_scene_anno,
                annotation_root=annotation_path,
                splits=splits,
                human_model_path=human_model_path), folders)
