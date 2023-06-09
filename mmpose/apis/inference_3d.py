# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmpose.structures import PoseDataSample


def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition, so that they are compatible with the definitions
    required for 3D pose lifting.

    Args:
        keypoints (ndarray[N, K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.

    Returns:
        ndarray[K, 2 or 3]: the transformed 2D keypoints.
    """
    assert pose_lift_dataset in [
        'Human36mDataset'], '`pose_lift_dataset` should be ' \
        f'`Human36mDataset`, but got {pose_lift_dataset}.'

    coco_style_datasets = [
        'CocoDataset', 'PoseTrack18VideoDataset', 'PoseTrack18Dataset'
    ]
    keypoints_new = np.zeros((keypoints.shape[0], 17, keypoints.shape[2]),
                             dtype=keypoints.dtype)
    if pose_lift_dataset == 'Human36mDataset':
        if pose_det_dataset in ['Human36mDataset']:
            keypoints_new = keypoints
        elif pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[:, 0] = (keypoints[:, 11] + keypoints[:, 12]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[:, 8] = (keypoints[:, 5] + keypoints[:, 6]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[:,
                          7] = (keypoints_new[:, 0] + keypoints_new[:, 8]) / 2
            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[:, 10] = (keypoints[:, 1] + keypoints[:, 2]) / 2
            # rearrange other keypoints
            keypoints_new[:, [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                keypoints[:, [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        elif pose_det_dataset in ['AicDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[:, 0] = (keypoints[:, 9] + keypoints[:, 6]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[:, 8] = (keypoints[:, 3] + keypoints[:, 0]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[:,
                          7] = (keypoints_new[:, 0] + keypoints_new[:, 8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 9] = (3 * keypoints[:, 13] + keypoints[:, 12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 10] = (5 * keypoints[:, 13] +
                                    7 * keypoints[:, 12]) / 12

            keypoints_new[:, [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[:, [6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
        elif pose_det_dataset in ['CrowdPoseDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[:, 0] = (keypoints[:, 6] + keypoints[:, 7]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[:, 8] = (keypoints[:, 0] + keypoints[:, 1]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[:,
                          7] = (keypoints_new[:, 0] + keypoints_new[:, 8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 9] = (3 * keypoints[:, 13] + keypoints[:, 12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[:, 10] = (5 * keypoints[:, 13] +
                                    7 * keypoints[:, 12]) / 12

            keypoints_new[:, [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[:, [7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    return keypoints_new


def extract_pose_sequence(pose_results, frame_idx, causal, seq_len, step=1):
    """Extract the target frame from 2D pose results, and pad the sequence to a
    fixed length.

    Args:
        pose_results (List[List[:obj:`PoseDataSample`]]): Multi-frame pose
            detection results stored in a list.
        frame_idx (int): The index of the frame in the original video.
        causal (bool): If True, the target frame is the last frame in
            a sequence. Otherwise, the target frame is in the middle of
            a sequence.
        seq_len (int): The number of frames in the input sequence.
        step (int): Step size to extract frames from the video.

    Returns:
        List[List[:obj:`PoseDataSample`]]: Multi-frame pose detection results
            stored in a nested list with a length of seq_len.
    """
    if causal:
        frames_left = seq_len - 1
        frames_right = 0
    else:
        frames_left = (seq_len - 1) // 2
        frames_right = frames_left
    num_frames = len(pose_results)

    # get the padded sequence
    pad_left = max(0, frames_left - frame_idx // step)
    pad_right = max(0, frames_right - (num_frames - 1 - frame_idx) // step)
    start = max(frame_idx % step, frame_idx - frames_left * step)
    end = min(num_frames - (num_frames - 1 - frame_idx) % step,
              frame_idx + frames_right * step + 1)
    pose_results_seq = [pose_results[0]] * pad_left + \
        pose_results[start:end:step] + [pose_results[-1]] * pad_right
    return pose_results_seq


def collate_pose_sequence(pose_results_2d,
                          with_track_id=True,
                          target_frame=-1):
    """Reorganize multi-frame pose detection results into individual pose
    sequences.

    Note:
        - The temporal length of the pose detection results: T
        - The number of the person instances: N
        - The number of the keypoints: K
        - The channel number of each keypoint: C

    Args:
        pose_results_2d (List[List[:obj:`PoseDataSample`]]): Multi-frame pose
            detection results stored in a nested list. Each element of the
            outer list is the pose detection results of a single frame, and
            each element of the inner list is the pose information of one
            person, which contains:

                - keypoints (ndarray[K, 2 or 3]): x, y, [score]
                - track_id (int): unique id of each person, required when
                    ``with_track_id==True```

        with_track_id (bool): If True, the element in pose_results is expected
            to contain "track_id", which will be used to gather the pose
            sequence of a person from multiple frames. Otherwise, the pose
            results in each frame are expected to have a consistent number and
            order of identities. Default is True.
        target_frame (int): The index of the target frame. Default: -1.

    Returns:
        List[:obj:`PoseDataSample`]: Indivisual pose sequence in with length N.
    """
    T = len(pose_results_2d)
    assert T > 0

    target_frame = (T + target_frame) % T  # convert negative index to positive

    N = len(
        pose_results_2d[target_frame])  # use identities in the target frame
    if N == 0:
        return []

    B, K, C = pose_results_2d[target_frame][0].pred_instances.keypoints.shape

    track_ids = None
    if with_track_id:
        track_ids = [res.track_id for res in pose_results_2d[target_frame]]

    pose_sequences = []
    for idx in range(N):
        pose_seq = PoseDataSample()
        gt_instances = InstanceData()
        pred_instances = InstanceData()

        for k in pose_results_2d[target_frame][idx].gt_instances.keys():
            gt_instances.set_field(
                pose_results_2d[target_frame][idx].gt_instances[k], k)
        for k in pose_results_2d[target_frame][idx].pred_instances.keys():
            if k != 'keypoints':
                pred_instances.set_field(
                    pose_results_2d[target_frame][idx].pred_instances[k], k)
        pose_seq.pred_instances = pred_instances
        pose_seq.gt_instances = gt_instances

        if not with_track_id:
            pose_seq.pred_instances.keypoints = np.stack([
                frame[idx].pred_instances.keypoints
                for frame in pose_results_2d
            ],
                                                         axis=1)
        else:
            keypoints = np.zeros((B, T, K, C), dtype=np.float32)
            keypoints[:, target_frame] = pose_results_2d[target_frame][
                idx].pred_instances.keypoints
            # find the left most frame containing track_ids[idx]
            for frame_idx in range(target_frame - 1, -1, -1):
                contains_idx = False
                for res in pose_results_2d[frame_idx]:
                    if res.track_id == track_ids[idx]:
                        keypoints[:, frame_idx] = res.pred_instances.keypoints
                        contains_idx = True
                        break
                if not contains_idx:
                    # replicate the left most frame
                    keypoints[:, :frame_idx + 1] = keypoints[:, frame_idx + 1]
                    break
            # find the right most frame containing track_idx[idx]
            for frame_idx in range(target_frame + 1, T):
                contains_idx = False
                for res in pose_results_2d[frame_idx]:
                    if res.track_id == track_ids[idx]:
                        keypoints[:, frame_idx] = res.pred_instances.keypoints
                        contains_idx = True
                        break
                if not contains_idx:
                    # replicate the right most frame
                    keypoints[:, frame_idx + 1:] = keypoints[:, frame_idx]
                    break
            pose_seq.pred_instances.keypoints = keypoints
        pose_sequences.append(pose_seq)

    return pose_sequences


def inference_pose_lifter_model(model,
                                pose_results_2d,
                                with_track_id=True,
                                image_size=None,
                                norm_pose_2d=False):
    """Inference 3D pose from 2D pose sequences using a pose lifter model.

    Args:
        model (nn.Module): The loaded pose lifter model
        pose_results_2d (List[List[:obj:`PoseDataSample`]]): The 2D pose
            sequences stored in a nested list.
        with_track_id: If True, the element in pose_results_2d is expected to
            contain "track_id", which will be used to gather the pose sequence
            of a person from multiple frames. Otherwise, the pose results in
            each frame are expected to have a consistent number and order of
            identities. Default is True.
        image_size (tuple|list): image width, image height. If None, image size
            will not be contained in dict ``data``.
        norm_pose_2d (bool): If True, scale the bbox (along with the 2D
            pose) to the average bbox scale of the dataset, and move the bbox
            (along with the 2D pose) to the average bbox center of the dataset.

    Returns:
        List[:obj:`PoseDataSample`]: 3D pose inference results. Specifically,
        the predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints_3d``.
    """
    init_default_scope(model.cfg.get('default_scope', 'mmpose'))
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    causal = model.cfg.test_dataloader.dataset.get('causal', False)
    target_idx = -1 if causal else len(pose_results_2d) // 2

    dataset_info = model.dataset_meta
    if dataset_info is not None:
        if 'stats_info' in dataset_info:
            bbox_center = dataset_info['stats_info']['bbox_center']
            bbox_scale = dataset_info['stats_info']['bbox_scale']
        else:
            bbox_center = None
            bbox_scale = None

    for i, pose_res in enumerate(pose_results_2d):
        for j, data_sample in enumerate(pose_res):
            kpts = data_sample.pred_instances.keypoints
            bboxes = data_sample.pred_instances.bboxes
            keypoints = []
            for k in range(len(kpts)):
                kpt = kpts[k]
                if norm_pose_2d:
                    bbox = bboxes[k]
                    center = np.array([[(bbox[0] + bbox[2]) / 2,
                                        (bbox[1] + bbox[3]) / 2]])
                    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                    keypoints.append((kpt[:, :2] - center) / scale *
                                     bbox_scale + bbox_center)
                else:
                    keypoints.append(kpt[:, :2])
            pose_results_2d[i][j].pred_instances.keypoints = np.array(
                keypoints)

    pose_sequences_2d = collate_pose_sequence(pose_results_2d, with_track_id,
                                              target_idx)

    if not pose_sequences_2d:
        return []

    data_list = []
    for i, pose_seq in enumerate(pose_sequences_2d):
        data_info = dict()

        keypoints_2d = pose_seq.pred_instances.keypoints
        keypoints_2d = np.squeeze(
            keypoints_2d, axis=0) if keypoints_2d.ndim == 4 else keypoints_2d

        T, K, C = keypoints_2d.shape

        data_info['keypoints'] = keypoints_2d
        data_info['keypoints_visible'] = np.ones((
            T,
            K,
        ), dtype=np.float32)
        data_info['lifting_target'] = np.zeros((K, 3), dtype=np.float32)
        data_info['lifting_target_visible'] = np.ones((K, 1), dtype=np.float32)

        if image_size is not None:
            assert len(image_size) == 2
            data_info['camera_param'] = dict(w=image_size[0], h=image_size[1])

        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    if data_list:
        # collate data list into a batch, which is a dict with following keys:
        # batch['inputs']: a list of input images
        # batch['data_samples']: a list of :obj:`PoseDataSample`
        batch = pseudo_collate(data_list)
        with torch.no_grad():
            results = model.test_step(batch)
    else:
        results = []

    return results
