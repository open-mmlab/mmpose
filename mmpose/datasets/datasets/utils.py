# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import numpy as np
from mmengine import Config


def parse_pose_metainfo(metainfo: dict):
    """Load meta information of pose dataset and check its integrity.

    Args:
        metainfo (dict): Raw data of pose meta information, which should
            contain following contents:

            - "dataset_name" (str): The name of the dataset
            - "keypoint_info" (dict): The keypoint-related meta information,
                e.g., name, upper/lower body, and symmetry
            - "skeleton_info" (dict): The skeleton-related meta information,
                e.g., start/end keypoint of limbs
            - "joint_weights" (list[float]): The loss weights of keypoints
            - "sigmas" (list[float]): The keypoint distribution parameters
                to calculate OKS score. See `COCO keypoint evaluation
                <https://cocodataset.org/#keypoints-eval>`__.

            An example of metainfo is shown as follows.

            .. code-block:: none
                {
                    "dataset_name": "coco",
                    "keypoint_info":
                    {
                        0:
                        {
                            "name": "nose",
                            "type": "upper",
                            "swap": "",
                            "color": [51, 153, 255],
                        },
                        1:
                        {
                            "name": "right_eye",
                            "type": "upper",
                            "swap": "left_eye",
                            "color": [51, 153, 255],
                        },
                        ...
                    },
                    "skeleton_info":
                    {
                        0:
                        {
                            "link": ("left_ankle", "left_knee"),
                            "color": [0, 255, 0],
                        },
                        ...
                    },
                    "joint_weights": [1., 1., ...],
                    "sigmas": [0.026, 0.025, ...],
                }


            A special case is that `metainfo` can have the key "from_file",
            which should be the path of a config file. In this case, the
            actual metainfo will be loaded by:

            .. code-block:: python
                metainfo = mmengine.Config.fromfile(metainfo['from_file'])

    Returns:
        Dict: pose meta information that contains following contents:

        - "dataset_name" (str): Same as ``"dataset_name"`` in the input
        - "num_keypoints" (int): Number of keypoints
        - "keypoint_id2name" (dict): Mapping from keypoint id to name
        - "keypoint_name2id" (dict): Mapping from keypoint name to id
        - "upper_body_ids" (list): Ids of upper-body keypoint
        - "lower_body_ids" (list): Ids of lower-body keypoint
        - "flip_indices" (list): The Id of each keypoint's symmetric keypoint
        - "flip_pairs" (list): The Ids of symmetric keypoint pairs
        - "keypoint_colors" (numpy.ndarray): The keypoint color matrix of
            shape [K, 3], where each row is the color of one keypint in bgr
        - "num_skeleton_links" (int): The number of links
        - "skeleton_links" (list): The links represented by Id pairs of start
             and end points
        - "skeleton_link_colors" (numpy.ndarray): The link color matrix
        - "dataset_keypoint_weights" (numpy.ndarray): Same as the
            ``"joint_weights"`` in the input
        - "sigmas" (numpy.ndarray): Same as the ``"sigmas"`` in the input
    """

    if 'from_file' in metainfo:
        cfg_file = metainfo['from_file']
        if not osp.isfile(cfg_file):
            # Search configs in 'mmpose/.mim/configs/' in case that mmpose
            # is installed in non-editable mode.
            import mmpose
            mmpose_path = osp.dirname(mmpose.__file__)
            _cfg_file = osp.join(mmpose_path, '.mim', 'configs', '_base_',
                                 'datasets', osp.basename(cfg_file))
            if osp.isfile(_cfg_file):
                warnings.warn(
                    f'The metainfo config file "{cfg_file}" does not exist. '
                    f'A matched config file "{_cfg_file}" will be used '
                    'instead.')
                cfg_file = _cfg_file
            else:
                raise FileNotFoundError(
                    f'The metainfo config file "{cfg_file}" does not exist.')

        # TODO: remove the nested structure of dataset_info
        # metainfo = Config.fromfile(metainfo['from_file'])
        metainfo = Config.fromfile(cfg_file).dataset_info

    # check data integrity
    assert 'dataset_name' in metainfo
    assert 'keypoint_info' in metainfo
    assert 'skeleton_info' in metainfo
    assert 'joint_weights' in metainfo
    assert 'sigmas' in metainfo

    # parse metainfo
    parsed = dict(
        dataset_name=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed['dataset_name'] = metainfo['dataset_name']

    # parse keypoint information
    parsed['num_keypoints'] = len(metainfo['keypoint_info'])

    for kpt_id, kpt in metainfo['keypoint_info'].items():
        kpt_name = kpt['name']
        parsed['keypoint_id2name'][kpt_id] = kpt_name
        parsed['keypoint_name2id'][kpt_name] = kpt_id
        parsed['keypoint_colors'].append(kpt.get('color', [255, 128, 0]))

        kpt_type = kpt.get('type', '')
        if kpt_type == 'upper':
            parsed['upper_body_ids'].append(kpt_id)
        elif kpt_type == 'lower':
            parsed['lower_body_ids'].append(kpt_id)

        swap_kpt = kpt.get('swap', '')
        if swap_kpt == kpt_name or swap_kpt == '':
            parsed['flip_indices'].append(kpt_name)
        else:
            parsed['flip_indices'].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed['flip_pairs']:
                parsed['flip_pairs'].append(pair)

    # parse skeleton information
    parsed['num_skeleton_links'] = len(metainfo['skeleton_info'])
    for _, sk in metainfo['skeleton_info'].items():
        parsed['skeleton_links'].append(sk['link'])
        parsed['skeleton_link_colors'].append(sk.get('color', [96, 96, 255]))

    # parse extra information
    parsed['dataset_keypoint_weights'] = np.array(
        metainfo['joint_weights'], dtype=np.float32)
    parsed['sigmas'] = np.array(metainfo['sigmas'], dtype=np.float32)

    # formatting
    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed['flip_pairs'] = _map(
        parsed['flip_pairs'], mapping=parsed['keypoint_name2id'])
    parsed['flip_indices'] = _map(
        parsed['flip_indices'], mapping=parsed['keypoint_name2id'])
    parsed['skeleton_links'] = _map(
        parsed['skeleton_links'], mapping=parsed['keypoint_name2id'])

    parsed['keypoint_colors'] = np.array(
        parsed['keypoint_colors'], dtype=np.uint8)
    parsed['skeleton_link_colors'] = np.array(
        parsed['skeleton_link_colors'], dtype=np.uint8)

    return parsed
