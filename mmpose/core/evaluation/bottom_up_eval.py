import torch
import numpy as np
import math
from mmpose.core.post_processing import transform_preds


def get_multi_stage_outputs(outputs,
                            outputs_flip,
                            num_joints,
                            with_heatmaps,
                            with_ae,
                            tag_per_joint=True,
                            flip_index=None,
                            project2image=True,
                            size_projected=None):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.

    Args:
        outputs (list(torch.Tensor)): Outputs of network
        outputs_flip (list(torch.Tensor)): Flip outputs of network
        num_joints (int): Number of joints
        with_heatmaps (list[bool]): Option to output
            heatmaps for different stages.
        with_ae (list[bool]): Option to output
            ae tags for different stages.
        tag_per_joint (bool): Option to use one tag map per joint.
        flip_index (list[int]): Keypoint flip index.
        project2image (bool): Option to resize to base scale.
        size_projected ([w, h]): Base size of heatmaps.

    Returns:
        tuple: A tuple containing multi-stage outputs.

        - outputs (list(torch.Tensor)): List of simple outputs and
          flip outputs.
        - heatmaps (torch.Tensor): Multi-stage heatmaps that are resized to
          the base size.
        - tags (torch.Tensor): Multi-stage tags that are resized to
          the base size.
    """

    heatmaps_avg = 0
    num_heatmaps = 0
    heatmaps = []
    tags = []

    flip_test = outputs_flip is not None

    # aggregate heatmaps from different stages
    for i, output in enumerate(outputs):
        if i != len(outputs) - 1:
            output = torch.nn.functional.interpolate(
                output,
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=False)

        # staring index of the associative embeddings
        offset_feat = num_joints if with_heatmaps[i] else 0

        if with_heatmaps[i]:
            heatmaps_avg += output[:, :num_joints]
            num_heatmaps += 1

        if with_ae[i]:
            tags.append(output[:, offset_feat:])

    if num_heatmaps > 0:
        heatmaps.append(heatmaps_avg / num_heatmaps)

    if flip_test and flip_index:
        # perform flip testing
        heatmaps_avg = 0
        num_heatmaps = 0

        for i, output in enumerate(outputs_flip):
            if i != len(outputs_flip) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=False)
            output = torch.flip(output, [3])
            outputs.append(output)

            offset_feat = num_joints if with_heatmaps[i] else 0

            if with_heatmaps[i]:
                heatmaps_avg += output[:, :num_joints][:, flip_index, :, :]
                num_heatmaps += 1

            if with_ae[i]:
                tags.append(output[:, offset_feat:])
                if tag_per_joint:
                    tags[-1] = tags[-1][:, flip_index, :, :]

        heatmaps.append(heatmaps_avg / num_heatmaps)

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False) for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=False) for tms in tags
        ]

    return outputs, heatmaps, tags


def get_multi_stage_outputs_udp(outputs,
                                outputs_flip,
                                num_joints,
                                with_heatmaps,
                                with_ae,
                                tag_per_joint=True,
                                flip_index=None,
                                project2image=True,
                                size_projected=None):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.

    Args:
        outputs (list(torch.Tensor)): Outputs of network
        outputs_flip (list(torch.Tensor)): Flip outputs of network
        num_joints (int): Number of joints
        with_heatmaps (list[bool]): Option to output
            heatmaps for different stages.
        with_ae (list[bool]): Option to output
            ae tags for different stages.
        tag_per_joint (bool): Option to use one tag map per joint.
        flip_index (list[int]): Keypoint flip index.
        project2image (bool): Option to resize to base scale.
        size_projected ([w, h]): Base size of heatmaps.

    Returns:
        tuple: A tuple containing multi-stage outputs.

        - outputs (list(torch.Tensor)): List of simple outputs and
          flip outputs.
        - heatmaps (torch.Tensor): Multi-stage heatmaps that are resized to
          the base size.
        - tags (torch.Tensor): Multi-stage tags that are resized to
          the base size.
    """

    heatmaps_avg = 0
    num_heatmaps = 0
    align_corners = True
    heatmaps = []
    tags = []

    flip_test = outputs_flip is not None

    # aggregate heatmaps from different stages
    for i, output in enumerate(outputs):
        if i != len(outputs) - 1:
            output = torch.nn.functional.interpolate(
                output,
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=align_corners)

        # staring index of the associative embeddings
        offset_feat = num_joints if with_heatmaps[i] else 0

        if with_heatmaps[i]:
            heatmaps_avg += output[:, :num_joints]
            num_heatmaps += 1

        if with_ae[i]:
            tags.append(output[:, offset_feat:])

    if num_heatmaps > 0:
        heatmaps.append(heatmaps_avg / num_heatmaps)

    if flip_test and flip_index:
        # perform flip testing
        heatmaps_avg = 0
        num_heatmaps = 0

        for i, output in enumerate(outputs_flip):
            if i != len(outputs_flip) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=align_corners)
            output = torch.flip(output, [3])
            outputs.append(output)

            offset_feat = num_joints if with_heatmaps[i] else 0

            if with_heatmaps[i]:
                heatmaps_avg += output[:, :num_joints][:, flip_index, :, :]
                num_heatmaps += 1

            if with_ae[i]:
                tags.append(output[:, offset_feat:])
                if tag_per_joint:
                    tags[-1] = tags[-1][:, flip_index, :, :]

        heatmaps.append(heatmaps_avg / num_heatmaps)

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=align_corners) for hms in heatmaps
        ]

        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=align_corners) for tms in tags
        ]

    return outputs, heatmaps, tags


def aggregate_results(scale, aggregated_heatmaps, tags_list, heatmaps, tags,
                      test_scale_factor, project2image, flip_test):
    """Aggregate multi-scale outputs.

    Note:
        batch size: N
        keypoints num : K
        heatmap width: W
        heatmap height: H

    Args:
        scale (int): current scale
        aggregated_heatmaps (torch.Tensor | None): Aggregated heatmaps.
        tags_list (list(torch.Tensor)): Tags list of previous scale.
        heatmaps (List(torch.Tensor[NxKxWxH])): A batch of heatmaps.
        tags (List(torch.Tensor[NxKxWxH])): A batch of tag maps.
        test_scale_factor (List(int)): Multi-scale factor for testing.
        project2image (bool): Option to resize to base scale.
        flip_test (bool): Option to use flip test.

    Return:
        tuple: a tuple containing aggregated results.

        - aggregated_heatmaps (torch.Tensor): Heatmaps with multi scale.
        - tags_list (list(torch.Tensor)): Tag list of multi scale.
    """
    if scale == 1 or len(test_scale_factor) == 1:
        if aggregated_heatmaps is not None and not project2image:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(aggregated_heatmaps.size(2),
                          aggregated_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False) for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] +
                    heatmaps[1]) / 2.0 if flip_test else heatmaps[0]

    if aggregated_heatmaps is None:
        aggregated_heatmaps = heatmaps_avg
    elif project2image:
        aggregated_heatmaps += heatmaps_avg
    else:
        aggregated_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(aggregated_heatmaps.size(2), aggregated_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False)

    return aggregated_heatmaps, tags_list

def aggregate_results_udp(scale, aggregated_heatmaps, tags_list, heatmaps, tags,
                      test_scale_factor, project2image, flip_test):
    """Aggregate multi-scale outputs.

    Note:
        batch size: N
        keypoints num : K
        heatmap width: W
        heatmap height: H

    Args:
        scale (int): current scale
        aggregated_heatmaps (torch.Tensor | None): Aggregated heatmaps.
        tags_list (list(torch.Tensor)): Tags list of previous scale.
        heatmaps (List(torch.Tensor[NxKxWxH])): A batch of heatmaps.
        tags (List(torch.Tensor[NxKxWxH])): A batch of tag maps.
        test_scale_factor (List(int)): Multi-scale factor for testing.
        project2image (bool): Option to resize to base scale.
        flip_test (bool): Option to use flip test.

    Return:
        tuple: a tuple containing aggregated results.

        - aggregated_heatmaps (torch.Tensor): Heatmaps with multi scale.
        - tags_list (list(torch.Tensor)): Tag list of multi scale.
    """
    if scale == 1 or len(test_scale_factor) == 1:
        if aggregated_heatmaps is not None and not project2image:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(aggregated_heatmaps.size(2),
                          aggregated_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False) for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] +
                    heatmaps[1]) / 2.0 if flip_test else heatmaps[0]

    if aggregated_heatmaps is None:
        aggregated_heatmaps = heatmaps_avg
    elif project2image:
        aggregated_heatmaps += heatmaps_avg
    else:
        aggregated_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(aggregated_heatmaps.size(2), aggregated_heatmaps.size(3)),
            mode='bilinear',
            align_corners=True)

    return aggregated_heatmaps, tags_list


def get_warpmatrix(theta, size_input, size_dst, size_target):
    """
    :param theta: angle x y
    :param size_input:[w,h]
    :param size_dst: [w,h] i
    :param size_target: [w,h] b
    :return:
    """
    size_target = size_target
    theta = theta / 180.0 * math.pi
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix


def _affine_joints(joints, mat):
    """Affine the joints by the transform matrix."""
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
                  mat.T).reshape(shape)


def get_group_preds(grouped_joints, center, scale, heatmap_size):
    """Transform the grouped joints back to the image.

    Args:
        grouped_joints (list): Grouped person joints.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.

    Returns:
        list: List of the pose result for each person.
    """
    results = []
    for person in grouped_joints[0]:
        joints = transform_preds(person, center, scale, heatmap_size)
        results.append(joints)

    return results


def get_group_preds_udp(grouped_joints, center, scale, heatmap_size):
    """Transform the grouped joints back to the image.

    Args:
        grouped_joints (list): Grouped person joints.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.

    Returns:
        list: List of the pose result for each person.
    """
    if grouped_joints[0].shape[0] > 0:
        trans = get_warpmatrix(theta=0,
                               size_input=np.array(heatmap_size,
                                                   dtype=np.float32)-1.0,
                               size_dst=scale,
                               size_target=np.array(heatmap_size,
                                                    dtype=np.float32)-1.0)
        grouped_joints[0][:, :, :2] = \
            _affine_joints(grouped_joints[0][:, :, :2], trans)
    results = [person for person in grouped_joints[0]]
    return results
