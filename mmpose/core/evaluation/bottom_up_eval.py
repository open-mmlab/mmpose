import numpy as np
import torch

from mmpose.core.post_processing import (get_warp_matrix, transform_preds,
                                         warp_affine_joints)


def _get_multi_stage_heatmaps(outputs,
                              outputs_flip,
                              with_heatmaps,
                              flip_index=None,
                              project2image=True,
                              size_projected=None,
                              align_corners=False,
                              num_joints=None,
                              flip_paf=False):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.

    Args:
        outputs (list(torch.Tensor)): Outputs of network.
        outputs_flip (list(torch.Tensor)): Flip outputs of network.
        with_heatmaps (list[bool]): Option to output
            heatmaps for different stages.
        flip_index (list[int]): Keypoint flip index.
        project2image (bool): Option to resize to base scale.
        size_projected ([w, h]): Base size of heatmaps.
        align_corners (bool): Align corners when performing interpolation.
        num_joints (int): Number of joints.
        flip_paf (bool): Whether to flip for paf maps.

    Returns:
        tuple: A tuple containing multi-stage outputs.

        - outputs_with_flipped (list(torch.Tensor)): List of simple outputs
          and flip outputs.
        - heatmaps (torch.Tensor): Multi-stage heatmaps that are resized to
          the base size.
    """

    outputs_with_flipped = outputs
    heatmaps_avg = 0
    num_heatmaps = 0
    heatmaps = []

    flip_test = outputs_flip is not None

    # aggregate heatmaps from different stages
    for i, output in enumerate(outputs):
        if i != len(outputs) - 1:
            output = torch.nn.functional.interpolate(
                output,
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=align_corners)

        if with_heatmaps[i]:
            if num_joints is not None:
                heatmaps_avg += output[:, :num_joints]
            else:
                heatmaps_avg += output
            num_heatmaps += 1

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
            outputs_with_flipped.append(output)

            if with_heatmaps[i]:
                if flip_paf:
                    heatmaps_avg[:, ::2, :, :] -= output[:,
                                                         flip_index[::2], :, :]
                    heatmaps_avg[:,
                                 1::2, :, :] += output[:,
                                                       flip_index[1::2], :, :]
                else:
                    if num_joints is not None:
                        heatmaps_avg += output[:, :
                                               num_joints][:, flip_index, :, :]
                    else:
                        heatmaps_avg += output[:, flip_index, :, :]
                num_heatmaps += 1

        heatmaps.append(heatmaps_avg / num_heatmaps)

    if project2image and size_projected:
        heatmaps = [
            torch.nn.functional.interpolate(
                hms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=align_corners) for hms in heatmaps
        ]

    return outputs_with_flipped, heatmaps


def _get_multi_stage_tags(outputs,
                          outputs_flip,
                          with_heatmaps,
                          with_ae,
                          tag_per_joint=True,
                          flip_index=None,
                          project2image=True,
                          size_projected=None,
                          align_corners=False,
                          num_joints=None):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.

    Args:
        outputs (list(torch.Tensor)): Outputs of network.
        outputs_flip (list(torch.Tensor)): Flip outputs of network.
        with_heatmaps (list[bool]): Option to output
            heatmaps for different stages.
        with_ae (list[bool]): Option to output
            ae tags for different stages.
        tag_per_joint (bool): Option to use one tag map per joint.
        flip_index (list[int]): Keypoint flip index.
        project2image (bool): Option to resize to base scale.
        size_projected ([w, h]): Base size of heatmaps.
        align_corners (bool): Align corners when performing interpolation.
        num_joints (int): Number of joints.

    Returns:
        tags (torch.Tensor): Multi-stage tags that are resized to
        the base size.
    """

    tags = []

    flip_test = outputs_flip is not None

    # aggregate tags from different stages
    for i, output in enumerate(outputs):
        if i != len(outputs) - 1:
            output = torch.nn.functional.interpolate(
                output,
                size=(outputs[-1].size(2), outputs[-1].size(3)),
                mode='bilinear',
                align_corners=align_corners)

        # staring index of the associative embeddings
        offset_feat = num_joints if with_heatmaps[i] else 0

        if with_ae[i]:
            tags.append(output[:, offset_feat:])

    if flip_test and flip_index:
        # perform flip testing

        for i, output in enumerate(outputs_flip):
            if i != len(outputs_flip) - 1:
                output = torch.nn.functional.interpolate(
                    output,
                    size=(outputs_flip[-1].size(2), outputs_flip[-1].size(3)),
                    mode='bilinear',
                    align_corners=align_corners)
            output = torch.flip(output, [3])

            offset_feat = num_joints if with_heatmaps[i] else 0

            if with_ae[i]:
                tags.append(output[:, offset_feat:])
                if tag_per_joint:
                    tags[-1] = tags[-1][:, flip_index, :, :]

    if project2image and size_projected:
        tags = [
            torch.nn.functional.interpolate(
                tms,
                size=(size_projected[1], size_projected[0]),
                mode='bilinear',
                align_corners=align_corners) for tms in tags
        ]

    return tags


def get_multi_stage_outputs(outputs,
                            outputs_flip,
                            num_joints,
                            with_heatmaps,
                            with_ae,
                            tag_per_joint=True,
                            flip_index=None,
                            project2image=True,
                            size_projected=None,
                            align_corners=False):
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
        align_corners (bool): Align corners when performing interpolation.

    Returns:
        tuple: A tuple containing multi-stage outputs.

        - outputs (list(torch.Tensor)): List of simple outputs and
          flip outputs.
        - heatmaps (torch.Tensor): Multi-stage heatmaps that are resized to
          the base size.
        - tags (torch.Tensor): Multi-stage tags that are resized to
          the base size.
    """
    outputs_with_flipped, heatmaps = _get_multi_stage_heatmaps(
        outputs, outputs_flip, num_joints, with_heatmaps, flip_index,
        project2image, size_projected, align_corners)

    tags = _get_multi_stage_tags(outputs, outputs_flip, num_joints,
                                 with_heatmaps, with_ae, tag_per_joint,
                                 flip_index, project2image, size_projected,
                                 align_corners)

    return outputs_with_flipped, heatmaps, tags


def get_multi_stage_outputs_paf(outputs,
                                outputs_flip,
                                with_heatmaps,
                                with_pafs,
                                flip_index=None,
                                flip_index_paf=None,
                                project2image=True,
                                size_projected=None,
                                align_corners=False):
    """Inference the model to get multi-stage outputs (heatmaps & pafs), and
    resize them to base sizes.

    Args:
        outputs (dict): Outputs of network, including heatmaps and pafs.
        outputs_flip (dict): Flip outputs of network, including
            heatmaps and pafs.
        with_heatmaps (list[bool]): Option to output
            heatmaps for different stages.
        with_pafs (list[bool]): Option to output
            pafs for different stages.
        flip_index (list[int]): Keypoint flip index.
        flip_index_paf (list[int]): PAF flip index.
        project2image (bool): Option to resize to base scale.
        size_projected ([w, h]): Base size of heatmaps.
        align_corners (bool): Align corners when performing interpolation.

    Returns:
        tuple: A tuple containing multi-stage outputs.

        - outputs (list(torch.Tensor)): List of simple outputs and
          flip outputs.
        - heatmaps (torch.Tensor): Multi-stage heatmaps that are resized to
          the base size.
        - pafs (torch.Tensor): Multi-stage pafs that are resized to
          the base size.
    """

    outputs['heatmaps'], heatmaps = _get_multi_stage_heatmaps(
        outputs['heatmaps'],
        outputs_flip['heatmaps'],
        with_heatmaps,
        flip_index,
        project2image,
        size_projected,
        align_corners,
        flip_paf=True)

    outputs['pafs'], pafs = _get_multi_stage_heatmaps(
        outputs['pafs'],
        outputs_flip['pafs'],
        with_heatmaps,
        flip_index,
        project2image,
        size_projected,
        align_corners,
        flip_paf=True)

    return outputs, heatmaps, pafs


def aggregate_results(scale,
                      aggregated_heatmaps,
                      tags_list,
                      heatmaps,
                      tags,
                      test_scale_factor,
                      project2image,
                      flip_test,
                      align_corners=False):
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
        align_corners (bool): Align corners when performing interpolation.

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
                    align_corners=align_corners) for tms in tags
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
            align_corners=align_corners)

    return aggregated_heatmaps, tags_list


def aggregate_results_paf(aggregated_heatmaps,
                          aggregated_pafs,
                          heatmaps,
                          pafs,
                          project2image,
                          flip_test,
                          align_corners=False):
    """Aggregate multi-scale outputs.

    Note:
        batch size: N
        keypoints num : K
        paf maps num: P
        heatmap width: W
        heatmap height: H

    Args:
        aggregated_heatmaps (torch.Tensor | None): Aggregated heatmaps.
        aggregated_pafs (torch.Tensor | None): Aggregated pafs.
        heatmaps (List(torch.Tensor[NxKxWxH])): A batch of heatmaps.
        pafs (List(torch.Tensor[NxPxWxH])): A batch of paf maps.
        project2image (bool): Option to resize to base scale.
        flip_test (bool): Option to use flip test.
        align_corners (bool): Align corners when performing interpolation.

    Return:
        tuple: a tuple containing aggregated results.

        - aggregated_heatmaps (torch.Tensor): Heatmaps with multi scale.
        - aggregated_pafs (torch.Tensor): PAF maps of multi scale.
    """

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
            align_corners=align_corners)

    pafs_avg = (pafs[0] + pafs[1]) / 2.0 if flip_test else pafs[0]
    if aggregated_pafs is None:
        aggregated_pafs = pafs_avg
    elif project2image:
        aggregated_pafs += pafs_avg
    else:
        aggregated_pafs += torch.nn.functional.interpolate(
            pafs_avg,
            size=(aggregated_pafs.size(2), aggregated_pafs.size(3)),
            mode='bilinear',
            align_corners=align_corners)

    return aggregated_heatmaps, aggregated_pafs


def get_group_preds(grouped_joints,
                    center,
                    scale,
                    heatmap_size,
                    use_udp=False):
    """Transform the grouped joints back to the image.

    Args:
        grouped_joints (list): Grouped person joints.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.
        use_udp (bool): Unbiased data processing.
             Paper ref: Huang et al. The Devil is in the Details: Delving into
             Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        list: List of the pose result for each person.
    """
    if use_udp:
        if grouped_joints[0].shape[0] > 0:
            heatmap_size_t = np.array(heatmap_size, dtype=np.float32) - 1.0
            trans = get_warp_matrix(
                theta=0,
                size_input=heatmap_size_t,
                size_dst=scale,
                size_target=heatmap_size_t)
            grouped_joints[0][..., :2] = \
                warp_affine_joints(grouped_joints[0][..., :2], trans)
        results = [person for person in grouped_joints[0]]
    else:
        results = []
        for person in grouped_joints[0]:
            joints = transform_preds(person, center, scale, heatmap_size)
            results.append(joints)

    return results
