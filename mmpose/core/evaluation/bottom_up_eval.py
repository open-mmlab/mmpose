import numpy as np
import torch

from mmpose.core.post_processing import transform_preds
from mmpose.datasets.pipelines import get_warp_matrix, warp_affine_joints


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
