from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from munkres import Munkres

from mmpose.core.evaluation import post_dark_udp


def _py_max_match(scores):
    """Apply munkres algorithm to get the best match.

    Args:
        scores(np.ndarray): cost matrix.

    Returns:
        np.ndarray: best match.
    """
    m = Munkres()
    tmp = m.compute(scores)
    tmp = np.array(tmp).astype(int)
    return tmp


def _match_by_tag(inp, params):
    """Match joints by tags. Use Munkres algorithm to calculate the best match
    for keypoints grouping.

    Note:
        number of keypoints: K
        max number of people in an image: M (M=30 by default)
        dim of tags: L
            If use flip testing, L=2; else L=1.

    Args:
        inp(tuple):
            tag_k (np.ndarray[KxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[KxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[KxM]): top k value of the
                feature maps per keypoint.
        params(Params): class Params().

    Returns:
        np.ndarray: result of pose groups.
    """
    assert isinstance(params, _Params), 'params should be class _Params()'

    tag_k, loc_k, val_k = inp

    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]),
                        dtype=np.float32)

    joint_dict = {}
    tag_dict = {}
    for i in range(params.num_joints):
        idx = params.joint_order[i]

        tags = tag_k[idx]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        mask = joints[:, 2] > params.detection_threshold
        tags = tags[mask]
        joints = joints[mask]

        if joints.shape[0] == 0:
            continue

        if i == 0 or len(joint_dict) == 0:
            for tag, joint in zip(tags, joints):
                key = tag[0]
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                tag_dict[key] = [tag]
        else:
            grouped_keys = list(joint_dict.keys())[:params.max_num_people]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            if (params.ignore_too_much
                    and len(grouped_keys) == params.max_num_people):
                continue

            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            num_added = diff.shape[0]
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                diff_normed = np.concatenate(
                    (diff_normed,
                     np.zeros((num_added, num_added - num_grouped),
                              dtype=np.float32) + 1e10),
                    axis=1)

            pairs = _py_max_match(diff_normed)
            for row, col in pairs:
                if (row < num_added and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold):
                    key = grouped_keys[col]
                    joint_dict[key][idx] = joints[row]
                    tag_dict[key].append(tags[row])
                else:
                    key = tags[row][0]
                    joint_dict.setdefault(key, np.copy(default_))[idx] = \
                        joints[row]
                    tag_dict[key] = [tags[row]]

    ans = np.array([joint_dict[i] for i in joint_dict]).astype(np.float32)
    return ans


class _Params:
    """A class of parameters for keypoint grouping.

    Args:
        cfg(Config): config.
    """

    def __init__(self, cfg):
        self.num_joints = cfg['num_joints']
        self.max_num_people = cfg['max_num_people']
        self.detection_threshold = cfg['detection_threshold']
        self.ignore_too_much = cfg['ignore_too_much']

        if self.num_joints == 17:
            self.joint_order = [
                i - 1 for i in
                [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = list(np.arange(self.num_joints))


class BaseBottomUpParser(metaclass=ABCMeta):
    """The base bottom-up parser for post processing."""

    def __init__(self, cfg):
        self.params = _Params(cfg)
        self.pool = torch.nn.MaxPool2d(cfg['nms_kernel'], 1,
                                       cfg['nms_padding'])
        self.use_udp = cfg.get('use_udp', False)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmap(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """

        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        heatmaps = heatmaps * maxm

        return heatmaps

    @staticmethod
    def top_k_value(feature_maps, M):
        """Find top_k values in the feature_maps.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M

        Args:
            feature_maps (torch.Tensor[NxKxHxW])

        Return:
            - val_k (torch.Tensor[NxKxM]):
                top k value of feature map per keypoint.
            - ind_k (torch.Tensor[NxKxM]):
                index of the selected locations.
        """
        N, K, H, W = feature_maps.size()
        feature_maps = feature_maps.view(N, K, -1)
        val_k, ind_k = feature_maps.topk(M, dim=2)

        return val_k, ind_k

    @staticmethod
    def refine(heatmap, tag, keypoints, use_udp=False):
        """Given initial keypoint predictions, we identify missing joints.

        Note:
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmap: np.ndarray(K, H, W).
            tag: np.ndarray(K, H, W) |  np.ndarray(K, H, W, L)
            keypoints: np.ndarray of size (K, 3 + L)
                        last dim is (x, y, score, tag).
            use_udp: bool-unbiased data processing

        Returns:
            np.ndarray: The refined keypoints.
        """

        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            tag = tag[..., None]

        tags = []
        for i in range(K):
            if keypoints[i, 2] > 0:
                # save tag value of detected keypoint
                x, y = keypoints[i][:2].astype(int)
                x = np.clip(x, 0, W - 1)
                y = np.clip(y, 0, H - 1)
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        prev_tag = np.mean(tags, axis=0)
        ans = []

        for _heatmap, _tag in zip(heatmap, tag):
            # distance of all tag values with mean tag of
            # current detected people
            distance_tag = (((_tag -
                              prev_tag[None, None, :])**2).sum(axis=2)**0.5)
            norm_heatmap = _heatmap - np.round(distance_tag)

            # find maximum position
            y, x = np.unravel_index(np.argmax(norm_heatmap), _heatmap.shape)
            xx = x.copy()
            yy = y.copy()
            # detection score at maximum position
            val = _heatmap[y, x]
            if not use_udp:
                # offset by 0.5
                x += 0.5
                y += 0.5

            # add a quarter offset
            if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
                x += 0.25
            else:
                x -= 0.25

            if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            ans.append((x, y, val))
        ans = np.array(ans)

        if ans is not None:
            for i in range(K):
                # add keypoint if it is not detected
                if ans[i, 2] > 0 and keypoints[i, 2] == 0:
                    keypoints[i, :3] = ans[i, :3]

        return keypoints

    @staticmethod
    def adjust(ans, heatmaps, use_udp=False):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of person: M
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            ans (list(np.array([M,K,3+]))): Keypoint predictions.
            heatmaps (torch.Tensor[NxKxHxW]): Heatmaps.
        """

        if use_udp:
            for i in range(len(ans)):
                if ans[i].shape[0] > 0:
                    ans[i][..., :2] = post_dark_udp(ans[i][..., :2].copy(),
                                                    heatmaps[i:i + 1, :])
        else:
            _, _, H, W = heatmaps.shape
            for batch_id, people in enumerate(ans):
                for people_id, people_i in enumerate(people):
                    for joint_id, joint in enumerate(people_i):
                        if joint[2] > 0:
                            x, y = joint[0:2]
                            xx, yy = int(x), int(y)
                            tmp = heatmaps[batch_id][joint_id]
                            if tmp[min(H - 1, yy + 1),
                                   xx] > tmp[max(0, yy - 1), xx]:
                                y += 0.25
                            else:
                                y -= 0.25

                            if tmp[yy, min(W - 1, xx +
                                           1)] > tmp[yy, max(0, xx - 1)]:
                                x += 0.25
                            else:
                                x -= 0.25
                            ans[batch_id][people_id, joint_id,
                                          0:2] = (x + 0.5, y + 0.5)
        return ans

    def filter_pose(self, ans, kpt_num_thr=3, mean_score_thr=0.2):
        """Filter out the poses with #keypoints < kpt_num_thr, and those with
        keypoint score < mean_score_thr.

        Note:
            number of person: M
            number of keypoints: K

        Args:
            filtered_ans (list(np.array([M,K,3+]))): Keypoint predictions.
        """
        filtered_ans = []
        for i in range(len(ans[0])):
            score = ans[0][i, :, 2]
            if sum(score > 0) < kpt_num_thr or (score[score > 0].mean() <
                                                mean_score_thr):
                continue
            filtered_ans.append(ans[0][i])
        filtered_ans = np.asarray(filtered_ans)

        return [filtered_ans]

    @abstractmethod
    def parse(self, *args, **kwargs):
        """Group keypoints into poses."""


class HeatmapParser(BaseBottomUpParser):
    """The associative embedding parser.

    Paper ref: Alejandro Newell et al. "Associative Embedding:
    End-to-end Learning for Joint Detection and Grouping." (NeurIPS'2017)

    Adapted from https://github.com/princeton-vl/pose-ae-train/
    Original licence: Copyright (c) 2017, umich-vl, under BSD 3-Clause License.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tag_per_joint = cfg['tag_per_joint']

        self.params.tag_threshold = cfg['tag_threshold']
        self.params.use_detection_val = cfg['use_detection_val']

    def match(self, tag_k, loc_k, val_k):
        """Group keypoints to human poses in a batch.

        Args:
            tag_k (np.ndarray[NxKxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[NxKxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[NxKxM]): top k value of the
                feature maps per keypoint.

        Returns:
            list
        """

        def _match(x):
            return _match_by_tag(x, self.params)

        return list(map(_match, zip(tag_k, loc_k, val_k)))

    def top_k(self, heatmaps, tags):
        """Find top_k values in the feature maps.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW])
            tags (torch.Tensor[NxKxHxWxL])

        Return:
            dict: A dict containing top_k values.

            - tag_k (np.ndarray[NxKxMxL]):
                tag corresponding to the top k values of
                feature map per keypoint.
            - loc_k (np.ndarray[NxKxMx2]):
                top k location of feature map per keypoint.
            - val_k (np.ndarray[NxKxM]):
                top k value of feature map per keypoint.
        """

        heatmaps = self.nms(heatmaps)
        N, K, H, W = heatmaps.size()
        val_k, ind_k = self.top_k_value(heatmaps, self.params.max_num_people)

        x = ind_k % W
        y = ind_k // W

        loc_k = torch.stack((x, y), dim=3)

        tags = tags.view(tags.size(0), tags.size(1), W * H, -1)
        if not self.tag_per_joint:
            tags = tags.expand(-1, self.params.num_joints, -1, -1)

        tag_k = torch.stack([
            torch.gather(tags[..., i], 2, ind_k) for i in range(tags.size(3))
        ],
                            dim=3)

        ans = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': loc_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        return ans

    def parse(self, heatmaps, tags, adjust=True, refine=True, filter=False):
        """Group keypoints into poses given heatmap and tag.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            tags (torch.Tensor[NxKxHxWxL]): model output tagmaps.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - ans (list(np.ndarray)): Pose results.
            - scores (list): Score of people.
        """
        ans = self.match(**self.top_k(heatmaps, tags))

        if len(ans) == 0:
            return [], []

        if filter:
            ans = self.filter_pose(ans)

        if adjust:
            ans = self.adjust(ans, heatmaps, self.use_udp)

        scores = [i[:, 2].mean() for i in ans[0]]

        if refine:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                heatmap_numpy = heatmaps[0].cpu().numpy()
                tag_numpy = tags[0].cpu().numpy()
                if not self.tag_per_joint:
                    tag_numpy = np.tile(tag_numpy,
                                        (self.params.num_joints, 1, 1, 1))
                ans[i] = self.refine(
                    heatmap_numpy, tag_numpy, ans[i], use_udp=self.use_udp)
            ans = [ans]

        return ans, scores


class PAFParser(BaseBottomUpParser):
    """The part-affinity field parser.

    Paper ref: Cao, Zhe, et al. "OpenPose: realtime multi-person 2D pose
    estimation using Part Affinity Fields." (TPAMI'2019)

    Adapted from 'https://github.com/Daniil-Osokin/
    lightweight-human-pose-estimation.pytorch'

    Original licence: Copyright 2018, under Apache License 2.0.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.paf_thr = 0.05
        self.add_neck = cfg.get('add_neck', False)

    def output_format(self, all_keypoints, pose_entries):
        """Format transform.

        Note:
            batch size: N
            number of people: M
            number of keypoints: K
            number of detected keypoints in the image: P

        Args:
            all_keypoints (np.ndarray(P, 4)): Each keypoint contains
                (x, y, score, keypoint id)
            pose_entries (np.ndarray(M, K + 2)): For each person,
                it contains K keypoint id, the human score, and
                the number of detected keypoints.

        Returns:
            ans (list(np.ndarray)): Pose results.
        """
        ans = []

        if len(pose_entries) > 0:
            for person in pose_entries:
                ans_person = np.zeros(
                    (self.params.num_joints, all_keypoints.shape[1]),
                    np.float32)
                for j in range(self.params.num_joints):
                    joint_id = int(person[j])
                    if joint_id < 0:
                        continue
                    ans_person[j] = all_keypoints[joint_id]
                ans.append(ans_person)
            return [np.stack(ans)]
        else:
            return []

    def connections_nms(self, a_idx, b_idx, affinity_scores):
        """From all retrieved connections that share the same starting/ending
        keypoints leave only the top-scoring ones.

        Args:
            a_idx (list(int)): index of the starting keypoints.
            b_idx (list(int)): index of the ending keypoints.
            affinity_scores (list(float)): affinity scores.

        Returns:
            a_idx (list(int)): index of the starting keypoints.
            b_idx (list(int)): index of the ending keypoints.
            affinity_scores (list(float)): affinity scores.
        """
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        idx = []
        has_kpt_a = set()
        has_kpt_b = set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_kpt_a and j not in has_kpt_b:
                idx.append(t)
                has_kpt_a.add(i)
                has_kpt_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

    def group_keypoints(self, all_keypoints_by_type, pafs):
        """Group keypoints based on part-affinity fields.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            all_keypoints_by_type (list(tuple)): list of tuples
                containing keypoint detection results (x, y, score, id).
            pafs (np.ndarray[W, H, C]): part-affinity fields

        Returns:
            - ans (list(np.array([M, K, 3+]))): Keypoint predictions.
            - scores (list): Score of people.
        """
        pose_entries = []
        all_keypoints = np.array(
            [item for sublist in all_keypoints_by_type for item in sublist])
        points_per_limb = 10
        grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
        all_keypoints_by_type = [
            np.array(keypoints, np.float32)
            for keypoints in all_keypoints_by_type
        ]
        for part_id in range(len(self.limb2paf)):
            part_pafs = pafs[:, :, self.limb2paf[part_id]]
            kpts_a = all_keypoints_by_type[self.limb2joint[part_id][0]]
            kpts_b = all_keypoints_by_type[self.limb2joint[part_id][1]]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            # Get vectors between all pairs of keypoints,
            # i.e. candidate limb vectors.
            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            # Sample points along every candidate limb vector.
            steps = (1 / (points_per_limb - 1) * vec_raw)
            points = steps * grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            # Compute affinity score between candidate
            # limb vectors and part affinity field.
            field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field *
                               vec).sum(-1).reshape(-1, points_per_limb)
            valid_affinity_scores = affinity_scores > self.paf_thr
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores *
                               valid_affinity_scores).sum(1) / (
                                   valid_num + 1e-6)
            success_ratio = valid_num / points_per_limb

            # Get a list of limbs according to the obtained affinity score.
            valid_limbs = np.where(
                np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]

            # Suppress incompatible connections.
            a_idx, b_idx, affinity_scores = self.connections_nms(
                a_idx, b_idx, affinity_scores)
            connections = list(
                zip(kpts_a[a_idx, 3].astype(np.int32),
                    kpts_b[b_idx, 3].astype(np.int32), affinity_scores))
            if len(connections) == 0:
                continue

            if part_id == 0:
                pose_entries = [
                    np.ones(self.params.num_joints + 2) * -1
                    for _ in range(len(connections))
                ]
                for i in range(len(connections)):
                    pose_entries[i][self.limb2joint[0][0]] = connections[i][0]
                    pose_entries[i][self.limb2joint[0][1]] = connections[i][1]
                    pose_entries[i][-1] = 2
                    pose_entries[i][-2] = np.sum(
                        all_keypoints[connections[i][0:2],
                                      2]) + connections[i][2]
            else:
                kpt_a_id = self.limb2joint[part_id][0]
                kpt_b_id = self.limb2joint[part_id][1]
                for i in range(len(connections)):
                    found_pose_list = []
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][
                                0] and pose_entries[j][kpt_b_id] == -1:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                            pose_entries[j][-1] += 1
                            pose_entries[j][-2] += all_keypoints[
                                connections[i][1], 2] + connections[i][2]
                            found_pose_list.append(
                                (j, all_keypoints[connections[i][1], 2] +
                                 connections[i][2]))

                        if pose_entries[j][kpt_b_id] == connections[i][
                                1] and pose_entries[j][kpt_a_id] == -1:
                            pose_entries[j][kpt_a_id] = connections[i][0]
                            pose_entries[j][-1] += 1
                            pose_entries[j][-2] += all_keypoints[
                                connections[i][1], 2] + connections[i][2]
                            found_pose_list.append(
                                (j, all_keypoints[connections[i][1], 2] +
                                 connections[i][2]))

                    if len(found_pose_list) == 0:
                        pose_entry = np.ones(self.params.num_joints + 2) * -1
                        pose_entry[kpt_a_id] = connections[i][0]
                        pose_entry[kpt_b_id] = connections[i][1]
                        pose_entry[-1] = 2
                        pose_entry[-2] = np.sum(
                            all_keypoints[connections[i][0:2],
                                          2]) + connections[i][2]
                        pose_entries.append(pose_entry)

                    elif len(found_pose_list) == 2:
                        # merge two pose entries
                        found_pose_list.sort(key=lambda x: x[0], reverse=True)
                        pose_entry = np.ones(self.params.num_joints + 2) * -1

                        entry_id1, score1 = found_pose_list[0]
                        entry_id2, score2 = found_pose_list[1]
                        assert score1 == score2

                        pose_entry1 = pose_entries.pop(entry_id1)
                        pose_entry2 = pose_entries.pop(entry_id2)

                        num_kpt = 0
                        score = pose_entry1[-2] + pose_entry2[-2] - score1

                        for j in range(self.params.num_joints):
                            kpt_id1 = int(pose_entry1[j])
                            kpt_id2 = int(pose_entry2[j])

                            if kpt_id1 == -1 and kpt_id2 == -1:
                                continue
                            elif kpt_id1 == -1 and kpt_id2 != -1:
                                pose_entry[j] = kpt_id2
                                num_kpt += 1
                            elif kpt_id2 == -1 and kpt_id1 != -1:
                                pose_entry[j] = kpt_id1
                                num_kpt += 1
                            else:
                                # both have the same joint-id,
                                # choose the one with higher score.
                                if all_keypoints[kpt_id1,
                                                 2] > all_keypoints[kpt_id2,
                                                                    2]:
                                    pose_entry[j] = kpt_id1
                                else:
                                    pose_entry[j] = kpt_id2
                                num_kpt += 1

                        pose_entry[-2] = score
                        pose_entry[-1] = num_kpt

                        pose_entries.append(pose_entry)

        ans = self.output_format(all_keypoints, pose_entries)
        scores = [person[-2] for person in pose_entries]

        return ans, scores

    def get_keypoints(self, heatmaps):
        """Extract keypoints from heatmaps.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.

        Returns:
            list(tuple): list of tuples containing keypoint detection
                results (x, y, score, id).
        """

        keypoint_num = 0
        all_keypoints_by_type = [[] for _ in range(self.params.num_joints)]

        heatmaps = self.nms(heatmaps)
        N, K, H, W = heatmaps.size()
        val_k, ind_k = self.top_k_value(heatmaps, self.params.max_num_people)

        x = ind_k % W
        y = ind_k // W

        loc_k = torch.stack((x, y), dim=3)

        for kpt_idx in range(self.params.num_joints):
            for m in range(self.params.max_num_people):
                if val_k[0][kpt_idx][m] < self.params.detection_threshold:
                    break
                else:
                    x = loc_k[0][kpt_idx][m][0].item()
                    y = loc_k[0][kpt_idx][m][1].item()
                    score = val_k[0][kpt_idx][m].item()
                    all_keypoints_by_type[kpt_idx].append(
                        (x, y, score, keypoint_num))
                    keypoint_num += 1

        return all_keypoints_by_type

    def define_limb(self, skeleton):
        if self.add_neck:
            # Heatmap indices to find each limb (joint connection).
            self.limb2joint = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
                               [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                               [12, 13], [1, 0], [0, 14], [14, 16], [0, 15],
                               [15, 17], [2, 16], [5, 17]]

            # PAF indices containing the x and y coordinates of the PAF for a
            # given limb.
            self.limb2paf = [[12, 13], [20, 21], [14, 15], [16, 17], [22, 23],
                             [24, 25], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
                             [10, 11], [28, 29], [30, 31], [34, 35], [32, 33],
                             [36, 37], [18, 19], [26, 27]]

        elif skeleton is None:
            # Heatmap indices to find each limb (joint connection).
            self.limb2joint = [[15, 13], [13, 11], [16, 14], [14, 12],
                               [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                               [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                               [1, 3], [2, 4], [3, 5], [4, 6]]

            # PAF indices containing the x and y coordinates of the PAF for a
            # given limb.
            self.limb2paf = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
                             [12, 13], [14, 15], [16, 17], [18, 19], [20, 21],
                             [22, 23], [24, 25], [26, 27], [28, 29], [30, 31],
                             [32, 33], [34, 35], [36, 37]]

        else:
            # Heatmap indices to find each limb (joint connection).
            self.limb2joint = skeleton

            # PAF indices containing the x and y coordinates of the PAF for a
            # given limb.
            self.limb2paf = np.array(range(len(self.limb2joint *
                                               2))).reshape(-1, 2).tolist()

        self.NUM_LIMBS = len(self.limb2joint)

    def parse(self,
              heatmaps,
              pafs,
              skeleton=None,
              adjust=True,
              refine=True,
              filter=False):
        """Group keypoints into poses given heatmap and paf.

        Note:
            batch size: N (currently we only support N==1)
            number of people: M
            number of keypoints: K
            number of paf maps: P
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            pafs (torch.Tensor[NxPxHxW]): model output pafs.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - ans (list(np.array([M,K,4]))): Keypoint predictions.
            - scores (list): Score of people.
        """

        assert heatmaps.shape[0] == 1, 'The batch size is ' \
            f'{heatmaps.shape[0]}, but we only support batch size==1.'

        self.define_limb(skeleton)

        all_keypoints_by_type = self.get_keypoints(heatmaps)
        pafs_np = np.transpose(pafs.detach().cpu().numpy()[0], [1, 2, 0])
        ans, scores = self.group_keypoints(all_keypoints_by_type, pafs_np)

        if len(ans) == 0:
            return [], []

        if filter:
            ans = self.filter_pose(ans)

        if adjust:
            if self.use_udp:
                for i in range(len(ans)):
                    if ans[i].shape[0] > 0:
                        ans[i][..., :2] = post_dark_udp(
                            ans[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                ans = self.adjust(ans, heatmaps)

        if refine:
            ans = ans[0]
            # for every detected person
            for i in range(len(ans)):
                heatmap_numpy = heatmaps[0].cpu().numpy()
                _, image_height, image_width = heatmap_numpy.shape
                y_coords = 2.0 * np.repeat(
                    np.arange(image_height)[:, None], image_width,
                    axis=1) / (image_height - 1.0) - 1.0
                x_coords = 2.0 * np.repeat(
                    np.arange(image_width)[None, :], image_height,
                    axis=0) / (image_width - 1.0) - 1.0
                coord_numpy = np.tile(
                    np.stack([x_coords, y_coords], axis=-1),
                    (self.params.num_joints, 1, 1, 1))
                ans[i] = self.refine(
                    heatmap_numpy, coord_numpy, ans[i], use_udp=self.use_udp)
            ans = [ans]

        return ans, scores
