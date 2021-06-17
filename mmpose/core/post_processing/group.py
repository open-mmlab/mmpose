# ------------------------------------------------------------------------------
# Adapted from https://github.com/princeton-vl/pose-ae-train/
# Original licence: Copyright (c) 2017, umich-vl, under BSD 3-Clause License.
# ------------------------------------------------------------------------------

from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import torch
from munkres import Munkres
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

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
    def adjust(ans, heatmaps, use_udp=False):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            ans (list(np.ndarray)): Keypoint predictions.
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

    @abstractmethod
    def parse(self, *args, **kwargs):
        """Group keypoints into poses."""


class HeatmapParser(BaseBottomUpParser):
    """The heatmap parser for post processing."""

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

    def parse(self, heatmaps, tags, adjust=True, refine=True):
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
    """The paf parser for post processing.

    Paper ref: Cao, Zhe, et al. "OpenPose: realtime multi-person 2D pose
    estimation using Part Affinity Fields." (TPAMI'2019)

    TODO: Rewrite this class. Currently, this part is copied from
    'https://github.com/tensorboy/'
    'pytorch_Realtime_Multi-Person_Pose_Estimation'
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.thre1 = 0.1
        self.thre2 = 0.05

        # Heatmap indices to find each limb (joint connection).
        # Eg: limb_type=1 is
        # Neck->LShoulder, so joint_to_limb_heatmap_relationship[1]
        # represents the
        # indices of heatmaps to look for joints: neck=1, LShoulder=5
        # TODO should be self.ann_info['skeleton']
        self.joint_to_limb_heatmap_relationship = [[15, 13], [13,
                                                              11], [16, 14],
                                                   [14, 12], [11, 12], [5, 11],
                                                   [6, 12], [5, 6], [5, 7],
                                                   [6, 8], [7, 9], [8, 10],
                                                   [1, 2], [0, 1], [0, 2],
                                                   [1, 3], [2, 4], [3, 5],
                                                   [4, 6]]

        # PAF indices containing the x and y coordinates of the PAF for a
        # given limb.
        # Eg: limb_type=1 is Neck->LShoulder, so
        # PAFneckLShoulder_x=paf_xy_coords_per_limb[1][0] and
        # PAFneckLShoulder_y=paf_xy_coords_per_limb[1][1]
        self.paf_xy_coords_per_limb = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
                                       [10, 11], [12, 13], [14, 15], [16, 17],
                                       [18, 19], [20, 21], [22, 23], [24, 25],
                                       [26, 27], [28, 29], [30, 31], [32, 33],
                                       [34, 35], [36, 37]]

        self.NUM_LIMBS = len(self.joint_to_limb_heatmap_relationship)

    def find_peaks(self, img):
        """Given a (grayscale) image, find local maxima whose value is above a
        given threshold (param['thre1'])

        :param img: Input image (2d array) where we want to find peaks
        :return: 2d np.array containing the [x,y] coordinates of each
        peak found
        in the image
        """

        peaks_binary = (maximum_filter(
            img, footprint=generate_binary_structure(2, 1)) == img) * (
                img > self.thre1)
        # Note reverse ([::-1]): we return [[x y], [x y]...] instead of
        # [[y x], [y
        # x]...]
        return np.array(np.nonzero(peaks_binary)[::-1]).T

    def compute_resized_coords(self, coords, resizeFactor):
        """
        Given the index/coordinates of a cell in some input array (e.g. image
        ),
        provides the new coordinates if that array was resized by making it
        resizeFactor times bigger.
        E.g.: image of size 3x3 is resized to 6x6 (resizeFactor=2), we'd like
         to
        know the new coordinates of cell [1,2] -> Function would return [2.5,
        4.5]
        :param coords: Coordinates (indices) of a cell in some input array
        :param resizeFactor: Resize coefficient = shape_dest/shape_source.
        E.g.:
        resizeFactor=2 means the destination array is twice as big as the
        original one
        :return: Coordinates in an array of size
        shape_dest=resizeFactor*shape_source, expressing the array indices
        of the
        closest point to 'coords' if an image of size shape_source was
        resized to
        shape_dest
        """

        # 1) Add 0.5 to coords to get coordinates of center of the pixel
        # (e.g.
        # index [0,0] represents the pixel at location [0.5,0.5])
        # 2) Transform those coordinates to shape_dest, by multiplying by
        # resizeFactor
        # 3) That number represents the location of the pixel center in the
        # new array,
        # so subtract 0.5 to get coordinates of the array index/indices
        # (revert
        # step 1)
        return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5

    def NMS(self,
            heatmaps,
            upsampFactor=1.,
            bool_refine_center=True,
            bool_gaussian_filt=False):
        """
        NonMaximaSuppression: find peaks (local maxima) in a set of grayscale
         images
        :param heatmaps: set of grayscale images on which to find local maxima
         (3d np.array,
        with dimensions image_height x image_width x num_heatmaps)
        :param upsampFactor: Size ratio between CPM heatmap output and the
         input image size.
        Eg: upsampFactor=16 if original image was 480x640 and heatmaps are
         30x40xN
        :param bool_refine_center: Flag indicating whether:
         - False: Simply return the low-res peak found upscaled by upsampFactor
         (subject to grid-snap)
         - True: (Recommended, very accurate) Upsample a small
         patch around each
         low-res peak and
         fine-tune the location of the peak at the resolution of the original
         input image
        :param bool_gaussian_filt: Flag indicating whether to apply a
        1d-GaussianFilter (smoothing)
        to each upsampled patch before fine-tuning the location of each peak.
        :return: a NUM_JOINTS x 4 np.array where each row represents a joint
        type (0=nose, 1=neck...)
        and the columns indicate the {x,y} position, the score (probability)
         and a unique id (counter)
        """
        # MODIFIED BY CARLOS: Instead of upsampling the heatmaps to
        # heatmap_avg and
        # then performing NMS to find peaks, this step can be sped up
        # by ~25-50x by:
        # (9-10ms [with GaussFilt] or 5-6ms [without GaussFilt] vs
        # 250-280ms on RoG
        # 1. Perform NMS at (low-res) CPM's output resolution
        # 1.1. Find peaks using scipy.ndimage.filters.maximum_filter
        # 2. Once a peak is found, take a patch of 5x5 centered around
        # the peak, upsample it, and
        # fine-tune the position of the actual maximum.
        #  '-> That's equivalent to having found the peak on heatmap_avg,
        #  but much faster because we only
        #      upsample and scan the 5x5 patch instead of the full (e.g.)
        #      480x640

        joint_list_per_joint_type = []
        cnt_total_joints = 0

        # For every peak found, win_size specifies how many pixels in each
        # direction from the peak we take to obtain the patch that will be
        # upsampled. Eg: win_size=1 -> patch is 3x3; win_size=2 -> 5x5
        # (for BICUBIC interpolation to be accurate, win_size needs to
        # be >=2!)
        win_size = 2

        for joint in range(heatmaps.shape[2]):
            map_orig = heatmaps[:, :, joint]
            peak_coords = self.find_peaks(map_orig)
            peaks = np.zeros((len(peak_coords), 4))
            for i, peak in enumerate(peak_coords):
                if bool_refine_center:
                    x_min, y_min = np.maximum(0, peak - win_size)
                    x_max, y_max = np.minimum(
                        np.array(map_orig.T.shape) - 1, peak + win_size)

                    # Take a small patch around each peak and only upsample
                    # that
                    # tiny region
                    patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                    map_upsamp = cv2.resize(
                        patch,
                        None,
                        fx=upsampFactor,
                        fy=upsampFactor,
                        interpolation=cv2.INTER_CUBIC)

                    # Gaussian filtering takes an average of 0.8ms/peak
                    # (and there might be
                    # more than one peak per joint!) -> For now, skip it
                    # (it's
                    # accurate enough)
                    map_upsamp = self.gaussian_filter(
                        map_upsamp,
                        sigma=3) if bool_gaussian_filt else map_upsamp

                    # Obtain the coordinates of the maximum value in the
                    # patch
                    location_of_max = np.unravel_index(map_upsamp.argmax(),
                                                       map_upsamp.shape)
                    # Remember that peaks indicates [x,y] -> need to
                    # reverse it for
                    # [y,x]
                    location_of_patch_center = self.compute_resized_coords(
                        peak[::-1] - [y_min, x_min], upsampFactor)
                    # Calculate the offset wrt to the patch center where
                    # the actual
                    # maximum is
                    refined_center = (
                        location_of_max - location_of_patch_center)
                    peak_score = map_upsamp[location_of_max]
                else:
                    refined_center = [0, 0]
                    # Flip peak coordinates since they are [x,y] instead
                    # of [y,x]
                    peak_score = map_orig[tuple(peak[::-1])]
                peaks[i, :] = tuple([
                    int(round(x)) for x in self.compute_resized_coords(
                        peak_coords[i], upsampFactor) + refined_center[::-1]
                ]) + (peak_score, cnt_total_joints)
                cnt_total_joints += 1
            joint_list_per_joint_type.append(peaks)

        return joint_list_per_joint_type

    def find_connected_joints(self,
                              paf_upsamp,
                              joint_list_per_joint_type,
                              num_intermed_pts=10):
        """For every type of limb (eg: forearm, shin, etc.), look for every
        potential pair of joints (eg: every wrist-elbow combination) and
        evaluate the PAFs to determine which pairs are indeed body limbs.

        :param paf_upsamp: PAFs upsampled to the original input image
         resolution
        :param joint_list_per_joint_type: See 'return' doc of NMS()
        :param num_intermed_pts: Int indicating how many intermediate
        points to take
        between joint_src and joint_dst, at which the PAFs will be
        evaluated
        :return: List of NUM_LIMBS rows. For every limb_type (a row)
        we store
        a list of all limbs of that type found (eg: all the right
        forearms).
        For each limb (each item in connected_limbs[limb_type]), we
        store 5 cells:
        # {joint_src_id,joint_dst_id}: a unique number associated with
        each joint,
        # limb_score_penalizing_long_dist: a score of how good a
         connection
        of the joints is, penalized if the limb length is too long
        # {joint_src_index,joint_dst_index}: the index of the joint
         within
        all the joints of that type found (eg: the 3rd right elbow
         found)
        """
        connected_limbs = []

        # Auxiliary array to access paf_upsamp quickly
        limb_intermed_coords = np.empty((4, num_intermed_pts), dtype=np.intp)
        for limb_type in range(self.NUM_LIMBS):
            # List of all joints of type A found, where A is specified
            # by limb_type
            # (eg: a right forearm starts in a right elbow)
            joints_src = joint_list_per_joint_type[
                self.joint_to_limb_heatmap_relationship[limb_type][0]]
            # List of all joints of type B found, where B is specified
            # by limb_type
            # (eg: a right forearm ends in a right wrist)
            joints_dst = joint_list_per_joint_type[
                self.joint_to_limb_heatmap_relationship[limb_type][1]]
            if len(joints_src) == 0 or len(joints_dst) == 0:
                # No limbs of this type found (eg: no right forearms
                # found because
                # we didn't find any right wrists or right elbows)
                connected_limbs.append([])
            else:
                connection_candidates = []
                # Specify the paf index that contains the x-coord of
                # the paf for
                # this limb
                limb_intermed_coords[
                    2, :] = self.paf_xy_coords_per_limb[limb_type][0]
                # And the y-coord paf index
                limb_intermed_coords[
                    3, :] = self.paf_xy_coords_per_limb[limb_type][1]
                for i, joint_src in enumerate(joints_src):
                    # Try every possible joints_src[i]-joints_dst[j]
                    # pair and see
                    # if it's a feasible limb
                    for j, joint_dst in enumerate(joints_dst):
                        # Subtract the position of both joints to obtain
                        # the
                        # direction of the potential limb
                        limb_dir = joint_dst[:2] - joint_src[:2]
                        # Compute the distance/length of the potential
                        # limb (norm
                        # of limb_dir)
                        limb_dist = np.sqrt(np.sum(limb_dir**2)) + 1e-8
                        limb_dir = limb_dir / limb_dist
                        # Normalize limb_dir to be a unit vector

                        # Linearly distribute num_intermed_pts points
                        # from the x
                        # coordinate of joint_src to the x coordinate
                        # of joint_dst
                        limb_intermed_coords[1, :] = np.round(
                            np.linspace(
                                joint_src[0],
                                joint_dst[0],
                                num=num_intermed_pts))
                        limb_intermed_coords[0, :] = np.round(
                            np.linspace(
                                joint_src[1],
                                joint_dst[1],
                                num=num_intermed_pts)
                        )  # Same for the y coordinate
                        intermed_paf = paf_upsamp[limb_intermed_coords[0, :],
                                                  limb_intermed_coords[1, :],
                                                  limb_intermed_coords[
                                                      2:4, :]].T

                        score_intermed_pts = intermed_paf.dot(limb_dir)
                        score_penalizing_long_dist = score_intermed_pts.mean(
                        ) + min(0.5 * paf_upsamp.shape[0] / limb_dist - 1, 0)
                        # Criterion 1: At least 80% of the intermediate
                        # points have
                        # a score higher than thre2
                        criterion1 = (
                            np.count_nonzero(score_intermed_pts > self.thre2) >
                            0.8 * num_intermed_pts)
                        # Criterion 2: Mean score, penalized for large limb
                        # distances (larger than half the image height), is
                        # positive
                        criterion2 = (score_penalizing_long_dist > 0)
                        if criterion1 and criterion2:
                            # Last value is the combined paf(+limb_dist)
                            # + heatmap
                            # scores of both joints
                            connection_candidates.append([
                                i, j, score_penalizing_long_dist,
                                score_penalizing_long_dist + joint_src[2] +
                                joint_dst[2]
                            ])

                # Sort connection candidates based on their
                # score_penalizing_long_dist
                connection_candidates = sorted(
                    connection_candidates, key=lambda x: x[2], reverse=True)
                connections = np.empty((0, 5))
                # There can only be as many limbs as the smallest number of
                # source
                # or destination joints (eg: only 2 forearms if there's 5
                # wrists
                # but 2 elbows)
                max_connections = min(len(joints_src), len(joints_dst))
                # Traverse all potential joint connections (sorted by their
                # score)
                for potential_connection in connection_candidates:
                    i, j, s = potential_connection[0:3]
                    # Make sure joints_src[i] or joints_dst[j] haven't
                    # already been
                    # connected to other joints_dst or joints_src
                    if i not in connections[:, 3] and j not in connections[:,
                                                                           4]:
                        # [joint_src_id, joint_dst_id,
                        # limb_score_penalizing_long_dist, joint_src_index,
                        # joint_dst_index]
                        connections = np.vstack([
                            connections,
                            [joints_src[i][3], joints_dst[j][3], s, i, j]
                        ])
                        # Exit if we've already established max_connections
                        # connections (each joint can't be connected to more
                        # than
                        # one joint)
                        if len(connections) >= max_connections:
                            break
                connected_limbs.append(connections)

        return connected_limbs

    def group_limbs_of_same_person(self, connected_limbs, joint_list):
        """Associate limbs belonging to the same person together.

        :param connected_limbs: See 'return' doc of
        find_connected_joints()
        :param joint_list: unravel'd version of joint_list_per_joint
         [See 'return' doc of NMS()]
        :return: 2d np.array of size num_people x (NUM_JOINTS+2).
         For each person found:
        # First NUM_JOINTS columns contain the index (in joint_list)
         of the joints associated
        with that person (or -1 if their i-th joint wasn't found)
        # 2nd-to-last column: Overall score of the joints+limbs that
        belong to this person
        # Last column: Total count of joints found for this person
        """
        person_to_joint_assoc = []

        for limb_type in range(self.NUM_LIMBS):
            joint_src_type, joint_dst_type = \
                self.joint_to_limb_heatmap_relationship[limb_type]

            for limb_info in connected_limbs[limb_type]:
                person_assoc_idx = []
                for person, person_limbs in enumerate(person_to_joint_assoc):
                    if person_limbs[joint_src_type] == limb_info[
                            0] or person_limbs[joint_dst_type] == limb_info[1]:
                        person_assoc_idx.append(person)

                # If one of the joints has been associated to a person,
                # and either
                # the other joint is also associated with the same
                # person or not
                # associated to anyone yet:
                if len(person_assoc_idx) == 1:
                    person_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                    # If the other joint is not associated to anyone yet,
                    if person_limbs[joint_dst_type] != limb_info[1]:
                        # Associate it with the current person
                        person_limbs[joint_dst_type] = limb_info[1]
                        # Increase the number of limbs associated to
                        # this person
                        person_limbs[-1] += 1
                        # And update the total score (+= heatmap score
                        # of joint_dst
                        # + score of connecting joint_src with joint_dst)
                        person_limbs[-2] += joint_list[
                            limb_info[1].astype(int), 2] + limb_info[2]
                elif len(person_assoc_idx
                         ) == 2:  # if found 2 and disjoint, merge them
                    person1_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                    person2_limbs = person_to_joint_assoc[person_assoc_idx[1]]
                    membership = ((person1_limbs >= 0) &
                                  (person2_limbs >= 0))[:-2]
                    if not membership.any(
                    ):  # If both people have no same joints connected,
                        # merge them into a single person
                        # Update which joints are connected
                        person1_limbs[:-2] += (person2_limbs[:-2] + 1)
                        # Update the overall score and total count of
                        # joints
                        # connected by summing their counters
                        person1_limbs[-2:] += person2_limbs[-2:]
                        # Add the score of the current joint connection
                        # to the
                        # overall score
                        person1_limbs[-2] += limb_info[2]
                        person_to_joint_assoc.pop(person_assoc_idx[1])
                    else:  # Same case as len(person_assoc_idx)==1 above
                        person1_limbs[joint_dst_type] = limb_info[1]
                        person1_limbs[-1] += 1
                        person1_limbs[-2] += joint_list[
                            limb_info[1].astype(int), 2] + limb_info[2]
                else:  # No person has claimed any of these joints, create
                    # a new person
                    # Initialize person info to all -1
                    # (no joint associations)
                    row = -1 * np.ones(20)
                    # Store the joint info of the new connection
                    row[joint_src_type] = limb_info[0]
                    row[joint_dst_type] = limb_info[1]
                    # Total count of connected joints for this person: 2
                    row[-1] = 2
                    # Compute overall score: score joint_src + score
                    # joint_dst + score connection
                    # {joint_src,joint_dst}
                    row[-2] = sum(joint_list[limb_info[:2].astype(int),
                                             2]) + limb_info[2]
                    person_to_joint_assoc.append(row)

        # Delete people who have very few parts connected
        people_to_delete = []
        for person_id, person_info in enumerate(person_to_joint_assoc):
            if person_info[-1] < 3 or person_info[-2] / person_info[-1] < 0.2:
                people_to_delete.append(person_id)
        # Traverse the list in reverse order so we delete indices
        # starting from the
        # last one (otherwise, removing item for example 0 would
        # modify the indices of
        # the remaining people to be deleted!)
        for index in people_to_delete[::-1]:
            person_to_joint_assoc.pop(index)

        # Appending items to a np.array can be very costly
        # (allocating new memory, copying over the array,
        # then adding new row)
        # Instead, we treat the set of people as a list
        # (fast to append items) and
        # only convert to np.array at the end
        return np.array(person_to_joint_assoc)

    def output_format(self, joint_list, person_to_joint_assoc):
        ans = []

        if len(person_to_joint_assoc) > 0:
            num_kpt = person_to_joint_assoc.shape[1] - 2

            for person in person_to_joint_assoc:
                ans_person = np.zeros((num_kpt, 5))
                for j in range(num_kpt):
                    joint_id = person[j]
                    if joint_id < 0:
                        continue
                    assert j == joint_list[joint_id, 4]
                    ans_person[j] = joint_list[joint_id]
                ans.append(ans_person)

        return ans

    def match(self, heatmaps, pafs):
        # Bottom-up approach:
        # Step 1: find all joints in the image (organized by joint type:
        # [0]=nose, [1]=neck...)
        # 4 = img_orig.shape[0] / float(heatmaps.shape[0])

        heatmaps = np.transpose(heatmaps[0], [1, 2, 0])
        pafs = np.transpose(pafs[0], [1, 2, 0])

        joint_list_per_joint_type = self.NMS(heatmaps, 4)
        # joint_list is an unravel'd version of joint_list_per_joint,
        # where we
        # add a 5th column to indicate the joint_type (0=nose, 1=neck...)
        joint_list = np.array([
            tuple(peak) + (joint_type, )
            for joint_type, joint_peaks in enumerate(joint_list_per_joint_type)
            for peak in joint_peaks
        ])

        # Step 2: find which joints go together to form limbs
        # (which wrists go
        # with which elbows)
        paf_upsamp = cv2.resize(
            pafs, (heatmaps.shape[1] * 4, heatmaps.shape[0] * 4),
            interpolation=cv2.INTER_CUBIC)
        connected_limbs = self.find_connected_joints(
            paf_upsamp, joint_list_per_joint_type)

        # Step 3: associate limbs that belong to the same person
        person_to_joint_assoc = self.group_limbs_of_same_person(
            connected_limbs, joint_list)

        ans = self.output_format(joint_list, person_to_joint_assoc)
        scores = [person[-2] for person in person_to_joint_assoc]

        return ans, scores

    def parse(self, heatmaps, pafs, adjust=True, refine=True):
        """Group keypoints into poses given heatmap and paf.

        Note:
            batch size: N
            number of keypoints: K
            number of paf maps: P
            heatmap height: H
            heatmap width: W

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            pafs (torch.Tensor[NxPxHxW]): model output pafs.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - ans (list(np.ndarray)): Pose results.
            - scores (list): Score of people.
        """

        # assert 0, 'The post-process of paf have not been completed.'
        ans, scores = self.match(heatmaps.cpu().numpy(), pafs.cpu().numpy())

        if adjust:
            if self.use_udp:
                for i in range(len(ans)):
                    if ans[i].shape[0] > 0:
                        ans[i][..., :2] = post_dark_udp(
                            ans[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                ans = self.adjust(ans, heatmaps)

        return ans, scores
