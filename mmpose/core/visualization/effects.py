# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np


def apply_bugeye_effect(img,
                        pose_results,
                        left_eye_index,
                        right_eye_index,
                        kpt_thr=0.5):
    """Apply bug-eye effect.

    Args:
        img (np.ndarray): Image data.
        pose_results (list[dict]): The pose estimation results containing:
            - "bbox" ([K, 4(or 5)]): detection bbox in
                [x1, y1, x2, y2, (score)]
            - "keypoints" ([K,3]): keypoint detection result in [x, y, score]
        left_eye_index (int): Keypoint index of left eye
        right_eye_index (int): Keypoint index of right eye
        kpt_thr (float): The score threshold of required keypoints.
    """

    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    for pose in pose_results:
        bbox = pose['bbox']
        kpts = pose['keypoints']

        if kpts[left_eye_index, 2] < kpt_thr or kpts[right_eye_index,
                                                     2] < kpt_thr:
            continue

        kpt_leye = kpts[left_eye_index, :2]
        kpt_reye = kpts[right_eye_index, :2]
        for xc, yc in [kpt_leye, kpt_reye]:

            # distortion parameters
            k1 = 0.001
            epe = 1e-5

            scale = (bbox[2] - bbox[0])**2 + (bbox[3] - bbox[1])**2
            r2 = ((xx - xc)**2 + (yy - yc)**2)
            r2 = (r2 + epe) / scale  # normalized by bbox scale

            xx = (xx - xc) / (1 + k1 / r2) + xc
            yy = (yy - yc) / (1 + k1 / r2) + yc

        img = cv2.remap(
            img,
            xx,
            yy,
            interpolation=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REPLICATE)
    return img


def apply_sunglasses_effect(img,
                            pose_results,
                            sunglasses_img,
                            left_eye_index,
                            right_eye_index,
                            kpt_thr=0.5):
    """Apply sunglasses effect.

    Args:
        img (np.ndarray): Image data.
        pose_results (list[dict]): The pose estimation results containing:
            - "keypoints" ([K,3]): keypoint detection result in [x, y, score]
        sunglasses_img (np.ndarray): Sunglasses image with white background.
        left_eye_index (int): Keypoint index of left eye
        right_eye_index (int): Keypoint index of right eye
        kpt_thr (float): The score threshold of required keypoints.
    """

    hm, wm = sunglasses_img.shape[:2]
    # anchor points in the sunglasses mask
    pts_src = np.array([[0.3 * wm, 0.3 * hm], [0.3 * wm, 0.7 * hm],
                        [0.7 * wm, 0.3 * hm], [0.7 * wm, 0.7 * hm]],
                       dtype=np.float32)

    for pose in pose_results:
        kpts = pose['keypoints']

        if kpts[left_eye_index, 2] < kpt_thr or kpts[right_eye_index,
                                                     2] < kpt_thr:
            continue

        kpt_leye = kpts[left_eye_index, :2]
        kpt_reye = kpts[right_eye_index, :2]
        # orthogonal vector to the left-to-right eyes
        vo = 0.5 * (kpt_reye - kpt_leye)[::-1] * [-1, 1]

        # anchor points in the image by eye positions
        pts_tar = np.vstack(
            [kpt_reye + vo, kpt_reye - vo, kpt_leye + vo, kpt_leye - vo])

        h_mat, _ = cv2.findHomography(pts_src, pts_tar)
        patch = cv2.warpPerspective(
            sunglasses_img,
            h_mat,
            dsize=(img.shape[1], img.shape[0]),
            borderValue=(255, 255, 255))
        #  mask the white background area in the patch with a threshold 200
        mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (mask < 200).astype(np.uint8)
        img = cv2.copyTo(patch, mask, img)

    return img


def apply_moustache_effect(img,
                           pose_results,
                           moustache_img,
                           face_indices,
                           kpt_thr=0.5):
    """Apply moustache effect.

    Args:
        img (np.ndarray): Image data.
        pose_results (list[dict]): The pose estimation results containing:
            - "keypoints" ([K,3]): keypoint detection result in [x, y, score]
        moustache_img (np.ndarray): Moustache image with white background.
        left_eye_index (int): Keypoint index of left eye
        right_eye_index (int): Keypoint index of right eye
        kpt_thr (float): The score threshold of required keypoints.
    """

    hm, wm = moustache_img.shape[:2]
    # anchor points in the moustache mask
    pts_src = np.array([[1164, 741], [1729, 741], [1164, 1244], [1729, 1244]],
                       dtype=np.float32)

    for pose in pose_results:
        kpts = pose['keypoints']
        if kpts[face_indices[32], 2] < kpt_thr \
                or kpts[face_indices[34], 2] < kpt_thr \
                or kpts[face_indices[61], 2] < kpt_thr \
                or kpts[face_indices[63], 2] < kpt_thr:
            continue

        kpt_32 = kpts[face_indices[32], :2]
        kpt_34 = kpts[face_indices[34], :2]
        kpt_61 = kpts[face_indices[61], :2]
        kpt_63 = kpts[face_indices[63], :2]
        # anchor points in the image by eye positions
        pts_tar = np.vstack([kpt_32, kpt_34, kpt_61, kpt_63])

        h_mat, _ = cv2.findHomography(pts_src, pts_tar)
        patch = cv2.warpPerspective(
            moustache_img,
            h_mat,
            dsize=(img.shape[1], img.shape[0]),
            borderValue=(255, 255, 255))
        #  mask the white background area in the patch with a threshold 200
        mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (mask < 200).astype(np.uint8)
        img = cv2.copyTo(patch, mask, img)

    return img


def apply_saiyan_effect(img,
                        pose_results,
                        saiyan_img,
                        face_indices,
                        kpt_thr=0.5):
    """Apply saiyan hair effect.

    Args:
        img (np.ndarray): Image data.
        pose_results (list[dict]): The pose estimation results containing:
            - "keypoints" ([K,3]): keypoint detection result in [x, y, score]
        saiyan_img (np.ndarray): Saiyan image with transparent background.
        face_indices (int): Keypoint index of the face
        kpt_thr (float): The score threshold of required keypoints.
    """

    hm, wm = saiyan_img.shape[:2]
    # anchor points in the mask
    pts_src = np.array(
        [
            [84, 398],  # face kpt 0
            [331, 393],  # face kpt 16
            [84, 145],
            [331, 140]
        ],
        dtype=np.float32)

    for pose in pose_results:
        kpts = pose['keypoints']
        if kpts[face_indices[0], 2] < kpt_thr or kpts[face_indices[16],
                                                      2] < kpt_thr:
            continue

        kpt_0 = kpts[face_indices[0], :2]
        kpt_16 = kpts[face_indices[16], :2]
        # orthogonal vector
        vo = (kpt_0 - kpt_16)[::-1] * [-1, 1]

        # anchor points in the image by eye positions
        pts_tar = np.vstack([kpt_0, kpt_16, kpt_0 + vo, kpt_16 + vo])

        h_mat, _ = cv2.findHomography(pts_src, pts_tar)
        patch = cv2.warpPerspective(
            saiyan_img,
            h_mat,
            dsize=(img.shape[1], img.shape[0]),
            borderValue=(0, 0, 0))
        mask = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (mask > 1).astype(np.uint8)
        img = cv2.copyTo(patch, mask, img)

    return img


def apply_background_effect(img,
                            det_results,
                            background_img,
                            effect_region=(0.2, 0.2, 0.8, 0.8),
                            bbox_thr=0.5):
    """Change background.

    Args:
        img (np.ndarray): Image data.
        det_results (list[dict]): The detection results containing:

            - "cls_id" (int): Class index.
            - "label" (str): Class label (e.g. 'person').
            - "bbox" (ndarray:(5, )): bounding box result [x, y, w, h, score].
            - "mask" (ndarray:(w, h)): instance segmentation result.
        effect_region (tuple(4, )): The region to apply mask, the coordinates
            are normalized (x1, y1, x2, y2).
        background_img (np.ndarray): Background image.
    """
    background_img = background_img.copy()
    background_h, background_w = background_img.shape[:2]
    region_h = (effect_region[3] - effect_region[1]) * background_h
    region_w = (effect_region[2] - effect_region[0]) * background_w
    region_aspect_ratio = region_w / region_h

    for det_result in det_results:

        bbox = det_result['bbox']
        instance_w = bbox[2] - bbox[0]
        instance_h = bbox[3] - bbox[1]

        if bbox[-1] > bbox_thr and instance_h > 20 and instance_w > 20:
            aspect_ratio = instance_w / instance_h
            if region_aspect_ratio > aspect_ratio:
                resize_rate = region_h / instance_h
            else:
                resize_rate = region_w / instance_w

            mask = det_result['mask'].astype(np.uint8)
            mask_inst = mask[int(bbox[0]):int(bbox[2]),
                             int(bbox[1]):int(bbox[3])]
            img_inst = img[int(bbox[0]):int(bbox[2]),
                           int(bbox[1]):int(bbox[3])]
            img_inst = cv2.resize(img_inst, (int(
                resize_rate * instance_w), int(resize_rate * instance_h)))
            mask_inst = cv2.resize(
                mask_inst,
                (int(resize_rate * instance_w), int(resize_rate * instance_h)),
                interpolation=cv2.INTER_NEAREST)

            mask_ids = list(np.where(mask_inst == 1))
            mask_ids[0] += int(effect_region[0] * background_w)
            mask_ids[1] += int(effect_region[1] * background_h)

            background_img[tuple(mask_ids)] = img_inst[np.where(
                mask_inst == 1)]

    return background_img
