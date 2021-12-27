# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np


def screen_matting(img, color_low=None, color_high=None, color=None):
    """Screen Matting.

    Args:
        img (np.ndarray): Image data.
        color_low (tuple): Lower limit (b, g, r).
        color_high (tuple): Higher limit (b, g, r).
        color (str): Support colors include:

            - 'green' or 'g'
            - 'blue' or 'b'
            - 'black' or 'k'
            - 'white' or 'w'
    """

    if color_high is None or color_low is None:
        if color is not None:
            if color.lower() == 'g' or color.lower() == 'green':
                color_low = (0, 200, 0)
                color_high = (60, 255, 60)
            elif color.lower() == 'b' or color.lower() == 'blue':
                color_low = (230, 0, 0)
                color_high = (255, 40, 40)
            elif color.lower() == 'k' or color.lower() == 'black':
                color_low = (0, 0, 0)
                color_high = (40, 40, 40)
            elif color.lower() == 'w' or color.lower() == 'white':
                color_low = (230, 230, 230)
                color_high = (255, 255, 255)
            else:
                NotImplementedError(f'Not supported color: {color}.')
        else:
            ValueError('color or color_high | color_low should be given.')

    mask = cv2.inRange(img, np.array(color_low), np.array(color_high)) == 0

    return mask.astype(np.uint8)


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


def expand_and_clamp(box, im_shape, s=1.25):
    """Expand the bbox and clip it to fit the image shape.

    Args:
        box (list): x1, y1, x2, y2
        im_shape (ndarray): image shape (h, w, c)
        s (float): expand ratio

    Returns:
        list: x1, y1, x2, y2
    """

    x1, y1, x2, y2 = box[:4]
    w = x2 - x1
    h = y2 - y1
    deta_w = w * (s - 1) / 2
    deta_h = h * (s - 1) / 2

    x1, y1, x2, y2 = x1 - deta_w, y1 - deta_h, x2 + deta_w, y2 + deta_h

    img_h, img_w = im_shape[:2]

    x1 = min(max(0, int(x1)), img_w - 1)
    y1 = min(max(0, int(y1)), img_h - 1)
    x2 = min(max(0, int(x2)), img_w - 1)
    y2 = min(max(0, int(y2)), img_h - 1)

    return [x1, y1, x2, y2]


def apply_saiyan_effect(img,
                        pose_results,
                        saiyan_img,
                        light_frame,
                        face_indices,
                        bbox_thr=0.3,
                        kpt_thr=0.5):
    """Apply saiyan hair effect.

    Args:
        img (np.ndarray): Image data.
        pose_results (list[dict]): The pose estimation results containing:
            - "keypoints" ([K,3]): keypoint detection result in [x, y, score]
        saiyan_img (np.ndarray): Saiyan image with transparent background.
        light_frame (np.ndarray): Light image with green screen.
        face_indices (int): Keypoint index of the face
        kpt_thr (float): The score threshold of required keypoints.
    """
    img = img.copy()
    im_shape = img.shape
    # Apply lightning effects.
    light_mask = screen_matting(light_frame, color='green')

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
        bbox = pose['bbox']

        if bbox[-1] < bbox_thr:
            continue

        mask_inst = pose['mask']
        # cache
        fg = img[np.where(mask_inst)]

        bbox = expand_and_clamp(bbox[:4], im_shape, s=1.4)
        # Apply light effects between fg and bg
        img = copy_and_paste(
            light_frame,
            img,
            light_mask,
            effect_region=(bbox[0] / im_shape[1], bbox[1] / im_shape[0],
                           bbox[2] / im_shape[1], bbox[3] / im_shape[0]))
        # pop
        img[np.where(mask_inst)] = fg

        # Apply Saiyan hair effects
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
        mask_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask_patch = (mask_patch > 1).astype(np.uint8)
        img = cv2.copyTo(patch, mask_patch, img)

    return img


def find_connected_components(mask):
    """Find connected components and sort with areas.

    Args:
        mask (ndarray): instance segmentation result.

    Returns:
        ndarray (N, 5): Each item contains (x, y, w, h, area).
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    stats = stats[stats[:, 4].argsort()]
    return stats


def find_bbox(mask):
    """Find the bounding box for the mask.

    Args:
        mask (ndarray): Mask.

    Returns:
        list(4, ): Returned box (x1, y1, x2, y2).
    """
    mask_shape = mask.shape
    if len(mask_shape) == 3:
        assert mask_shape[-1] == 1, 'the channel of the mask should be 1.'
    elif len(mask_shape) == 2:
        pass
    else:
        NotImplementedError()

    h, w = mask_shape[:2]
    mask_w = mask.sum(0)
    mask_h = mask.sum(1)

    left = 0
    right = w - 1
    up = 0
    down = h - 1

    for i in range(w):
        if mask_w[i] > 0:
            break
        left += 1

    for i in range(w - 1, left, -1):
        if mask_w[i] > 0:
            break
        right -= 1

    for i in range(h):
        if mask_h[i] > 0:
            break
        up += 1

    for i in range(h - 1, up, -1):
        if mask_h[i] > 0:
            break
        down -= 1

    return [left, up, right, down]


def copy_and_paste(img,
                   background_img,
                   mask,
                   bbox=None,
                   effect_region=(0.2, 0.2, 0.8, 0.8),
                   min_size=(20, 20)):
    """Copy the image region and paste to the background.

    Args:
        img (np.ndarray): Image data.
        background_img (np.ndarray): Background image data.
        mask (ndarray): instance segmentation result.
        bbox (ndarray): instance bbox, (x1, y1, x2, y2).
        effect_region (tuple(4, )): The region to apply mask, the coordinates
            are normalized (x1, y1, x2, y2).
    """
    background_img = background_img.copy()
    background_h, background_w = background_img.shape[:2]
    region_h = (effect_region[3] - effect_region[1]) * background_h
    region_w = (effect_region[2] - effect_region[0]) * background_w
    region_aspect_ratio = region_w / region_h

    if bbox is None:
        bbox = find_bbox(mask)
    instance_w = bbox[2] - bbox[0]
    instance_h = bbox[3] - bbox[1]

    if instance_w > min_size[0] and instance_h > min_size[1]:
        aspect_ratio = instance_w / instance_h
        if region_aspect_ratio > aspect_ratio:
            resize_rate = region_h / instance_h
        else:
            resize_rate = region_w / instance_w

        mask_inst = mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_inst = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_inst = cv2.resize(img_inst, (int(
            resize_rate * instance_w), int(resize_rate * instance_h)))
        mask_inst = cv2.resize(
            mask_inst,
            (int(resize_rate * instance_w), int(resize_rate * instance_h)),
            interpolation=cv2.INTER_NEAREST)

        mask_ids = list(np.where(mask_inst == 1))
        mask_ids[1] += int(effect_region[0] * background_w)
        mask_ids[0] += int(effect_region[1] * background_h)

        background_img[tuple(mask_ids)] = img_inst[np.where(mask_inst == 1)]

    return background_img


def apply_background_effect(img,
                            det_results,
                            background_img,
                            effect_region=(0.2, 0.2, 0.8, 0.8)):
    """Change background.

    Args:
        img (np.ndarray): Image data.
        det_results (list[dict]): The detection results containing:

            - "cls_id" (int): Class index.
            - "label" (str): Class label (e.g. 'person').
            - "bbox" (ndarray:(5, )): bounding box result [x, y, w, h, score].
            - "mask" (ndarray:(w, h)): instance segmentation result.
        background_img (np.ndarray): Background image.
        effect_region (tuple(4, )): The region to apply mask, the coordinates
            are normalized (x1, y1, x2, y2).
    """
    # Choose the one with the highest score.
    det_result = det_results[0]
    bbox = det_result['bbox']
    mask = det_result['mask'].astype(np.uint8)
    img = copy_and_paste(img, background_img, mask, bbox, effect_region)
    return img
