# ------------------------------------------------------------------------------
# Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import numpy as np
import torch


def nms(dets, thr):
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets: [[x1, y1, x2, y2, score]].
        thr: Retain overlap < thr.

    Returns:
         list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    if sigmas is None:
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        inds = np.where(oks_ovr <= thr)[0]
        order = order[inds + 1]

    keep = np.array(keep)

    return keep


def _rescore(overlap, scores, thr, type='gaussian'):
    """Rescoring mechanism gaussian or linear.

    Args:
        overlap: calculated ious
        scores: target scores.
        thr: retain oks overlap < thr.
        type: 'gaussian' or 'linear'

    Returns:
        np.ndarray: indexes to keep
    """
    assert len(overlap) == len(scores)
    assert type in ['gaussian', 'linear']

    if type == 'linear':
        inds = np.where(overlap >= thr)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(-overlap**2 / thr)

    return scores


def soft_oks_nms(kpts_db,
                 thr,
                 max_dets=20,
                 sigmas=None,
                 vis_thr=None,
                 score_per_joint=False):
    """Soft OKS NMS implementations.

    Args:
        kpts_db
        thr: retain oks overlap < thr.
        max_dets: max number of detections to keep.
        sigmas: Keypoint labelling uncertainty.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while len(order) > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        order = order[1:]
        scores = _rescore(oks_ovr, scores[1:], thr)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep

def get_heat_value(pose_coord, heatmap):
    _, h, w = heatmap.shape
    heatmap_nocenter = heatmap[1:].flatten(1,2).transpose(0,1)

    y_b = torch.clamp(torch.floor(pose_coord[:,:,1]), 0, h-1).long()
    x_l = torch.clamp(torch.floor(pose_coord[:,:,0]), 0, w-1).long()
    heatval = torch.gather(heatmap_nocenter, 0, y_b * w + x_l).unsqueeze(-1)
    return heatval


def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def nms_core(cfg, pose_coord, heat_score):
    num_people, num_joints, _ = pose_coord.shape
    pose_area = cal_area_2_torch(pose_coord)[:,None].repeat(1,num_people*num_joints)
    pose_area = pose_area.reshape(num_people,num_people,num_joints)
    
    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = cfg['nms_thre'] * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > cfg['nms_num_thre']

    ignored_pose_inds = []
    keep_pose_inds = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        keep_scores = heat_score[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        ignored_pose_inds += list(set(keep_inds)-set(ignored_pose_inds))

    return keep_pose_inds

def pose_nms(cfg, img_metas, heatmap_avg, poses):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    scale1_index = sorted(img_metas['test_scale_factor'], reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:,:,2].max() if pose_norm.shape[0] else 1

    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:,:,2].max() if pose.shape[0] else 1
            pose[:,:,2] = pose[:,:,2]/max_score_scale*max_score*cfg['decrease']

    pose_score = torch.cat([pose[:,:,2:] for pose in poses], dim=0)
    pose_coord = torch.cat([pose[:,:,:2] for pose in poses], dim=0)

    if pose_coord.shape[0] == 0:
        return [], []

    num_people, num_joints, _ = pose_coord.shape
    heatval = get_heat_value(pose_coord, heatmap_avg[0])
    heat_score = (torch.sum(heatval, dim=1)/num_joints)[:,0]

    pose_score = pose_score*heatval
    poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=2)

    keep_pose_inds = nms_core(cfg, pose_coord, heat_score)
    poses = poses[keep_pose_inds]
    heat_score = heat_score[keep_pose_inds]
    
    if len(keep_pose_inds) > cfg['max_num_people']:
        heat_score, topk_inds = torch.topk(heat_score, cfg['max_num_people'])
        poses = poses[topk_inds]

    poses = [poses.numpy()]
    scores = [i[:, 2].mean() for i in poses[0]]
    
    return poses, scores
