from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import copy
from scipy.optimize import curve_fit

def absorb_heat(cfg, poses, heatmap_avg):
    poses = torch.tensor(np.array(poses))[0]

    pool = torch.nn.MaxPool2d(5, 1, 2)
    kpt_heatmap = heatmap_avg[:,1:]
    maxm = pool(kpt_heatmap)
    maxm = torch.eq(maxm, kpt_heatmap).float()
    kpt_heatmap = kpt_heatmap * maxm
    _, num_joints, h, w = kpt_heatmap.shape
    kpt_heatmap = kpt_heatmap.view(num_joints, -1)
    val_k, ind = kpt_heatmap.topk(cfg['max_num_people'], dim=1)

    val_k = val_k.cpu()
    ind = ind.cpu()

    x = ind % w
    y = (ind / w).long()
    heats_ind = torch.stack((x, y), dim=2)

    for i in range(num_joints):

        heat_ind = heats_ind[i].float()
        pose_ind = poses[:, i, :-1]
        pose_heat_diff = pose_ind[:, None, :] - heat_ind
        pose_heat_diff.pow_(2)
        pose_heat_diff = pose_heat_diff.sum(2)
        pose_heat_diff.sqrt_()
        keep_ind = torch.argmin(pose_heat_diff, dim=1)

        for p in range(keep_ind.shape[0]):
            if pose_heat_diff[p, keep_ind[p]] < cfg['max_absorb_distance']:
                poses[p, i, :-1] = heat_ind[keep_ind[p]]            

    return [poses.cpu().numpy()]

    
def unnormalized_gaussian2d(data_tuple, A, y0, x0, sigma):
    (y, x) = data_tuple
    g = A * np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g


def fit_gaussian_heatmap(func, heatmap, maxval, init_y, init_x, sigma):
    """
    Find the precise float joint coordinates of coarse int coordinate (init_y, init_x) 
    by fitting guassian on heatmap near (init_y, init_x).

    Args:
        func: gaussian2d function
        heatmap: heatmap near (init_x, init_y)
        maxval: the heatmap value at (init_x, init_y)
        sigma: guassian sigma
    Returns:
        fitted guassian's parameter: center_x, center_y, peak value, sigma
    """
    heatmap_y_length = heatmap.shape[0]
    heatmap_x_length = heatmap.shape[1]
    y = np.linspace(0, heatmap_y_length - 1, heatmap_y_length)
    x = np.linspace(0, heatmap_x_length - 1, heatmap_x_length)
    Y, X = np.meshgrid(y, x)
    x_data = np.vstack((X.ravel(), Y.ravel()))

    init_guess = (maxval, init_y, init_x, sigma)
    popt, _ = curve_fit(func, x_data, heatmap.ravel(),
                        p0=init_guess, maxfev=300)
    return popt[1], popt[2], popt[0], popt[3]


def adjust(cfg, ans, det):
    """
    Use guassian fit to refine final results.
    """
    det = det[:,1:,:,:]
    N = cfg['gaussian_kernel']
    local_hm = np.zeros((N, N))
    for batch_id in range(len(ans)):
        for joint_id in range(ans[0].shape[1]):
            dist_xy = {}
            for people_id in range(ans[0].shape[0]):
                if ans[batch_id][people_id, joint_id, 2] > cfg['adjust_threshold']:
                    y, x = ans[batch_id][people_id, joint_id, 0:2]
                    xx, yy = int(x+0.5), int(y+0.5)
                    dist_index = str([xx, yy])
                    tmp = det[batch_id, joint_id, :, :]

                    if dist_index in dist_xy:
                        ans[batch_id][people_id, joint_id, 1] = dist_xy[dist_index][0]
                        ans[batch_id][people_id, joint_id, 0] = dist_xy[dist_index][1]
                        continue

                    safe_xx = min(max(0, xx), tmp.shape[0]-1)
                    safe_yy = min(max(0, yy), tmp.shape[1]-1)

                    safe_y_lower_bound = max(0, safe_yy - N)
                    safe_y_upper_bound = min(tmp.shape[1]-1, safe_yy + N)

                    safe_x_lower_bound = max(0, safe_xx - N)
                    safe_x_upper_bound = min(tmp.shape[0]-1, safe_xx + N)

                    local_hm = tmp[safe_x_lower_bound:safe_x_upper_bound + 1,
                                    safe_y_lower_bound:safe_y_upper_bound + 1]
                    
                    try:
                        # If neighborhood around (xx, yy) on heatmap is not a guassian distribution, 
                        # optimal parameters can not be found after max iteration
                        # and the curve_fit function in scipy will raise error. 
                        # This keypoint coordinates will not be adjusted
                        # and this parts do not influence the results. 
                        mean_x, mean_y, value, _ = fit_gaussian_heatmap(
                            unnormalized_gaussian2d, \
                            local_hm.cpu().numpy(), tmp[safe_xx][safe_yy].cpu().numpy(), \
                            safe_xx - safe_x_lower_bound, safe_yy - safe_y_lower_bound, sigma=2.0
                        )
                        ans[batch_id][people_id, joint_id, 1] = safe_x_lower_bound + mean_x
                        ans[batch_id][people_id, joint_id, 0] = safe_y_lower_bound + mean_y
                        dist_xy[dist_index] = [safe_x_lower_bound + mean_x, safe_y_lower_bound + mean_y, value]
                    except:
                        continue

    return ans


def match_pose_to_heatmap(cfg, poses, heatmap_avg):
    poses = absorb_heat(cfg, poses, heatmap_avg)
    poses = adjust(cfg, poses, heatmap_avg)

    return poses
