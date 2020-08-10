import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sigmas = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
    .87, .89, .89
])


def candidate_reselect(bboxes, pose_preds, num_joints, img, box_scores,
                       in_vis_thre):
    """get final result with group and match.

    Note:
        num_person: N
    Args:
        bboxes (torch.Tensor[N,4]): bbox(x1,y1,x2,y2).
        pose_preds (dict): candidate keypoints information.
            pose_preds[i][j] (list): i-th person j-th joint 's canditate
                    keypoints information.
        num_joints (int): number of keypoints.
        img (int): image_id.
        box_scores (torch.Tensor[N,1]): bbox score.
        in_vis_thre (float): threshold.
    Returns:
        final_result (list): predicted in the image.
    """

    # Group same keypointns together
    kp_groups = _grouping(bboxes, pose_preds, num_joints, box_scores)

    # Generate Matrix
    human_num = len(pose_preds.keys())
    costMatrix = []

    for k in range(num_joints):
        kp_group = kp_groups[k]
        joint_num = len(kp_group.keys())
        costMatrix.append(np.zeros((human_num, joint_num)))

    for n, person in pose_preds.items():
        h_id = n
        assert 0 <= h_id < human_num
        for k in range(num_joints):
            g_id = person['group_id'][k]
            if g_id is not None:

                g_id = int(g_id) - 1
                _, _, score = person[k][0]
                h_score = person['human_score']

                if score < 0.05:
                    costMatrix[k][h_id][g_id] = 0
                else:
                    costMatrix[k][h_id][g_id] = -(h_score * score)

    pose_preds = _matching(pose_preds, costMatrix, num_joints, kp_groups)

    # To JSON
    final_result = []

    for n, person in pose_preds.items():

        final_pose = torch.zeros(num_joints, 3)

        mean_score = 0
        vaild_num = 0
        for k in range(num_joints):
            assert len(person[k]) > 0
            x, y, s = person[k][0]
            final_pose[k][0] = x.item()
            final_pose[k][1] = y.item()
            final_pose[k][2] = s.item()
            if s.item() > in_vis_thre:
                mean_score += s.item()
                vaild_num += 1

        if torch.max(final_pose[:, 2]).item() < 0.05:
            continue
        if vaild_num != 0:
            mean_score = mean_score / vaild_num

        x1, y1, x2, y2 = person['bbox']
        final_result.append({
            'keypoints':
            final_pose.numpy(),
            'center':
            np.array([((x1 + x2) / 2).item(), ((y1 + y2) / 2).item()]),
            'scale':
            np.array([(x2 - x1).item(), (y2 - y1).item()]),
            'area': (x2 - x1).item() * (y2 - y1).item(),
            'score':
            person['bbox_score'].item() * mean_score,
            'image_id':
            img
        })

    return final_result


def _grouping(bboxes, pose_preds, num_joints, box_scores):
    """remove the joints that are repeated with group.

    Note:
        num_person: N
        num_keypoints: K
    Args:
        bboxes (torch.Tensor[n,4]): bbox(x1,y1,x2,y2).
        pose_preds (dict): candidate keypoints information.
            pose_preds[i][j] (list): i-th person j-th joint 's canditate
                    keypoints information.
        num_joints (int): num_keypoints.
        box_scores (torch.Tensor[n,1]): bbox score.
    Returns:
        kp_groups (dict[k,dict]): information after grouping.
    """

    kp_groups = {}
    for k in range(num_joints):
        kp_groups[k] = {}

    ids = np.zeros(num_joints)

    for n, person in pose_preds.items():
        pose_preds[n]['bbox'] = bboxes[n]
        pose_preds[n]['bbox_score'] = box_scores[n]
        pose_preds[n]['group_id'] = {}
        s = 0
        for k in range(num_joints):
            pose_preds[n]['group_id'][k] = None
            pose_preds[n][k] = np.array(pose_preds[n][k])
            assert len(pose_preds[n][k]) > 0
            s += pose_preds[n][k][0][-1]

        s /= num_joints

        pose_preds[n]['human_score'] = s

        for k in range(num_joints):
            latest_id = ids[k]
            kp_group = kp_groups[k]

            assert len(person[k]) > 0
            x0, y0, s0 = person[k][0]
            if s0 < 0.05:
                continue
            for g_id, g in kp_group.items():

                x_c, y_c = kp_group[g_id]['group_center']
                '''
                Get Average Box Size
                '''
                group_area = kp_group[g_id]['group_area']
                group_area = group_area[0] * group_area[1] / (group_area[2]**2)
                '''
                Groupingn Criterion
                '''
                # Joint Group
                # x0, y0, s0 = person[k][0]
                dist = np.sqrt(((x_c - x0)**2 + (y_c - y0)**2) / group_area)

                if dist <= 0.1 * sigmas[k]:  # Small Distance
                    if s0 > 0.3:
                        kp_group[g_id]['kp_list'][0] += x0 * s0
                        kp_group[g_id]['kp_list'][1] += y0 * s0
                        kp_group[g_id]['kp_list'][2] += s0

                        kp_group[g_id]['group_area'][0] += (
                            person['bbox'][2] -
                            person['bbox'][0]) * person['human_score']
                        kp_group[g_id]['group_area'][1] += (
                            person['bbox'][3] -
                            person['bbox'][1]) * person['human_score']
                        kp_group[g_id]['group_area'][2] += person[
                            'human_score']

                        x_c = kp_group[g_id]['kp_list'][0] / kp_group[g_id][
                            'kp_list'][2]
                        y_c = kp_group[g_id]['kp_list'][1] / kp_group[g_id][
                            'kp_list'][2]
                        kp_group[g_id]['group_center'] = (x_c, y_c)

                    pose_preds[n]['group_id'][k] = g_id

                    break
            else:
                # A new keypoint group
                latest_id += 1
                kp_group[latest_id] = {
                    'kp_list': None,
                    'group_center': person[k][0].copy()[:2],
                    'group_area': None
                }

                x, y, s = person[k][0]
                kp_group[latest_id]['kp_list'] = np.array((x * s, y * s, s))

                # Ref Area
                ref_width = person['bbox'][2] - person['bbox'][0]
                ref_height = person['bbox'][3] - person['bbox'][1]
                ref_score = person['human_score']
                kp_group[latest_id]['group_area'] = np.array(
                    (ref_width * ref_score, ref_height * ref_score, ref_score))

                pose_preds[n]['group_id'][k] = latest_id
                ids[k] = latest_id

    return kp_groups


def _matching(pose_preds, matrix, num_joints, kp_groups):
    """use hungarian algorithm to match person and keypoints.

    Note:
        num_person: N
        num_keypoints: K
    Args:
        pose_preds (dict)
        matric (list):cost matric.
        num_joints (int): num_keypoint.
        kp_groups (dict):group information.
    Returns:
        pose_preds (dict):final result.
    """
    index = []
    for k in range(num_joints):
        human_ind, joint_ind = linear_sum_assignment(matrix[k])

        index.append(list(zip(human_ind, joint_ind)))

    for n, person in pose_preds.items():
        for k in range(num_joints):
            g_id = person['group_id'][k]
            if g_id is not None:
                g_id = int(g_id) - 1
                h_id = n

                x, y, s = pose_preds[n][k][0]
                if ((h_id, g_id)
                        not in index[k]) and len(pose_preds[n][k]) > 1:
                    pose_preds[n][k] = np.delete(pose_preds[n][k], 0, 0)
                elif ((h_id, g_id) not in index[k]) and len(person[k]) == 1:
                    x, y, _ = pose_preds[n][k][0]
                    pose_preds[n][k][0] = (x, y, 1e-5)
                    pass
                elif ((h_id, g_id) in index[k]):
                    x, y = kp_groups[k][g_id + 1]['group_center']
                    s = pose_preds[n][k][0][2]
                    pose_preds[n][k][0] = (x, y, s)

    return pose_preds


def convert_crowd(kpt):
    """convert the corresponding information for subsequent processing.

    Note:
        num_person: N
        num_keypoints: K
    Args:
        kpt (list):
            keypoints(np.ndarray[k,5,3]): coordinate, socre of candidate
                    keypoints.
            center(np.ndarray[0:2]): Center of bbox.
            scale(np.ndarray[0:2]): Scale of bbox.
            area(float): Area of bbox.
            score(float): Score of bbox.
            image(int): image_id.
    Returns:
        boxes (torch.Tensor[n,4]): bbox(x1,y1,x2,y2).
        pose_preds (dict): candidate keypoints information.
            pose_preds[i][j] (list): i-th person j-th joint 's canditate
                    keypoints information.
        box_scores (torch.Tensor[n,1]): bbox score.
    """
    preds = {}
    for i in range(len(kpt)):
        preds[i] = {}
        box = torch.zeros(4)
        box_score = torch.zeros(1)
        box[0] = kpt[i]['center'][0] - kpt[i]['scale'][0] * 0.5
        box[1] = kpt[i]['center'][1] - kpt[i]['scale'][1] * 0.5
        box[2] = kpt[i]['center'][0] + kpt[i]['scale'][0] * 0.5
        box[3] = kpt[i]['center'][1] + kpt[i]['scale'][1] * 0.5
        box_score[0] = kpt[i]['score']
        if i == 0:
            boxes = box.unsqueeze(0)
            box_scores = box_score.unsqueeze(0)
        else:
            boxes = torch.cat((boxes, box.unsqueeze(0)), dim=0)
            box_scores = torch.cat((box_scores, box_score.unsqueeze(0)), dim=0)
        for k in range(len(kpt[i]['keypoints'])):
            preds[i][k] = []
            for j in range(len(kpt[i]['keypoints'][k])):
                if (kpt[i]['keypoints'][k][j][2] < 0.1) and (len(preds[i][k]) >
                                                             0):
                    continue
                preds[i][k].append(kpt[i]['keypoints'][k][j])
    return boxes, preds, box_scores
