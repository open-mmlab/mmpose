# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zigang Geng (zigang@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle

from mmpose.models.heads import one_stage_rescore_head
from mmpose.models.heads.one_stage_rescore_head import get_pose_net
from pycocotools.cocoeval import COCOeval as COCOEval
from crowdposetools.cocoeval import COCOeval as CrowdposeEval

JOINT_COCO_LINK_1 = [0, 0, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 7, 8, 11, 11, 12, 13, 14]
JOINT_COCO_LINK_2 = [1, 2, 2, 3, 4, 5, 6, 6, 7, 11, 8, 12, 9, 10, 12, 13, 14, 15, 16]

JOINT_CROWDPOSE_LINK_1 = [12, 13, 13, 0, 1, 2, 3, 0, 1, 6, 7,  8,  9, 6, 0]
JOINT_CROWDPOSE_LINK_2 = [13,  0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 7, 1]


# Data process for the RescoreNet
def read_rescore_data(cfg, img_path):
    train_file = cfg.RESCORE.DATA_FILE
    num_joints = cfg.DATASET.NUM_JOINTS
    x_train, y_train = get_joint(train_file, num_joints)
    feature_train = get_feature(x_train, img_path)
    return feature_train, y_train


def get_joint(filename, num_joints):
    obj = pickle.load(open(filename, "rb"))

    posx, posy = [], []
    for i in range(1, len(obj)):
        pose = list(np.concatenate(
            (obj[i][0], obj[i][1]), axis=1).reshape(3*num_joints))
        posx.append(pose)
        if obj[i][2] == 1:
            obj[i][2] = 0
        posy.append(obj[i][2])

    x = np.array(posx)
    y = np.array(posy)

    x = x.reshape((-1, num_joints, 3))
    y = torch.tensor(y.reshape((-1, 1)), dtype=torch.float)
    return x, y


def get_feature(x, dataset):
    joint_abs = x[:, :, :2]
    vis = x[:, :, 2]

    if 'coco' in dataset:
        joint_1, joint_2 = JOINT_COCO_LINK_1, JOINT_COCO_LINK_2
    elif 'crowd' in dataset:
        joint_1, joint_2 = JOINT_CROWDPOSE_LINK_1, JOINT_CROWDPOSE_LINK_2
    else:
        raise ValueError(
            'Please implement flip_index for new dataset: %s.' % dataset)

    #To get the Delta x Delta y
    joint_relate = joint_abs[:, joint_1] - joint_abs[:, joint_2]
    joint_length = ((joint_relate**2)[:, :, 0] +
                    (joint_relate**2)[:, :, 1])**(0.5)

    #To use the torso distance to normalize
    normalize = (joint_length[:, 9]+joint_length[:, 11])/2
    normalize = np.tile(normalize, (len(joint_1), 2, 1)).transpose(2, 0, 1)
    normalize[normalize < 1] = 1

    joint_length = joint_length/normalize[:, :, 0]
    joint_relate = joint_relate/normalize
    joint_relate = joint_relate.reshape((-1, len(joint_1)*2))

    feature = [joint_relate, joint_length, vis]
    feature = np.concatenate(feature, axis=1)
    feature = torch.tensor(feature, dtype=torch.float)
    return feature


# Train and Valid for RescoreNet
def rescore_fit(cfg, model, x_data, y_data):
    loss_fn = nn.MSELoss(reduction='mean')
    train_losses = []

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.RESCORE.LR)

    x_data = Variable(x_data, requires_grad=True)
    y_data = Variable(y_data, requires_grad=True)

    save_final_model_file = cfg.RESCORE.MODEL_FILE
    for epoch in range(cfg.RESCORE.END_EPOCH):
        train_loss = train_core(x_data, y_data, optimizer, model,
                           loss_fn, cfg.RESCORE.BATCHSIZE)
        train_losses.append(train_loss)

        if epoch % 1 == 0:
            print("step:", epoch+1, "train_loss:", train_loss)

    torch.save(model.state_dict(), save_final_model_file)
    return train_losses


def train_core(x_data, y_data, optimizer, model, loss_fn, batchsize):
    datasize = len(x_data)
    loss_sum = 0
    index = np.arange(datasize)
    np.random.shuffle(index)
    for i in range(int(datasize/batchsize)):
        x_temp = x_data[index[i*batchsize:(i+1)*(batchsize)]]
        y_temp = y_data[index[i*batchsize:(i+1)*(batchsize)]]
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_temp)

        loss = loss_fn(y_pred, y_temp)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    return loss_sum/int(datasize/batchsize)


def rescore_valid(cfg, img_metas, temp, ori_scores):
    temp = np.array(temp)

    feature = get_feature(temp, img_metas['img_file'])
    feature = feature.cuda()

    PredictOKSmodel = get_pose_net(
        cfg, feature.shape[1], is_train=False
    )
    pretrained_state_dict = torch.load(cfg['model_file'])
    need_init_state_dict = {}
    for name, m in pretrained_state_dict.items():
        need_init_state_dict[name] = m
    PredictOKSmodel.load_state_dict(need_init_state_dict, strict=False)
    PredictOKSmodel = torch.nn.DataParallel(
        PredictOKSmodel).to(feature)
    PredictOKSmodel.eval()

    # for debug
    # print('feature.device: ', feature.device)
    # print('PredictOKSmodel.device: ', next(PredictOKSmodel.parameters()).device)
    
    scores = PredictOKSmodel(feature)
    scores = scores.cpu().numpy()
    scores[np.isnan(scores)] = 0
    mul_scores = scores*np.array(ori_scores).reshape(scores.shape)
    scores = [np.float(i) for i in list(scores)]
    mul_scores = [np.float(i) for i in list(mul_scores)]
    return mul_scores


# Get Rescore training data for RescoreNet
class COCORescoreEval(COCOEval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        COCOEval.__init__(self, cocoGt, cocoDt, iouType)
        self.summary = [['pose', 'pose_heatval', 'oks']]
    
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        get predicted pose and oks score for single category and image
        change self.summary
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        
        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        gtIg = np.array([g['_ignore'] for g in gt])
        if not len(ious)==0:
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = 0
                m   = -1
                for gind, g in enumerate(gt):
                    #if not iscrowd[gind]:
                    #    continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind,gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou=ious[dind,gind]
                    m=gind
                
                dtkeypoint = np.array(d['keypoints']).reshape((17,3))
                self.summary.append([dtkeypoint[:,:2], dtkeypoint[:,2:], iou])

    def dumpdataset(self, data_file):
        pickle.dump(self.summary, open(data_file, 'wb'))



class CrowdRescoreEval(CrowdposeEval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        CrowdposeEval.__init__(self, cocoGt, cocoDt, iouType)
        self.summary = [['pose', 'pose_heatval', 'oks']]
    
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        get predicted pose and oks score for single category and image
        change self.summary
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None
        
        for g in gt:
            tmp_area = g['bbox'][2] * g['bbox'][3] * 0.53
            if g['ignore'] or (tmp_area < aRng[0] or tmp_area > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(
            self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        gtIg = np.array([g['_ignore'] for g in gt])
        if not len(ious)==0:
            for dind, d in enumerate(dt):
                # information about best match so far (m=-1 -> unmatched)
                iou = 0
                m   = -1
                for gind, g in enumerate(gt):
                    #if not iscrowd[gind]:
                    #    continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        break
                    # continue to next gt unless better match made
                    if ious[dind,gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou=ious[dind,gind]
                    m=gind
                
                dtkeypoint = np.array(d['keypoints']).reshape((14,3))
                self.summary.append([dtkeypoint[:,:2], dtkeypoint[:,2:], iou])

    def dumpdataset(self, data_file):
        pickle.dump(self.summary, open(data_file, 'wb'))




