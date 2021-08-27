# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmpose.core.post_processing import oks_nms
from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet
from mmpose.utils.hooks import OutputHook

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location=device)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def _xyxy2xywh(bbox_xyxy):
    """Transform the bbox format from x1y1x2y2 to xywh.

    Args:
        bbox_xyxy (np.ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5). (left, top, right, bottom, [score])

    Returns:
        np.ndarray: Bounding boxes (with scores),
          shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    """
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
    bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1

    return bbox_xywh


def _xywh2xyxy(bbox_xywh):
    """Transform the bbox format from xywh to x1y1x2y2.

    Args:
        bbox_xywh (ndarray): Bounding boxes (with scores),
            shaped (n, 4) or (n, 5). (left, top, width, height, [score])
    Returns:
        np.ndarray: Bounding boxes (with scores), shaped (n, 4) or
          (n, 5). (left, top, right, bottom, [score])
    """
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[:, 2] = bbox_xyxy[:, 2] + bbox_xyxy[:, 0] - 1
    bbox_xyxy[:, 3] = bbox_xyxy[:, 3] + bbox_xyxy[:, 1] - 1

    return bbox_xyxy


def _box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the img_or_path.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
        else:
            raise TypeError('"img_or_path" must be a numpy array or a str or '
                            'a pathlib.Path object')

        results['img'] = img
        return results


def _inference_single_pose_model(model,
                                 img_or_path,
                                 bboxes,
                                 dataset,
                                 return_heatmap=False):
    """Inference human bounding boxes.

    num_bboxes: N
    num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        dataset (str): Dataset name.
        outputs (list[str] | tuple[str]): Names of layers whose output is
            to be returned, default: None

    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    flip_pairs = None
    if dataset in ('TopDownCocoDataset', 'TopDownOCHumanDataset',
                   'AnimalMacaqueDataset'):
        flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                      [13, 14], [15, 16]]
    elif dataset == 'TopDownCocoWholeBodyDataset':
        body = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                [15, 16]]
        foot = [[17, 20], [18, 21], [19, 22]]

        face = [[23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34],
                [29, 33], [30, 32], [40, 49], [41, 48], [42, 47], [43, 46],
                [44, 45], [54, 58], [55, 57], [59, 68], [60, 67], [61, 66],
                [62, 65], [63, 70], [64, 69], [71, 77], [72, 76], [73, 75],
                [78, 82], [79, 81], [83, 87], [84, 86], [88, 90]]

        hand = [[91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
                [96, 117], [97, 118], [98, 119], [99, 120], [100, 121],
                [101, 122], [102, 123], [103, 124], [104, 125], [105, 126],
                [106, 127], [107, 128], [108, 129], [109, 130], [110, 131],
                [111, 132]]
        flip_pairs = body + foot + face + hand
    elif dataset == 'TopDownAicDataset':
        flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]
    elif dataset == 'TopDownMpiiDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    elif dataset == 'TopDownMpiiTrbDataset':
        flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7],
                      [8, 9], [10, 11], [14, 15], [16, 22], [28, 34], [17, 23],
                      [29, 35], [18, 24], [30, 36], [19, 25], [31,
                                                               37], [20, 26],
                      [32, 38], [21, 27], [33, 39]]
    elif dataset == 'TopDownH36MDataset':
        flip_pairs = [[1, 4], [2, 5], [3, 6], [11, 14], [12, 15], [13, 16]]
    elif dataset in ('OneHand10KDataset', 'FreiHandDataset', 'PanopticDataset',
                     'InterHand2DDataset'):
        flip_pairs = []
    elif dataset in 'Face300WDataset':
        flip_pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11],
                      [6, 10], [7, 9], [17, 26], [18, 25], [19, 24], [20, 23],
                      [21, 22], [31, 35], [32, 34], [36, 45], [37,
                                                               44], [38, 43],
                      [39, 42], [40, 47], [41, 46], [48, 54], [49,
                                                               53], [50, 52],
                      [61, 63], [60, 64], [67, 65], [58, 56], [59, 55]]

    elif dataset in 'FaceAFLWDataset':
        flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [7, 10], [8, 9],
                      [12, 14], [15, 17]]

    elif dataset in 'FaceCOFWDataset':
        flip_pairs = [[0, 1], [4, 6], [2, 3], [5, 7], [8, 9], [10, 11],
                      [12, 14], [16, 17], [13, 15], [18, 19], [22, 23]]

    elif dataset in 'FaceWFLWDataset':
        flip_pairs = [[0, 32], [1, 31], [2, 30], [3, 29], [4, 28], [5, 27],
                      [6, 26], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
                      [12, 20], [13, 19], [14, 18], [15, 17], [33,
                                                               46], [34, 45],
                      [35, 44], [36, 43], [37, 42], [38, 50], [39,
                                                               49], [40, 48],
                      [41, 47], [60, 72], [61, 71], [62, 70], [63,
                                                               69], [64, 68],
                      [65, 75], [66, 74], [67, 73], [55, 59], [56,
                                                               58], [76, 82],
                      [77, 81], [78, 80], [87, 83], [86, 84], [88, 92],
                      [89, 91], [95, 93], [96, 97]]

    elif dataset in 'AnimalFlyDataset':
        flip_pairs = [[1, 2], [6, 18], [7, 19], [8, 20], [9, 21], [10, 22],
                      [11, 23], [12, 24], [13, 25], [14, 26], [15, 27],
                      [16, 28], [17, 29], [30, 31]]
    elif dataset in 'AnimalHorse10Dataset':
        flip_pairs = []

    elif dataset in 'AnimalLocustDataset':
        flip_pairs = [[5, 20], [6, 21], [7, 22], [8, 23], [9, 24], [10, 25],
                      [11, 26], [12, 27], [13, 28], [14, 29], [15, 30],
                      [16, 31], [17, 32], [18, 33], [19, 34]]

    elif dataset in 'AnimalZebraDataset':
        flip_pairs = [[3, 4], [5, 6]]

    elif dataset in 'AnimalPoseDataset':
        flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15],
                      [16, 17], [18, 19]]
    else:
        raise NotImplementedError()

    batch_data = []
    for bbox in bboxes:
        center, scale = _box2cs(cfg, bbox)

        # prepare data
        data = {
            'img_or_path':
            img_or_path,
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            bbox[4] if len(bbox) == 5 else 1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }
        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)
    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_heatmap=return_heatmap)

    return result['preds'], result['output_heatmap']


def inference_top_down_pose_model(model,
                                  img_or_path,
                                  person_results,
                                  bbox_thr=None,
                                  format='xywh',
                                  dataset='TopDownCocoDataset',
                                  return_heatmap=False,
                                  outputs=None):
    """Inference a single image with a list of person bounding boxes.

    num_people: P
    num_keypoints: K
    bbox height: H
    bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        person_results (List(dict)): the item in the dict may contain
            'bbox' and/or 'track_id'.
            'bbox' (4, ) or (5, ): The person bounding box, which contains
            4 box coordinates (and score).
            'track_id' (int): The unique id for each human instance.
        bbox_thr: Threshold for bounding boxes. Only bboxes with higher scores
            will be fed into the pose detector. If bbox_thr is None, ignore it.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned, default: None

    Returns:
        list[dict]: The bbox & pose info,
            Each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score])
            and the pose (ndarray[Kx3]): x, y, score
        list[dict[np.ndarray[N, K, H, W] | torch.tensor[N, K, H, W]]]:
            Output feature maps from layers specified in `outputs`.
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []
    returned_outputs = []

    if len(person_results) == 0:
        return pose_results, returned_outputs

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])

    # Select bboxes by score threshold
    if bbox_thr is not None:
        assert bboxes.shape[1] == 5
        valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
        bboxes = bboxes[valid_idx]
        person_results = [person_results[i] for i in valid_idx]

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = _xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = _xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # poses is results['pred'] # N x 17x 3
        poses, heatmap = _inference_single_pose_model(
            model,
            img_or_path,
            bboxes_xywh,
            dataset,
            return_heatmap=return_heatmap)

        if return_heatmap:
            h.layer_outputs['heatmap'] = heatmap

        returned_outputs.append(h.layer_outputs)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results, returned_outputs


def inference_bottom_up_pose_model(model,
                                   img_or_path,
                                   pose_nms_thr=0.9,
                                   return_heatmap=False,
                                   outputs=None):
    """Inference a single image.

    num_people: P
    num_keypoints: K
    bbox height: H
    bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        pose_nms_thr (float): retain oks overlap < pose_nms_thr, default: 0.9.
        return_heatmap (bool) : Flag to return heatmap, default: False.
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned, default: None.

    Returns:
        list[ndarray]: The predicted pose info.
            The length of the list is the number of people (P).
            Each item in the list is a ndarray, containing each person's
            pose (ndarray[Kx3]): x, y, score.
        list[dict[np.ndarray[N, K, H, W] | torch.tensor[N, K, H, W]]]:
            Output feature maps from layers specified in `outputs`.
            Includes 'heatmap' if `return_heatmap` is True.
    """
    pose_results = []
    returned_outputs = []

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)
                     ] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    # prepare data
    data = {
        'img_or_path': img_or_path,
        'dataset': 'coco',
        'ann_info': {
            'image_size':
            cfg.data_cfg['image_size'],
            'num_joints':
            cfg.data_cfg['num_joints'],
            'flip_index':
            [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        }
    }

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'].data[0]

    with OutputHook(model, outputs=outputs, as_tensor=False) as h:
        # forward the model
        with torch.no_grad():
            result = model(
                img=data['img'],
                img_metas=data['img_metas'],
                return_loss=False,
                return_heatmap=return_heatmap)

        if return_heatmap:
            h.layer_outputs['heatmap'] = result['output_heatmap']

        returned_outputs.append(h.layer_outputs)

        for idx, pred in enumerate(result['preds']):
            area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (
                np.max(pred[:, 1]) - np.min(pred[:, 1]))
            pose_results.append({
                'keypoints': pred[:, :3],
                'score': result['scores'][idx],
                'area': area,
            })

        # pose nms
        keep = oks_nms(pose_results, pose_nms_thr, sigmas=None)
        pose_results = [pose_results[_keep] for _keep in keep]

    return pose_results, returned_outputs


def vis_pose_result(model,
                    img,
                    result,
                    radius=4,
                    thickness=1,
                    kpt_score_thr=0.3,
                    bbox_color='green',
                    dataset='TopDownCocoDataset',
                    show=False,
                    out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        radius (int): Radius of circles.
        thickness (int): Thickness of lines.
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]): Default None.
        show (bool):  Whether to show the image. Default True.
        out_file (str|None): The filename of the output visualization image.
    """
    if hasattr(model, 'module'):
        model = model.module

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    if dataset in ('TopDownCocoDataset', 'BottomUpCocoDataset',
                   'TopDownOCHumanDataset', 'AnimalMacaqueDataset'):
        # show the results
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        pose_limb_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
        ]]

    elif dataset == 'TopDownCocoWholeBodyDataset':
        # show the results
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [16, 18],
                    [16, 19], [16, 20], [17, 21], [17, 22], [17, 23], [92, 93],
                    [93, 94], [94, 95], [95, 96], [92, 97], [97, 98], [98, 99],
                    [99, 100], [92, 101], [101, 102], [102, 103], [103, 104],
                    [92, 105], [105, 106], [106, 107], [107, 108], [92, 109],
                    [109, 110], [110, 111], [111, 112], [113, 114], [114, 115],
                    [115, 116], [116, 117], [113, 118], [118, 119], [119, 120],
                    [120, 121], [113, 122], [122, 123], [123, 124], [124, 125],
                    [113, 126], [126, 127], [127, 128], [128, 129], [113, 130],
                    [130, 131], [131, 132], [132, 133]]

        pose_limb_color = palette[
            [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16] +
            [16, 16, 16, 16, 16, 16] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ] + [
                0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
                16
            ]]
        pose_kpt_color = palette[
            [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
            [0, 0, 0, 0, 0, 0] + [19] * (68 + 42)]

    elif dataset == 'TopDownAicDataset':
        skeleton = [[3, 2], [2, 1], [1, 14], [14, 4], [4, 5], [5, 6], [9, 8],
                    [8, 7], [7, 10], [10, 11], [11, 12], [13, 14], [1, 7],
                    [4, 10]]

        pose_limb_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 0, 7, 7
        ]]
        pose_kpt_color = palette[[
            9, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 0, 0
        ]]

    elif dataset == 'TopDownMpiiDataset':
        skeleton = [[1, 2], [2, 3], [3, 7], [7, 4], [4, 5], [5, 6], [7, 8],
                    [8, 9], [9, 10], [9, 13], [13, 12], [12, 11], [9, 14],
                    [14, 15], [15, 16]]

        pose_limb_color = palette[[
            16, 16, 16, 16, 16, 16, 7, 7, 0, 9, 9, 9, 9, 9, 9
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 16, 7, 7, 0, 0, 9, 9, 9, 9, 9, 9
        ]]

    elif dataset == 'TopDownMpiiTrbDataset':
        skeleton = [[13, 14], [14, 1], [14, 2], [1, 3], [2, 4], [3, 5], [4, 6],
                    [1, 7], [2, 8], [7, 8], [7, 9], [8, 10], [9, 11], [10, 12],
                    [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26],
                    [27, 28], [29, 30], [31, 32], [33, 34], [35, 36], [37, 38],
                    [39, 40]]

        pose_limb_color = palette[[16] * 14 + [19] * 13]
        pose_kpt_color = palette[[16] * 14 + [0] * 26]

    elif dataset == 'TopDownH36MDataset':
        skeleton = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8],
                    [8, 9], [9, 10], [10, 11], [9, 12], [12, 13], [13, 14],
                    [9, 15], [15, 16], [16, 17]]

        pose_kpt_color = palette[[
            9, 0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
        ]]
        pose_limb_color = palette[[
            0, 0, 0, 16, 16, 16, 9, 9, 9, 9, 16, 16, 16, 0, 0, 0
        ]]

    elif dataset in ('OneHand10KDataset', 'FreiHandDataset',
                     'PanopticDataset'):
        skeleton = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8],
                    [8, 9], [1, 10], [10, 11], [11, 12], [12, 13], [1, 14],
                    [14, 15], [15, 16], [16, 17], [1, 18], [18, 19], [19, 20],
                    [20, 21]]

        pose_limb_color = palette[[
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
            16
        ]]

    elif dataset == 'InterHand2DDataset':
        skeleton = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10],
                    [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18],
                    [18, 19], [19, 20], [4, 21], [8, 21], [12, 21], [16, 21],
                    [20, 21]]

        pose_limb_color = palette[[
            0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12, 16
        ]]
        pose_kpt_color = palette[[
            0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16, 16,
            0
        ]]

    elif dataset == 'Face300WDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 68]
        kpt_score_thr = 0

    elif dataset == 'FaceAFLWDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 19]
        kpt_score_thr = 0

    elif dataset == 'FaceCOFWDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 29]
        kpt_score_thr = 0

    elif dataset == 'FaceWFLWDataset':
        # show the results
        skeleton = []

        pose_limb_color = palette[[]]
        pose_kpt_color = palette[[19] * 98]
        kpt_score_thr = 0

    elif dataset == 'AnimalHorse10Dataset':
        skeleton = [[1, 2], [2, 13], [13, 17], [17, 22], [22, 18], [18, 12],
                    [12, 11], [11, 9], [9, 10], [10, 13], [3, 4], [4, 5],
                    [6, 7], [7, 8], [14, 15], [15, 16], [19, 20], [20, 21]]

        pose_limb_color = palette[[4] * 10 + [6] * 2 + [6] * 2 + [7] * 2 +
                                  [7] * 2]
        pose_kpt_color = palette[[
            4, 4, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 7, 7, 7, 4, 4, 7, 7, 7, 4
        ]]

    elif dataset == 'AnimalFlyDataset':
        skeleton = [[2, 1], [3, 1], [4, 1], [5, 4], [6, 5], [8, 7], [9, 8],
                    [10, 9], [12, 11], [13, 12], [14, 13], [16, 15], [17, 16],
                    [18, 17], [20, 19], [21, 20], [22, 21], [24, 23], [25, 24],
                    [26, 25], [28, 27], [29, 28], [30, 29], [31, 4], [32, 4]]

        pose_limb_color = palette[[0] * 25]
        pose_kpt_color = palette[[0] * 32]

    elif dataset == 'AnimalLocustDataset':
        skeleton = [[2, 1], [3, 2], [4, 3], [5, 4], [7, 6], [8, 7], [10, 9],
                    [11, 10], [12, 11], [14, 13], [15, 14], [16, 15], [18, 17],
                    [19, 18], [20, 19], [22, 21], [23, 22], [25, 24], [26, 25],
                    [27, 26], [29, 28], [30, 29], [31, 30], [33, 32], [34, 33],
                    [35, 34]]

        pose_limb_color = palette[[0] * 26]
        pose_kpt_color = palette[[0] * 35]

    elif dataset == 'AnimalZebraDataset':
        skeleton = [[2, 1], [3, 2], [4, 3], [5, 3], [6, 8], [7, 8], [8, 3],
                    [9, 8]]

        pose_limb_color = palette[[0] * 8]
        pose_kpt_color = palette[[0] * 9]

    elif dataset in 'AnimalPoseDataset':
        skeleton = [[1, 2], [1, 3], [2, 4], [1, 5], [2, 5], [5, 6], [6, 8],
                    [7, 8], [6, 9], [9, 13], [13, 17], [6, 10], [10, 14],
                    [14, 18], [7, 11], [11, 15], [15, 19], [7, 12], [12, 16],
                    [16, 20]]

        pose_kpt_color = palette[[15] * 5 + [0] * 7 + [9] * 8]
        pose_limb_color = palette[[15] * 5 + [0] * 3 + [0, 9, 9] * 4]

    else:
        raise NotImplementedError()

    img = model.show_result(
        img,
        result,
        skeleton,
        radius=radius,
        thickness=thickness,
        pose_kpt_color=pose_kpt_color,
        pose_limb_color=pose_limb_color,
        kpt_score_thr=kpt_score_thr,
        bbox_color=bbox_color,
        show=show,
        out_file=out_file)

    return img


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results (list|tuple): mmdet results.
        cat_id (int): category id (default: 1 for human)

    Returns:
        person_results (list): a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results
