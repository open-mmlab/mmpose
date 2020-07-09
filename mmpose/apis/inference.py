import os

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmpose.datasets.pipelines import Compose
from mmpose.models import build_posenet

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
        np.ndarray[float32](2,): Center of the bbox (x, y).
        np.ndarray[float32](2,): Scale of the bbox w & h.
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


def _inference_single_pose_model(model, image_name, bbox):
    """Inference a single bbox.

    num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        image_name (str | np.ndarray):Image_name
        bbox (list | np.ndarray): Bounding boxes (with scores),
            shaped (4, ) or (5, ). (left, top, width, height, [score])

    Returns:
        ndarray[Kx3]: Predicted pose x, y, score.
    """
    cfg = model.cfg
    device = next(model.parameters()).device
    # build the data pipeline
    test_pipeline = Compose(cfg.valid_pipeline)

    center, scale = _box2cs(cfg, bbox)
    # prepare data
    data = {
        'image_file': image_name,
        'center': center,
        'scale': scale,
        'bbox_score': bbox[4] if len(bbox) == 5 else 1,
        'dataset': 'coco',
        'rotation': 0,
        'imgnum': 0,
        'joints_3d': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float),
        'joints_3d_visible': np.zeros((cfg.data_cfg.num_joints, 3),
                                      dtype=np.float),
        'ann_info': {
            'image_size':
            cfg.data_cfg['image_size'],
            'num_joints':
            cfg.data_cfg['num_joints'],
            'flip_pairs': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12],
                           [13, 14], [15, 16]],
        }
    }
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        all_preds, _, _ = model(
            return_loss=False,
            img=data['img'],
            img_metas=data['img_metas'].data[0])

    return all_preds[0]


def inference_pose_model(model,
                         image_name,
                         person_bboxes,
                         bbox_thr=None,
                         format='xywh'):
    """Inference a single image with a list of person bounding boxes.

    num_people: P
    num_keypoints: K
    bbox height: H
    bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        image_name (str| np.ndarray): Image_name
        person_bboxes: (np.ndarray[P x 4] or [P x 5]): Each person bounding box
            shaped (4, ) or (5, ), contains 4 box coordinates (and score).
        bbox_thr: Threshold for bounding boxes. Only bboxes with higher scores
            will be fed into the pose detector. If bbox_thr is None, ignore it.
        format: bbox format ('xyxy' | 'xywh'). Default: 'xywh'.
            'xyxy' means (left, top, right, bottom),
            'xywh' means (left, top, width, height).

    Returns:
        list[dict]: Pose results: each item in the list is a dictionary,
            containing the bbox: (left, top, right, bottom, [score])
            and the pose (ndarray[Kx3]): x, y, score
    """
    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']
    # transform the bboxes format to xywh
    if format == 'xyxy':
        person_bboxes = _xyxy2xywh(np.array(person_bboxes))
    pose_results = []

    if len(person_bboxes) > 0:
        if bbox_thr is not None:
            person_bboxes = person_bboxes[person_bboxes[:, 4] > bbox_thr]
        for bbox in person_bboxes:
            pose = _inference_single_pose_model(model, image_name, bbox)
            pose_results.append({
                'bbox':
                _xywh2xyxy(np.expand_dims(np.array(bbox), 0)),
                'keypoints':
                pose,
            })

    return pose_results


def show_pose_result(model,
                     img,
                     result,
                     kpt_score_thr=0.3,
                     skeleton=None,
                     fig_size=(15, 10)):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str | np.ndarray): Image filename or loaded image.
        result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]):
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module

    img = model.show_result(
        img, result, skeleton, kpt_score_thr=kpt_score_thr, show=False)

    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.show()


def save_pose_vis(model,
                  img,
                  result,
                  out_file=None,
                  kpt_score_thr=0.3,
                  skeleton=None):
    """Visualize the detection results on the image and save the img file.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (Tensor | tuple): The results to draw over `img`
                (bbox_result, pose_result).
        kpt_score_thr (float): The threshold to visualize the keypoints.
        skeleton (list[tuple()]):
        out_file (str | None): The filename to write the image.
                Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module

    assert out_file is not None
    model.show_result(
        img,
        result,
        skeleton,
        kpt_score_thr=kpt_score_thr,
        show=False,
        out_file=out_file)
    return 0
