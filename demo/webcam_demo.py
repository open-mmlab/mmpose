import argparse
from collections import defaultdict

import cv2
import mmcv
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


class StopWatch:
    r"""A helper class to measure FPS and detailed time consuming of each phase
    in a video processing loop or similar scenarios.

    Args:
        window (int): The sliding window size to calculate the running average
            of the time consuming.

    Example::
        >>> stop_watch = StopWatch(window=10)
        >>> while True:
        ...     with stop_watch.timeit('total'):
        ...         sleep(1)
        ...         # 'timeit' support nested use
        ...         with stop_watch.timeit('phase1'):
        ...             sleep(1)
        ...         with stop_watch.timeit('phase2'):
        ...             sleep(2)
        ...         sleep(2)
        ...     report = stop_watch.report()
        report = {'total': 6., 'phase1': 1., 'phase2': 2.}

    """

    def __init__(self, window=1):
        self._record = defaultdict(list)
        self._timer_stack = []
        self.window = window

    def timeit(self, timer_name='_FPS_'):
        """Timing a code snippet with an assigned name.

        Args:
            timer_name (str): The unique name of the interested code snippet to
                handle multiple timers and generate reports. Note that '_FPS_'
                is a special key that the measurement will be in `fps` instead
                of `millisecond`. Also see `report` and `report_strings`.
                Default: '_FPS_'.
        Note:
            This function should always be used in a `with` statement, as shown
            in the example.
        """
        self._timer_stack.append((timer_name, mmcv.Timer()))
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, trackback):
        timer_name, timer = self._timer_stack.pop()
        self._record[timer_name].append(timer.since_start())
        self._record[timer_name] = self._record[timer_name][-self.window:]

    def report(self):
        """Report timing information.

        Returns:
            dict: The key is the timer name and the value is the corresponding
                average time consuming.
        """
        result = {
            name: np.mean(vals) * 1000.
            for name, vals in self._record.items()
        }
        return result

    def report_strings(self):
        """Report timing information in texture strings.

        Returns:
            list(str): Each element is the information string of a timed event,
                in format of '{timer_name}: {time_in_ms}'. Specially, if
                timer_name is '_FPS_', the measurement will be converted to
                fps.
        """
        result = self.report()
        strings = []
        if '_FPS_' in result:
            fps = 1.0 / result.pop('_FPS_')
            strings.append(f'FPS: {fps:.1f}')
        strings += [f'{name}: {val:.0f}' for name, val in result.items()]
        return strings

    def reset(self):
        self._record = defaultdict(list)
        self._timer_stack = []


def apply_bugeye_effect(img,
                        pose_results,
                        dataset='TopDownCocoDataset',
                        kpt_thr=0.5):
    """Apply bug-eye effect.

    Args:
        img (np.ndarray): Image data.
        pose_results (list[dict]): The pose estimation results containing:
            - "bbox" ([K, 4(or 5)]): detection bbox in
                [x1, y1, x2, y2, (score)]
            - "keypoints" ([K,3]): keypoint detection result in [x, y, score]
        dataset (str): Dataset name (e.g. 'TopDownCocoDataset') to determine
            the keypoint order.
        kpt_thr (float): The score threshold of required keypoints.
    """

    if dataset == 'TopDownCocoDataset':
        leye_idx = 1
        reye_idx = 2
    elif dataset == 'AnimalPoseDataset':
        leye_idx = 0
        reye_idx = 1
    else:
        raise NotImplementedError()

    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    for pose in pose_results:
        bbox = pose['bbox']
        kpts = pose['keypoints']

        if kpts[leye_idx, 2] < kpt_thr or kpts[reye_idx, 2] < kpt_thr:
            continue

        kpt_leye = kpts[leye_idx, :2]
        kpt_reye = kpts[reye_idx, :2]
        for xc, yc in [kpt_leye, kpt_reye]:
            # draw dard dot at the eye position
            cv2.circle(img, (int(xc), int(yc)), 1, (20, 20, 20), cv2.FILLED)

            # distortion parameters
            k1 = 0.002
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
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE)
    return img


def apply_sunglasses_effect(img,
                            pose_results,
                            sunglasses_img,
                            dataset='TopDownCocoDataset',
                            kpt_thr=0.5):
    """Apply sunglasses effect.

    Args:
        img (np.ndarray): Image data.
        pose_results (list[dict]): The pose estimation results containing:
            - "keypoints" ([K,3]): keypoint detection result in [x, y, score]
        sunglasses_img (np.ndarray): Sunglasses image with white background.
        dataset (str): Dataset name (e.g. 'TopDownCocoDataset') to determine
            the keypoint order.
        kpt_thr (float): The score threshold of required keypoints.
    """

    if dataset == 'TopDownCocoDataset':
        leye_idx = 1
        reye_idx = 2
    elif dataset == 'AnimalPoseDataset':
        leye_idx = 0
        reye_idx = 1
    else:
        raise NotImplementedError()

    hm, wm = sunglasses_img.shape[:2]
    # anchor points in the sunglasses mask
    pts_src = np.array([[0.3 * wm, 0.3 * hm], [0.3 * wm, 0.7 * hm],
                        [0.7 * wm, 0.3 * hm], [0.7 * wm, 0.7 * hm]],
                       dtype=np.float32)

    for pose in pose_results:
        kpts = pose['keypoints']

        if kpts[leye_idx, 2] < kpt_thr or kpts[reye_idx, 2] < kpt_thr:
            continue

        kpt_leye = kpts[leye_idx, :2]
        kpt_reye = kpts[reye_idx, :2]
        # orthogonal vector to the left-to-right eyes
        vo = 0.5 * (kpt_reye - kpt_leye)[::-1]

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


def process_mmdet_results(mmdet_results, class_names=None, cat_ids=1):
    """Process mmdet results to mmpose input format.

    Args:
        mmdet_results: raw output of mmdet model
        class_names: class names of mmdet model
        cat_ids (int or List[int]): category id list that will be preserved
    Returns:
        List[Dict]: detection results for mmpose input
    """
    if isinstance(mmdet_results, tuple):
        mmdet_results = mmdet_results[0]

    if not isinstance(cat_ids, (list, tuple)):
        cat_ids = [cat_ids]

    # only keep bboxes of interested classes
    bbox_results = [mmdet_results[i - 1] for i in cat_ids]
    bboxes = np.vstack(bbox_results)

    # get textual labels of classes
    labels = np.concatenate([
        np.full(bbox.shape[0], i - 1, dtype=np.int32)
        for i, bbox in zip(cat_ids, bbox_results)
    ])
    if class_names is None:
        labels = [f'class: {i}' for i in labels]
    else:
        labels = [class_names[i] for i in labels]

    det_results = []
    for bbox, label in zip(bboxes, labels):
        det_result = dict(bbox=bbox, label=label)
        det_results.append(det_result)
    return det_results


def webcam_demo(args):
    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    # build detection model
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build pose models
    pose_models = []
    if args.enable_human_pose:
        pose_model = init_pose_model(
            args.human_pose_config,
            args.human_pose_checkpoint,
            device=args.device.lower())
        model_name = 'HumanPose'
        cat_ids = args.human_det_ids
        bbox_color = 'green'
        pose_models.append((model_name, pose_model, cat_ids, bbox_color))
    if args.enable_animal_pose:
        pose_model = init_pose_model(
            args.animal_pose_config,
            args.animal_pose_checkpoint,
            device=args.device.lower())
        model_name = 'AnimalPose'
        cat_ids = args.animal_det_ids
        bbox_color = 'cyan'
        pose_models.append((model_name, pose_model, cat_ids, bbox_color))

    # load resource
    sunglasses_img = None
    if args.sunglasses:
        # The image attributes to:
        # https://www.vecteezy.com/free-vector/glass
        # Glass Vectors by Vecteezy
        sunglasses_img = mmcv.imread('demo/resources/sunglasses.jpg')

    # init video reader and writer
    vid_cap = cv2.VideoCapture(args.cam_id)
    if not vid_cap.isOpened():
        print(f'Cannot open camera (ID={args.cam_id})')
        exit()
    vid_out = None

    # use stop_watch to measure time consuming
    stop_watch = StopWatch(window=10)
    while True:
        with stop_watch.timeit():
            # capture a camera frame
            ret_val, frame = vid_cap.read()
            if not ret_val:
                break

            # inference detection
            with stop_watch.timeit('Det'):
                mmdet_results = inference_detector(det_model, frame)
            img = frame.copy()

            for model_name, pose_model, cat_ids, color in pose_models:
                with stop_watch.timeit(model_name):
                    det_results = process_mmdet_results(
                        mmdet_results,
                        class_names=det_model.CLASSES,
                        cat_ids=cat_ids)

                    dataset_name = pose_model.cfg.data['test']['type']
                    pose_results, _ = inference_top_down_pose_model(
                        pose_model,
                        frame,
                        det_results,
                        bbox_thr=args.det_score_thr,
                        format='xyxy',
                        dataset=dataset_name)

                if args.sunglasses:
                    img = apply_sunglasses_effect(img, pose_results,
                                                  sunglasses_img, dataset_name)
                elif args.bugeye:
                    img = apply_bugeye_effect(img, pose_results, dataset_name)
                else:
                    img = vis_pose_result(
                        pose_model,
                        img,
                        pose_results,
                        dataset=dataset_name,
                        kpt_score_thr=args.kpt_thr,
                        bbox_color=color)

            str_info = [f'Shape: {img.shape[:2]}']
            str_info += stop_watch.report_strings()
            str_info = ' | '.join(str_info)
            cv2.putText(img, str_info, (30, 20), cv2.FONT_HERSHEY_DUPLEX, 0.3,
                        (228, 183, 61), 1)

            if args.out_video_file is not None:
                if vid_out is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = 20
                    frame_size = (img.shape[1], img.shape[0])
                    vid_out = cv2.VideoWriter(args.out_video_file, fourcc, fps,
                                              frame_size)

                vid_out.write(img)

            cv2.imshow('mmpose webcam demo', img)
            if cv2.waitKey(1) == 27:  # esc to stop
                break

    vid_cap.release()
    if vid_out is not None:
        vid_out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument(
        '--det_config',
        type=str,
        default='demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py',
        help='Config file for detection')
    parser.add_argument(
        '--det_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmdetection/v2.0/yolo/'
        'yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--enable_human_pose',
        type=int,
        default=1,
        help='Enable human pose estimation')
    parser.add_argument(
        '--enable_animal_pose',
        type=int,
        default=1,
        help='Enable animal pose estimation')
    parser.add_argument(
        '--human_pose_config',
        type=str,
        default='configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/'
        'hrnet_w48_coco_256x192.py',
        help='Config file for human pose')
    parser.add_argument(
        '--human_pose_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/'
        'hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth',
        help='Checkpoint file for human pose')
    parser.add_argument(
        '--human_det_ids',
        type=int,
        default=[1],
        nargs='+',
        help='Object category label of human in detection results.'
        'Default is [1(person)], following COCO definition.')
    parser.add_argument(
        '--animal_pose_config',
        type=str,
        default='configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/'
        'animalpose/hrnet_w32_animalpose_256x256.py',
        help='Config file for animal pose')
    parser.add_argument(
        '--animal_pose_checkpoint',
        type=str,
        default='https://download.openmmlab.com/mmpose/animal/hrnet/'
        'hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
        help='Checkpoint file for animal pose')
    parser.add_argument(
        '--animal_det_ids',
        type=int,
        default=[16, 17, 18, 19, 20],
        nargs='+',
        help='Object category label of animals in detection results'
        'Default is [16(cat), 17(dog), 18(horse), 19(sheep), 20(cow)], '
        'following COCO definition.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.5,
        help='bbox score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--sunglasses', action='store_true', help='Apply `sunglasses` effect.')
    parser.add_argument(
        '--bugeye', action='store_true', help='Apply `bug-eye` effect.')

    parser.add_argument(
        '--out-video-file',
        type=str,
        default=None,
        help='Record the video into a file. This may reduce the frame rate')
    args = parser.parse_args()

    webcam_demo(args)
