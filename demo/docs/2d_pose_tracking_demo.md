## 2D Pose Tracking Demo

<img src="https://user-images.githubusercontent.com/11788150/109099201-a93dde00-775d-11eb-9624-f9676fc0e478.gif" width="600px" alt><br>

### 2D Top-Down Video Human Pose Tracking Demo

We provide a video demo to illustrate the pose tracking results.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python demo/top_down_video_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR} --iou-thr ${IOU_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_pose_tracking_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/top_down/resnet/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results
```

### 2D Top-Down Video Human Pose Tracking Demo with MMTracking

MMTracking is an open source video perception toolbox based on PyTorch for tracking related tasks.
Here we show how to utilize MMTracking and MMPose to achieve human pose tracking.

Assume that you have already installed [mmtracking](https://github.com/open-mmlab/mmtracking).

```shell
python demo/top_down_video_demo_with_mmtracking.py \
    ${MMTRACKING_CONFIG_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_pose_tracking_demo_with_mmtracking.py \
    demo/mmtracking_cfg/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py \
    configs/top_down/resnet/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results
```

### Speed Up Inference

Some tips to speed up MMPose inference:

For top-down 2D human pose models, try to edit the config file. For example,

1. set `flip_test=False` in [topdown-res50](/configs/top_down/resnet/coco/res50_coco_256x192.py#L51).
2. set `post_process='default'` in [topdown-res50](/configs/top_down/resnet/coco/res50_coco_256x192.py#L52).
1. use faster human detector or human tracker, see [MMDetection](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) or [MMTracking](https://github.com/open-mmlab/mmtracking/blob/master/docs/model_zoo.md).
