## 2D Pose Tracking Demo

<img src="https://user-images.githubusercontent.com/11788150/109099201-a93dde00-775d-11eb-9624-f9676fc0e478.gif" width="600px" alt><br>

### 2D Top-Down Video Human Pose Tracking Demo

We provide a video demo to illustrate the pose tracking results.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python demo/top_down_pose_tracking_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_PATH} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}] \
    [--use-oks-tracking --tracking-thr ${TRACKING_THR} --euro] \
    [--use-multi-frames] [--online]
```

Note that

1. `${VIDEO_PATH}` can be the local path or **URL** link to video file.

2. You can turn on the `[--use-multi-frames]` option to use multi frames for inference in the pose estimation stage.

3. If the `[--online]` option is set to **True**, future frame information can **not** be used when using multi frames for inference in the pose estimation stage.

Examples:

For single-frame inference that do not rely on extra frames to get the final results of the current frame, try this:

```shell
python demo/top_down_pose_tracking_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results
```

For multi-frame inference that rely on extra frames to get the final results of the current frame, try this:

```shell
python demo/top_down_pose_tracking_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py \
    https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth  \
    --video-path https://user-images.githubusercontent.com/87690686/137440639-fb08603d-9a35-474e-b65f-46b5c06b68d6.mp4 \
    --out-video-root vis_results \
    --use-multi-frames --online
```

### 2D Top-Down Video Human Pose Tracking Demo with MMTracking

MMTracking is an open source video perception toolbox based on PyTorch for tracking related tasks.
Here we show how to utilize MMTracking and MMPose to achieve human pose tracking.

Assume that you have already installed [mmtracking](https://github.com/open-mmlab/mmtracking).

```shell
python demo/top_down_video_demo_with_mmtracking.py \
    ${MMTRACKING_CONFIG_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_PATH} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}] \
    [--use-multi-frames] [--online]
```

Note that

1. `${VIDEO_PATH}` can be the local path or **URL** link to video file.

2. You can turn on the `[--use-multi-frames]` option to use multi frames for inference in the pose estimation stage.

3. If the `[--online]` option is set to **True**, future frame information can **not** be used when using multi frames for inference in the pose estimation stage.

Examples:

For single-frame inference that do not rely on extra frames to get the final results of the current frame, try this:

```shell
python demo/top_down_pose_tracking_demo_with_mmtracking.py \
    demo/mmtracking_cfg/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results
```

For multi-frame inference that rely on extra frames to get the final results of the current frame, try this:

```shell
python demo/top_down_pose_tracking_demo_with_mmtracking.py \
    demo/mmtracking_cfg/tracktor_faster-rcnn_r50_fpn_4e_mot17-private.py \
    configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py \
    https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth  \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results \
    --use-multi-frames --online
```

### 2D Bottom-Up Video Human Pose Tracking Demo

We also provide a pose tracking demo with bottom-up pose estimation methods.

```shell
python demo/bottom_up_pose_tracking_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_PATH} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_SCORE_THR} --pose-nms-thr ${POSE_NMS_THR}]
    [--use-oks-tracking --tracking-thr ${TRACKING_THR} --euro]
```

Note that `${VIDEO_PATH}` can be the local path or **URL** link to video file.

Examples:

```shell
python demo/bottom_up_pose_tracking_demo.py \
    configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/hrnet_w32_coco_512x512.py \
    https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
    --video-path demo/resources/demo.mp4 \
    --out-video-root vis_results
```

### Speed Up Inference

Some tips to speed up MMPose inference:

For top-down models, try to edit the config file. For example,

1. set `flip_test=False` in [topdown-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L51).
2. set `post_process='default'` in [topdown-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/top_down/resnet/coco/res50_coco_256x192.py#L52).
3. use faster human detector or human tracker, see [MMDetection](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) or [MMTracking](https://mmtracking.readthedocs.io/en/latest/model_zoo.html).

For bottom-up models, try to edit the config file. For example,

1. set `flip_test=False` in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L80).
2. set `adjust=False` in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L78).
3. set `refine=False` in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L79).
4. use smaller input image size in [AE-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/bottom_up/resnet/coco/res50_coco_512x512.py#L39).
