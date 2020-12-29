## 2D Pose Tracking Demo

### 2D Top-Down Video Human Pose Tracking Demo

We provide a video demo to illustrate the pose tracking results.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python demo/top_down_video_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --output-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR} --iou-thr ${IOU_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_pose_tracking_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_1x_coco.py \
    http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/top_down/resnet/coco/res50_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth \
    --video-path demo/demo_video.mp4 \
    --out-video-root vis_results
```

### Speed Up Inference
Some tips to speed up MMPose inference:

For top-down 2D human pose models, try to edit the config file. For example,
1. set `flip_test=False` in [topdown-res50](/configs/top_down/resnet/coco/res50_coco_256x192.py#L51).
2. set `unbiased_decoding=False` in [topdown-res50](/configs/top_down/resnet/coco/res50_coco_256x192.py#L54).
