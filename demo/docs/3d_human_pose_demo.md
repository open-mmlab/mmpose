## 3D Human Pose Demo

<img src="https://user-images.githubusercontent.com/15977946/118820606-02df2000-b8e9-11eb-9984-b9228101e780.gif" width="600px" alt><br>

### 3D Human Pose Two-stage Estimation Image Demo

#### Using ground truth 2D poses as the 1st stage (pose detection) result, and inference the 2nd stage (2D-to-3D lifting)

We provide a demo script to test on single images with a given ground-truth Json file.

```shell
python demo/body3d_two_stage_img_demo.py \
    ${MMPOSE_CONFIG_FILE_3D} \
    ${MMPOSE_CHECKPOINT_FILE_3D} \
    --json-file ${JSON_FILE} \
    --img-root ${IMG_ROOT} \
    --only-second-stage \
    [--show] \
    [--device ${GPU_ID or CPU}] \
    [--out-img-root ${OUTPUT_DIR}] \
    [--rebase-keypoint-height] \
    [--show-ground-truth]
```

Example:

```shell
python demo/body3d_two_stage_img_demo.py \
    configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py \
    https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth \
    --json-file tests/data/h36m/h36m_coco.json \
    --img-root tests/data/h36m \
    --camera-param-file tests/data/h36m/cameras.pkl \
    --only-second-stage \
    --out-img-root vis_results \
    --rebase-keypoint-height \
    --show-ground-truth
```

### 3D Human Pose Two-stage Estimation Video Demo

#### Using mmdet for human bounding box detection and top-down model for the 1st stage (2D pose detection), and inference the 2nd stage (2D-to-3D lifting)

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python demo/body3d_two_stage_video_demo.py \
    ${MMDET_CONFIG_FILE} \
    ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE_2D} \
    ${MMPOSE_CHECKPOINT_FILE_2D} \
    ${MMPOSE_CONFIG_FILE_3D} \
    ${MMPOSE_CHECKPOINT_FILE_3D} \
    --video-path ${VIDEO_PATH} \
    [--rebase-keypoint-height] \
    [--norm-pose-2d] \
    [--num-poses-vis NUM_POSES_VIS] \
    [--show] \
    [--out-video-root ${OUT_VIDEO_ROOT}] \
    [--device ${GPU_ID or CPU}] \
    [--det-cat-id DET_CAT_ID] \
    [--bbox-thr BBOX_THR] \
    [--kpt-thr KPT_THR] \
    [--use-oks-tracking] \
    [--tracking-thr TRACKING_THR] \
    [--euro] \
    [--radius RADIUS] \
    [--thickness THICKNESS]
```

Example:

```shell
python demo/body3d_two_stage_video_demo.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py \
    https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth \
    --video-path demo/resources/<demo_body3d>.mp4 \
    --out-video-root vis_results \
    --rebase-keypoint-height
```
