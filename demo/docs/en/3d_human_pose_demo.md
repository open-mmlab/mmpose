## 3D Human Pose Demo

<img  src="https://user-images.githubusercontent.com/15977946/118820606-02df2000-b8e9-11eb-9984-b9228101e780.gif"  width="600px"  alt><br>

### 3D Human Pose Two-stage Estimation Demo

#### Using mmdet for human bounding box detection and top-down model for the 1st stage (2D pose detection), and inference the 2nd stage (2D-to-3D lifting)

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python  demo/body3d_pose_lifter_demo.py  \
${MMDET_CONFIG_FILE} \
${MMDET_CHECKPOINT_FILE} \
${MMPOSE_CONFIG_FILE_2D} \
${MMPOSE_CHECKPOINT_FILE_2D} \
${MMPOSE_CONFIG_FILE_3D} \
${MMPOSE_CHECKPOINT_FILE_3D} \
--input ${VIDEO_PATH or IMAGE_PATH or 'webcam'} \
[--show] \
[--disable-rebase-keypoint] \
[--disable-norm-pose-2d] \
[--num-instances ${NUM_INSTANCES}] \
[--output-root ${OUT_VIDEO_ROOT}] \
[--save-predictions] \
[--device ${GPU_ID  or  CPU}] \
[--det-cat-id ${DET_CAT_ID}] \
[--bbox-thr ${BBOX_THR}] \
[--kpt-thr ${KPT_THR}] \
[--use-oks-tracking] \
[--tracking-thr ${TRACKING_THR}] \
[--show-interval ${INTERVAL}] \
[--thickness ${THICKNESS}] \
[--radius ${RADIUS}] \
[--online]
```

Note that

1. `${VIDEO_PATH}` can be the local path or **URL** link to video file.

2. If the `[--online]` option is set to **True**, future frame information can **not** be used when using multi frames for inference in the 2D pose detection stage.

Examples:

During 2D pose detection, for single-frame inference that do not rely on extra frames to get the final results of the current frame and save the prediction results, try this:

```shell
python demo/body3d_pose_lifter_demo.py  \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py \
https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth  \
--input https://user-images.githubusercontent.com/87690686/164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.mp4 \
--output-root  vis_results \
--save-predictions
```

During 2D pose detection, for multi-frame inference that rely on extra frames to get the final results of the current frame, try this:

```shell
python demo/body3d_pose_lifter_demo.py \
demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py \
https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth \
--input https://user-images.githubusercontent.com/87690686/164970135-b14e424c-765a-4180-9bc8-fa8d6abc5510.mp4 \
--output-root  vis_results  \
--online
```

### 3D Human Pose Demo with Inferencer

The Inferencer provides a convenient interface for inference, allowing customization using model aliases instead of configuration files and checkpoint paths. It supports various input formats, including image paths, video paths, image folder paths, and webcams. Below is an example command:

```shell
python demo/inferencer_demo.py tests/data/coco/000000000785.jpg \
    --pose3d human3d --vis-out-dir vis_results/human3d
```

This command infers the image and saves the visualization results in the `vis_results/human3d` directory.

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/9621f51f-59e4-41e5-ab4c-3b03e97f0e9d" alt="Image 1" height="300"/>

In addition, the Inferencer supports saving predicted poses. For more information, please refer to the [inferencer document](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html#inferencer-a-unified-inference-interface).
