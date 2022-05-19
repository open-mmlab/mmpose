## 2D Face Keypoint Demo

<img src="https://user-images.githubusercontent.com/11788150/109144943-ccd44900-779c-11eb-9e9d-8682e7629654.gif" width="600px" alt><br>

### 2D Face Image Demo

#### Using gt face bounding boxes as input

We provide a demo script to test a single image, given gt json file.

*Face Keypoint Model Preparation:*
The pre-trained face keypoint estimation model can be found from [model zoo](https://mmpose.readthedocs.io/en/latest/topics/face.html).
Take [aflw model](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) as an example:

```shell
python demo/top_down_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_img_demo.py \
    configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --img-root tests/data/aflw/ --json-file tests/data/aflw/test_aflw.json \
    --out-img-root vis_results
```

To run demos on CPU:

```shell
python demo/top_down_img_demo.py \
    configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --img-root tests/data/aflw/ --json-file tests/data/aflw/test_aflw.json \
    --out-img-root vis_results \
    --device=cpu
```

#### Using face bounding box detectors

We provide a demo script to run face detection and face keypoint estimation.

Please install `face_recognition` before running the demo, by `pip install face_recognition`.
For more details, please refer to https://github.com/ageitgey/face_recognition.

```shell
python demo/face_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --img ${IMG_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

```shell
python demo/face_img_demo.py \
    configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --img-root tests/data/aflw/ \
    --img image04476.jpg \
    --out-img-root vis_results
```

### 2D Face Video Demo

We also provide a video demo to illustrate the results.

Please install `face_recognition` before running the demo, by `pip install face_recognition`.
For more details, please refer to https://github.com/ageitgey/face_recognition.

```shell
python demo/face_video_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_PATH} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

Note that `${VIDEO_PATH}` can be the local path or **URL** link to video file.

Examples:

```shell
python demo/face_video_demo.py \
    configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --video-path https://user-images.githubusercontent.com/87690686/137441355-ec4da09c-3a8f-421b-bee9-b8b26f8c2dd0.mp4 \
    --out-video-root vis_results
```

### Speed Up Inference

Some tips to speed up MMPose inference:

For 2D face keypoint estimation models, try to edit the config file. For example,

1. set `flip_test=False` in [face-hrnetv2_w18](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/face/hrnetv2/aflw/hrnetv2_w18_aflw_256x256.py#L83).
2. set `post_process='default'` in [face-hrnetv2_w18](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/face/hrnetv2/aflw/hrnetv2_w18_aflw_256x256.py#L84).
