## 2D Human Pose Demo

We provide demo scripts to perform human pose estimation on images or videos.

### 2D Human Pose Top-Down Image Demo

#### Use full image as input

We provide a demo script to test a single image, using the full image as input bounding box.

```shell
python demo/image_demo.py \
    ${IMG_FILE} ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --out-file ${OUTPUT_FILE} \
    [--device ${GPU_ID or CPU}] \
    [--draw_heatmap]
```

If you use a heatmap-based model and set argument `--draw-heatmap`, the predicted heatmap will be visualized together with the keypoints.

The pre-trained human pose estimation models can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html).
Take [coco model](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) as an example:

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
```

To run this demo on CPU:

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file vis_results.jpg \
    --draw-heatmap \
    --device=cpu
```

Visualization result:

<img src="https://user-images.githubusercontent.com/87690686/187824033-2cce0f55-034a-4127-82e2-52744178bc32.jpg" height="500px" alt><br>

#### Use mmdet for human bounding box detection

We provide a demo script to run mmdet for human detection, and mmpose for pose estimation.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection) with version >= 3.0.

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} \
    [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--show] [--draw-heatmap] [--device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR}] [--kpt-thr ${KPT_SCORE_THR}]
```

Example:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --input tests/data/coco/000000197388.jpg --show --draw-heatmap \
    --output-root vis_results/
```

Visualization result:

<img src="https://user-images.githubusercontent.com/87690686/187824368-1f1631c3-52bf-4b45-bf9a-a70cd6551e1a.jpg" height="500px" alt><br>

To save the predicted results on disk, please specify `--save-predictions`.

### 2D Human Pose Top-Down Video Demo

The above demo script can also take video as input, and run mmdet for human detection, and mmpose for pose estimation. The difference is, the `${INPUT_PATH}` for videos can be the local path or **URL** link to video file.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection) with version >= 3.0.

Example:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
    --input tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4 \
    --output-root=vis_results/demo --show --draw-heatmap
```

### 2D Human Pose Bottom-up Image/Video Demo

We also provide a demo script using bottom-up models to estimate the human pose in an image or a video, which does not rely on human detectors.

```shell
python demo/bottomup_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} \
    [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--show] [--device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

Example:

```shell
python demo/bottomup_demo.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth \
    --input tests/data/coco/000000197388.jpg --output-root=vis_results \
    --show --save-predictions
```

Visualization result:

<img src="https://user-images.githubusercontent.com/26127467/207224032-a8dab45d-39e4-4b4e-80e0-3c71a64f5f39.jpg" height="300px" alt><br>

### 2D Human Pose Estimation with Inferencer

The Inferencer provides a convenient interface for inference, allowing customization using model aliases instead of configuration files and checkpoint paths. It supports various input formats, including image paths, video paths, image folder paths, and webcams. Below is an example command:

```shell
python demo/inferencer_demo.py \
    tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4 \
    --pose2d human --vis-out-dir vis_results/posetrack18
```

This command infers the video and saves the visualization results in the `vis_results/posetrack18` directory.

<img src="https://user-images.githubusercontent.com/26127467/229831445-44c9662b-edc5-4ef0-92a6-13558f0906cc.gif" alt="Image 1" height="300"/>

In addition, the Inferencer supports saving predicted poses. For more information, please refer to the [inferencer document](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface).

### Speed Up Inference

Some tips to speed up MMPose inference:

For top-down models, try to edit the config file. For example,

1. set `model.test_cfg.flip_test=False` in [topdown-res50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py#L56).
2. use faster human bounding box detector, see [MMDetection](https://mmdetection.readthedocs.io/en/3.x/model_zoo.html).
