## 2D Human Pose Demo

<img src="https://raw.githubusercontent.com/open-mmlab/mmpose/master/demo/resources/demo_coco.gif" width="600px" alt><br>

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

The pre-trained hand pose estimation model can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/1.x/modelzoo_tasks/body.html).
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

#### Use mmdet for human bounding box detection

We provide a demo script to run mmdet for human detection, and mmpose for pose estimation.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${IMG_OR_VIDEO_FILE} \
    --output-root ${OUTPUT_DIR} \
    [--show --draw-heatmap --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --input tests/data/coco/000000197388.jpg --show --draw-heatmap \
    --output-root vis_results/
```

### 2D Human Pose Top-Down Video Demo

The above demo script can also take video as input, and run mmdet for human detection, and mmpose for pose estimation.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

Examples:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --input tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4 \
--output-root=vis_results/demo --show --draw-heatmap
```

### Speed Up Inference

Some tips to speed up MMPose inference:

For top-down models, try to edit the config file. For example,

1. set `flip_test=False` in [topdown-res50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py#L56).
2. use faster human bounding box detector, see [MMDetection](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).
