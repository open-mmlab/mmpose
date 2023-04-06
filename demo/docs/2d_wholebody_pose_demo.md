## 2D Human Whole-Body Pose Demo

### 2D Human Whole-Body Pose Top-Down Image Demo

#### Use full image as input

We provide a demo script to test a single image, using the full image as input bounding box.

```shell
python demo/image_demo.py \
    ${IMG_FILE} ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --out-file ${OUTPUT_FILE} \
    [--device ${GPU_ID or CPU}] \
    [--draw_heatmap]
```

The pre-trained hand pose estimation models can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/2d_wholebody_keypoint.html).
Take [coco-wholebody_vipnas_res50_dark](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth) model as an example:

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-res50_dark-8xb64-210e_coco-wholebody-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth \
    --out-file vis_results.jpg
```

To run demos on CPU:

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-res50_dark-8xb64-210e_coco-wholebody-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth \
    --out-file vis_results.jpg \
    --device=cpu
```

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

Examples:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --input tests/data/coco/000000196141.jpg \
    --output-root vis_results/ --show
```

To save the predicted results on disk, please specify `--save-predictions`.

### 2D Human Whole-Body Pose Top-Down Video Demo

The above demo script can also take video as input, and run mmdet for human detection, and mmpose for pose estimation.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

Examples:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --input https://user-images.githubusercontent.com/87690686/137440639-fb08603d-9a35-474e-b65f-46b5c06b68d6.mp4 \
    --output-root vis_results/ --show
```

Visualization result:

<img src="https://user-images.githubusercontent.com/87690686/190854069-634e1142-d13c-4863-9930-1120057ca77e.gif" height="350px" alt><br>

### 2D Human Whole-Body Pose Estimation with Inferencer

The Inferencer provides a convenient interface for inference, allowing customization using model aliases instead of configuration files and checkpoint paths. It supports various input formats, including image paths, video paths, image folder paths, and webcams. Below is an example command:

```shell
python demo/inferencer_demo.py tests/data/crowdpose \
    --pose2d wholebody --vis-out-dir vis_results/crowdpose
```

This command infers all images located in `tests/data/crowdpose` and saves the visualization results in the `vis_results/crowdpose` directory.

<img src="https://user-images.githubusercontent.com/26127467/229832887-31edb6d5-bcf0-44a4-a66f-9d523061a6e9.jpg" alt="Image 1" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229832908-bc82dbc9-5e43-4800-acc7-a7da85a653c7.jpg" alt="Image 2" height="200"/>

In addition, the Inferencer supports saving predicted poses. For more information, please refer to the [inferencer document](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface).

### Speed Up Inference

Some tips to speed up MMPose inference:

For top-down models, try to edit the config file. For example,

1. set `model.test_cfg.flip_test=False` in [pose_hrnet_w48_dark+](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py#L90).
2. use faster human bounding box detector, see [MMDetection](https://mmdetection.readthedocs.io/en/3.x/model_zoo.html).
