
### Top-down image demo

#### Using gt human bounding boxes as input

We provide a demo script to test a single image, given gt json file.

```shell
python demo/top_down_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_img_demo.py \
    configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py \
    hrnet_w48_coco_256x192/epoch_210.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results
```

#### Using mmdet for human bounding box detection

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

We provide a demo script to run mmdet for human detection, and mmpose for pose estimation.

```shell
python demo/top_down_img_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --img ${IMG_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_img_demo_with_mmdet.py mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py \
    hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --img-root tests/data/coco/ \
    --img 000000196141.jpg \
    --out-img-root vis_results
```

### Top-down video demo

We also provide a video demo to illustrate the results.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

```shell
python demo/top_down_video_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --output-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_video_demo_with_mmdet.py mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py \
    hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --video-path demo/demo_video.mp4 \
    --out-video-root vis_results
```

### Bottom-up image demo

We provide a demo script to test a single image.

```shell
python demo/bottom_up_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/bottom_up_img_demo.py \
    configs/bottom_up/hrnet/coco/hrnet_w32_coco_512x512.py \
    hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
    --img-root tests/data/coco/ --json-file tests/data/coco/test_coco.json \
    --out-img-root vis_results
```

### Bottom-up video demo

We also provide a video demo to illustrate the results.

```shell
python demo/bottom_up_video_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --output-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/bottom_up_video_demo.py \
    configs/bottom_up/hrnet/coco/hrnet_w32_coco_512x512.py \
    hrnet_w32_coco_512x512-bcb8c247_20200816.pth \
    --video-path demo/demo_video.mp4 \
    --out-video-root vis_results
```

### Speed up inference
Some tips to speed up MMPose inference:

For top-down models, try to edit the config file.
1. set `flip_test=False` (line 51 in [topdown-res50](/configs/top_down/resnet/coco/res50_coco_256x192.py))
2. set `unbiased_decoding=False` (line 54 in [topdown-res50](/configs/top_down/resnet/coco/res50_coco_256x192.py))

For bottom-up models, try to edit the config file.
1. set `flip_test=False` (line 80 in [bottomup-res50](/configs/bottom_up/resnet/coco/res50_coco_512x512.py))
2. set `adjust=False` (line 78 in [bottomup-res50](/configs/bottom_up/resnet/coco/res50_coco_512x512.py))
3. set `refine=False` (line 79 in [bottomup-res50](/configs/bottom_up/resnet/coco/res50_coco_512x512.py))
4. use smaller input image size (line 39 in [bottomup-res50](/configs/bottom_up/resnet/coco/res50_coco_512x512.py))
