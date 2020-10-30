
### 2D Hand Image Demo

#### Using gt hand bounding boxes as input

We provide a demo script to test a single image, given gt json file.

*Hand Pose Model Preparation:*
The pre-trained hand pose estimation model can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/latest/pretrained.html).
Take [onehand10k model](https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth) as an example:


```shell
python demo/top_down_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

```shell
python demo/top_down_img_demo.py \
    configs/top_down/resnet/onehand10k/res50_onehand10k_256x256.py \
    models/res50_onehand10k_256x256-e67998f6_20200813.pth \
    --img-root tests/data/onehand10k/ --json-file tests/data/onehand10k/test_onehand10k.json \
    --out-img-root vis_results
```


#### Using mmdet for hand bounding box detection

We provide a demo script to run mmdet for hand detection, and mmpose for hand pose estimation.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

*Hand Box Model Preparation:* The pre-trained hand box estimation model can be found in [det model zoo](/demo/mmdet_modelzoo.md).

*Hand Pose Model Preparation:* The pre-trained hand pose estimation model can be downloaded from [pose model zoo](https://mmpose.readthedocs.io/en/latest/pretrained.html).

```shell
python demo/top_down_img_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --img ${IMG_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

```shell
python demo/top_down_img_demo_with_mmdet.py demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k.py \
    models/mmdet/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/top_down/resnet/onehand10k/res50_onehand10k_256x256.py \
    models/res50_onehand10k_256x256-e67998f6_20200813.pth \
    --img-root tests/data/onehand10k/ \
    --img 9.jpg \
    --out-img-root vis_results
```

### 2D Hand Video Demo

We also provide a video demo to illustrate the results.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

*Hand Box Model Preparation:* The pre-trained hand box estimation model can be found in [det model zoo](/demo/mmdet_modelzoo.md).

*Hand Pose Model Preparation:* The pre-trained hand pose estimation model can be downloaded from [pose model zoo](https://mmpose.readthedocs.io/en/latest/pretrained.html).

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
python demo/top_down_video_demo_with_mmdet.py demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k.py \
    models/mmdet/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/top_down/resnet/onehand10k/res50_onehand10k_256x256.py \
    models/res50_onehand10k_256x256-e67998f6_20200813.pth \
    --video-path demo/demo_video.mp4 \
    --out-video-root vis_results
```

### Speed Up Inference
Some tips to speed up MMPose inference:

For 2D hand pose estimation models, try to edit the config file.
1. set `flip_test=False` (line 56 in [res50](/configs/top_down/resnet/onehand10k/res50_onehand10k_256x256.py))
2. set `unbiased_decoding=False` (line 59 in [res50](/configs/top_down/resnet/onehand10k/res50_onehand10k_256x256.py))
