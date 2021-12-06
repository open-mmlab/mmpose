## 2D Hand Keypoint Demo

<img src="https://user-images.githubusercontent.com/11788150/109098558-8c54db00-775c-11eb-8966-85df96b23dc5.gif" width="600px" alt><br>

### 2D Hand Image Demo

#### Using gt hand bounding boxes as input

We provide a demo script to test a single image, given gt json file.

*Hand Pose Model Preparation:*
The pre-trained hand pose estimation model can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/latest/topics/hand%282d%29.html).
Take [onehand10k model](https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth) as an example:

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
    configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth \
    --img-root tests/data/onehand10k/ --json-file tests/data/onehand10k/test_onehand10k.json \
    --out-img-root vis_results
```

To run demos on CPU:

```shell
python demo/top_down_img_demo.py \
    configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth \
    --img-root tests/data/onehand10k/ --json-file tests/data/onehand10k/test_onehand10k.json \
    --out-img-root vis_results \
    --device=cpu
```

#### Using mmdet for hand bounding box detection

We provide a demo script to run mmdet for hand detection, and mmpose for hand pose estimation.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

*Hand Box Model Preparation:* The pre-trained hand box estimation model can be found in [det model zoo](/demo/docs/mmdet_modelzoo.md).

*Hand Pose Model Preparation:* The pre-trained hand pose estimation model can be downloaded from [pose model zoo](https://mmpose.readthedocs.io/en/latest/topics/hand%282d%29.html).

```shell
python demo/top_down_img_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --img ${IMG_FILE} \
    --out-img-root ${OUTPUT_DIR} \
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

```shell
python demo/top_down_img_demo_with_mmdet.py demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth \
    --img-root tests/data/onehand10k/ \
    --img 9.jpg \
    --out-img-root vis_results
```

### 2D Hand Video Demo

We also provide a video demo to illustrate the results.

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

*Hand Box Model Preparation:* The pre-trained hand box estimation model can be found in [det model zoo](/demo/docs/mmdet_modelzoo.md).

*Hand Pose Model Preparation:* The pre-trained hand pose estimation model can be found in [pose model zoo](https://mmpose.readthedocs.io/en/latest/topics/hand%282d%29.html).

```shell
python demo/top_down_video_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_video_demo_with_mmdet.py demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py \
    https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth \
    --video-path https://user-images.githubusercontent.com/87690686/137441388-3ea93d26-5445-4184-829e-bf7011def9e4.mp4 \
    --out-video-root vis_results
```

### Speed Up Inference

Some tips to speed up MMPose inference:

For 2D hand pose estimation models, try to edit the config file. For example,

1. set `flip_test=False` in [hand-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/hand/resnet/onehand10k/res50_onehand10k_256x256.py#L56).
1. set `post_process='default'` in [hand-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/hand/resnet/onehand10k/res50_onehand10k_256x256.py#L57).
