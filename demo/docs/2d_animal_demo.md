## 2D Animal Pose Demo

### 2D Animal Pose Image Demo

#### Using gt hand bounding boxes as input

We provide a demo script to test a single image, given gt json file.

*Pose Model Preparation:*
The pre-trained pose estimation model can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/latest/topics/animal.html).
Take [macaque model](https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192-98f1dd3a_20210407.pth) as an example:

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
    configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/res50_macaque_256x192.py \
    https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192-98f1dd3a_20210407.pth \
    --img-root tests/data/macaque/ --json-file tests/data/macaque/test_macaque.json \
    --out-img-root vis_results
```

To run demos on CPU:

```shell
python demo/top_down_img_demo.py \
    configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/res50_macaque_256x192.py \
    https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192-98f1dd3a_20210407.pth \
    --img-root tests/data/macaque/ --json-file tests/data/macaque/test_macaque.json \
    --out-img-root vis_results \
    --device=cpu
```

### 2D Animal Pose Video Demo

We also provide video demos to illustrate the results.

#### Using the full image as input

If the video is cropped with the object centered in the screen, we can simply use the full image as the model input (without object detection).

```shell
python demo/top_down_video_demo_full_frame_without_det.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--show --device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_video_demo_full_frame_without_det.py \
    configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/fly/res152_fly_192x192.py \
    https://download.openmmlab.com/mmpose/animal/resnet/res152_fly_192x192-fcafbd5a_20210407.pth \
    --video-path demo/resources/<demo_fly_video.avi> \
    --out-video-root vis_results
```

<img src="https://user-images.githubusercontent.com/11788150/114023530-944c8280-98a5-11eb-86b0-5f6d3e232af0.gif" height="140px" alt><br>

#### Using MMDetection to detect animals

Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection).

**COCO-animals**

In COCO dataset, there are 80 object categories, including 10 common `animal` categories (15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe')
For these COCO-animals, please download the COCO pre-trained detection model from [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).

```shell
python demo/top_down_video_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    --det-cat-id ${CATEGORY_ID}
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_video_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split1.py \
    https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1-3a3dc37e_20210405.pth \
    --video-path demo/resources/<demo_horse.mp4> \
    --out-video-root vis_results \
    --bbox-thr 0.1 \
    --kpt-thr 0.4 \
    --det-cat-id 18
```

<img src="https://user-images.githubusercontent.com/11788150/114201893-4446ec00-9989-11eb-808b-5718c47c7b23.gif" height="140px" alt><br>

**Other Animals**

For other animals, we have also provided some pre-trained animal detection models (1-class models). Supported models can be found in [det model zoo](/demo/docs/mmdet_modelzoo.md).
The pre-trained animal pose estimation model can be found in [pose model zoo](https://mmpose.readthedocs.io/en/latest/topics/animal.html).

```shell
python demo/top_down_video_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --video-path ${VIDEO_FILE} \
    --out-video-root ${OUTPUT_VIDEO_ROOT} \
    [--det-cat-id ${CATEGORY_ID}]
    [--show --device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR} --kpt-thr ${KPT_SCORE_THR}]
```

Examples:

```shell
python demo/top_down_video_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_macaque-e45e36f5_20210409.pth \
    configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/macaque/res152_macaque_256x192.py \
    https://download.openmmlab.com/mmpose/animal/resnet/res152_macaque_256x192-c42abc02_20210407.pth \
    --video-path demo/resources/<demo_macaque.mp4> \
    --out-video-root vis_results \
    --bbox-thr 0.5 \
    --kpt-thr 0.3 \
```

<img src="https://user-images.githubusercontent.com/11788150/114205282-b5d46980-998c-11eb-9d6b-85ba47f81252.gif" height="140px" alt><br>

### Speed Up Inference

Some tips to speed up MMPose inference:

For 2D animal pose estimation models, try to edit the config file. For example,

1. set `flip_test=False` in [macaque-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/animal/resnet/macaque/res50_macaque_256x192.py#L51).
1. set `post_process='default'` in [macaque-res50](https://github.com/open-mmlab/mmpose/tree/e1ec589884235bee875c89102170439a991f8450/configs/animal/resnet/macaque/res50_macaque_256x192.py#L52).
