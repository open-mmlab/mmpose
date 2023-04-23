## 2D Animal Pose Demo

We provide a demo script to test a single image or video with top-down pose estimators and animal detectors. Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection) with version >= 3.0.

### 2D Animal Pose Image Demo

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} --det-cat-id ${DET_CAT_ID} \
    [--show] [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}] [--bbox-thr ${BBOX_SCORE_THR}] \
    [--device ${GPU_ID or CPU}]
```

The pre-trained animal pose estimation model can be found from [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/animal_2d_keypoint.html).
Take [animalpose model](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth) as an example:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --show --draw-heatmap --det-cat-id=15
```

Visualization result:

<img src="https://user-images.githubusercontent.com/26127467/187644168-5915551a-0876-4b85-9454-7f92c84ba6fb.jpeg" height="500px" alt><br>

If you use a heatmap-based model and set argument `--draw-heatmap`, the predicted heatmap will be visualized together with the keypoints.

The augement `--det-cat-id=15` selected detected bounding boxes with label 'cat'. 15 is the index of category 'cat' in COCO dataset, on which the detection model is trained.

**COCO-animals**
In COCO dataset, there are 80 object categories, including 10 common `animal` categories (14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe').

For other animals, we have also provided some pre-trained animal detection models (1-class models). Supported models can be found in [detection model zoo](/demo/docs/en/mmdet_modelzoo.md).

To save visualized results on disk:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --output-root vis_results --draw-heatmap --det-cat-id=15
```

To save predicted results on disk:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --output-root vis_results --save-predictions --draw-heatmap --det-cat-id=15
```

To run demos on CPU:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --show --draw-heatmap --det-cat-id=15 --device cpu
```

### 2D Animal Pose Video Demo

Videos share the same interface with images. The difference is that the `${INPUT_PATH}` for videos can be the local path or **URL** link to video file.

For example,

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input demo/resources/<demo_dog.mp4> \
    --output-root vis_results --draw-heatmap --det-cat-id=16
```

<img src="https://user-images.githubusercontent.com/26127467/187655602-907db86e-710b-447a-8ec9-5b623d43d160.gif" height="500px" alt><br>

The original video can be downloaded from [Google Drive](https://drive.google.com/file/d/18d8K3wuUpKiDFHvOx0mh1TEwYwpOc5UO/view?usp=sharing).

### 2D Animal Pose Demo with Inferencer

The Inferencer provides a convenient interface for inference, allowing customization using model aliases instead of configuration files and checkpoint paths. It supports various input formats, including image paths, video paths, image folder paths, and webcams. Below is an example command:

```shell
python demo/inferencer_demo.py tests/data/ap10k \
    --pose2d animal --vis-out-dir vis_results/ap10k
```

This command infers all images located in `tests/data/ap10k` and saves the visualization results in the `vis_results/ap10k` directory.

<img src="https://user-images.githubusercontent.com/26127467/229789306-83ea56fa-12f2-4e27-9031-329d335ec26d.jpg" alt="Image 1" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229789324-7fef5688-422d-4663-a57c-d1e1d511e83c.jpg" alt="Image 2" height="200"/>

In addition, the Inferencer supports saving predicted poses. For more information, please refer to the [inferencer document](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface).

### Speed Up Inference

Some tips to speed up MMPose inference:

1. set `model.test_cfg.flip_test=False` in [animalpose_hrnet-w32](../../configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py#85).
2. use faster human bounding box detector, see [MMDetection](https://mmdetection.readthedocs.io/en/3.x/model_zoo.html).
