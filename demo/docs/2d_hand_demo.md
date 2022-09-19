## 2D Hand Keypoint Demo

We provide a demo script to test a single image or video with hand detectors and top-down pose estimators. Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection) with version >= 3.0.

**Hand Box Model Preparation:** The pre-trained hand box estimation model can be found in [mmdet model zoo](/demo/docs/mmdet_modelzoo.md).

### 2D Hand Image Demo

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} [--output-root ${OUTPUT_DIR}] \
    [--show] [--device ${GPU_ID or CPU}] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}] [--bbox-thr ${BBOX_SCORE_THR}]

```

The pre-trained hand pose estimation model can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/1.x/model_zoo/hand_2d_keypoint.html).
Take [onehand10k model](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth) as an example:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --draw-heatmap
```

Visualization result:

<img src="https://user-images.githubusercontent.com/26127467/187664103-cfbe0c4e-5876-42f9-9023-5fb58ce00d7b.jpg" height="500px" alt><br>

If you use a heatmap-based model and set argument `--draw-heatmap`, the predicted heatmap will be visualized together with the keypoints.

To save visualized results on disk:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input tests/data/onehand10k/9.jpg \
    --output-root vis_results --show --draw-heatmap
```

To run demos on CPU:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --draw-heatmap  --device cpu
```

### 2D Hand Keypoints Video Demo

Videos share the same interface with images. The difference is that the `${INPUT_PATH}` for videos can be the local path or **URL** link to video file.

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input demo/resources/<demo_hand.mp4> \
    --output-root vis_results --show --draw-heatmap
```

<img src="https://user-images.githubusercontent.com/26127467/187665873-3ac836ec-8da5-45e1-8d78-c0abe962bd5e.gif" height="500px" alt><br>

The original video can be downloaded from [Github](https://raw.githubusercontent.com/open-mmlab/mmpose/master/tests/data/nvgesture/sk_color.avi).

### Speed Up Inference

For 2D hand keypoint estimation models, try to edit the config file. For example, set `model.test_cfg.flip_test=False` in [onehand10k_hrnetv2](../../configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py#90).
