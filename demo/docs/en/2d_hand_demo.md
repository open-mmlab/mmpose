## 2D Hand Keypoint Demo

We provide a demo script to test a single image or video with hand detectors and top-down pose estimators. Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection) with version >= 3.0.

**Hand Box Model Preparation:** The pre-trained hand box estimation model can be found in [mmdet model zoo](/demo/docs/en/mmdet_modelzoo.md#hand-bounding-box-detection-models).

### 2D Hand Image Demo

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} [--output-root ${OUTPUT_DIR}] \
    [--show] [--device ${GPU_ID or CPU}] [--save-predictions] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}] [--bbox-thr ${BBOX_SCORE_THR}]
```

The pre-trained hand pose estimation model can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_2d_keypoint.html).
Take [onehand10k model](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth) as an example:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --draw-heatmap
```

Visualization result:

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/3a2794cb-8071-4b9e-9498-c0bb46eb381e" height="500px" alt><br>

If you use a heatmap-based model and set argument `--draw-heatmap`, the predicted heatmap will be visualized together with the keypoints.

To save visualized results on disk:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input tests/data/onehand10k/9.jpg \
    --output-root vis_results --show --draw-heatmap
```

To save the predicted results on disk, please specify `--save-predictions`.

To run demos on CPU:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --draw-heatmap  --device cpu
```

### 2D Hand Keypoints Video Demo

Videos share the same interface with images. The difference is that the `${INPUT_PATH}` for videos can be the local path or **URL** link to video file.

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input data/tests_data_nvgesture_sk_color.avi \
    --output-root vis_results --kpt-thr 0.1
```

<img src="https://github.com/open-mmlab/mmpose/assets/26127467/558e8211-d7ca-4e04-b690-6c455e805ed7" height="300px" alt><br>

The original video can be downloaded from [Github](https://raw.githubusercontent.com/open-mmlab/mmpose/master/tests/data/nvgesture/sk_color.avi).

### 2D Hand Keypoints Demo with Inferencer

The Inferencer provides a convenient interface for inference, allowing customization using model aliases instead of configuration files and checkpoint paths. It supports various input formats, including image paths, video paths, image folder paths, and webcams. Below is an example command:

```shell
python demo/inferencer_demo.py tests/data/onehand10k \
    --pose2d hand --vis-out-dir vis_results/onehand10k \
    --bbox-thr 0.5 --kpt-thr 0.05
```

This command infers all images located in `tests/data/onehand10k` and saves the visualization results in the `vis_results/onehand10k` directory.

<img src="https://user-images.githubusercontent.com/26127467/229824447-b444e92d-9b5b-4a50-9a32-68be3ff8c527.jpg" alt="Image 1" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229824466-6ae47a40-70a6-451d-94ee-4ffc34204a9c.jpg" alt="Image 2" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229824477-679201c3-1e0b-45fe-b0c7-bab67b245a10.jpg" alt="Image 3" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229824488-bd874362-7401-41a5-8209-51bad1563a11.jpg" alt="Image 4" height="200"/>

In addition, the Inferencer supports saving predicted poses. For more information, please refer to the [inferencer document](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface).

### Speed Up Inference

For 2D hand keypoint estimation models, try to edit the config file. For example, set `model.test_cfg.flip_test=False` in [onehand10k_hrnetv2](../../configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py#90).
