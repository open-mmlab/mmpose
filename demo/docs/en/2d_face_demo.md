## 2D Face Keypoint Demo

We provide a demo script to test a single image or video with face detectors and top-down pose estimators. Assume that you have already installed [mmdet](https://github.com/open-mmlab/mmdetection) with version >= 3.0.

**Face Bounding Box Model Preparation:** The pre-trained face box estimation model can be found in [mmdet model zoo](/demo/docs/en/mmdet_modelzoo.md#face-bounding-box-detection-models).

### 2D Face Image Demo

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} [--output-root ${OUTPUT_DIR}] \
    [--show] [--device ${GPU_ID or CPU}] [--save-predictions] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}] [--bbox-thr ${BBOX_SCORE_THR}]
```

The pre-trained face keypoint estimation models can be found from [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/face_2d_keypoint.html).
Take [aflw model](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) as an example:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --show --draw-heatmap
```

Visualization result:

<img src="https://user-images.githubusercontent.com/26127467/220538388-582ce90d-751a-40dd-ac06-3bc078b773a0.jpg" height="500px" alt><br>

If you use a heatmap-based model and set argument `--draw-heatmap`, the predicted heatmap will be visualized together with the keypoints.

To save visualized results on disk:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --draw-heatmap --output-root vis_results
```

To save the predicted results on disk, please specify `--save-predictions`.

To run demos on CPU:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --show --draw-heatmap --device=cpu
```

### 2D Face Video Demo

Videos share the same interface with images. The difference is that the `${INPUT_PATH}` for videos can be the local path or **URL** link to video file.

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input demo/resources/<demo_face.mp4> \
    --show --draw-heatmap --output-root vis_results
```

<img src="https://user-images.githubusercontent.com/26127467/220541430-6ade5a59-3d06-466a-a94d-00c82ff96a00.gif" height="500px" alt><br>

The original video can be downloaded from [Google Drive](https://drive.google.com/file/d/1kQt80t6w802b_vgVcmiV_QfcSJ3RWzmb/view?usp=sharing).

### 2D Face Pose Demo with Inferencer

The Inferencer provides a convenient interface for inference, allowing customization using model aliases instead of configuration files and checkpoint paths. It supports various input formats, including image paths, video paths, image folder paths, and webcams. Below is an example command:

```shell
python demo/inferencer_demo.py tests/data/wflw \
    --pose2d face --vis-out-dir vis_results/wflw --radius 1
```

This command infers all images located in `tests/data/wflw` and saves the visualization results in the `vis_results/wflw` directory.

<img src="https://user-images.githubusercontent.com/26127467/229793095-702f9d3b-461f-45bd-8535-d628e33bc907.jpg" alt="Image 1" width="400"/>

<img src="https://user-images.githubusercontent.com/26127467/229793121-9969f014-70da-40b5-8561-e21c3edd1aeb.jpg" alt="Image 2" width="400"/>

In addition, the Inferencer supports saving predicted poses. For more information, please refer to the [inferencer document](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface).

### Speed Up Inference

For 2D face keypoint estimation models, try to edit the config file. For example, set `model.test_cfg.flip_test=False` in line 90 of [aflw_hrnetv2](../../../configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py).
