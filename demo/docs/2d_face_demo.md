## 2D Face Keypoint Demo

We provide a demo script to test a single image or video with face detectors and top-down pose estimators, Please install `face_recognition` before running the demo, by:

```
pip install face_recognition
```

For more details, please refer to [face_recognition](https://github.com/ageitgey/face_recognition).

### 2D Face Image Demo

```shell
python demo/topdown_face_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} [--output-root ${OUTPUT_DIR}] \
    [--show] [--device ${GPU_ID or CPU}] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

The pre-trained face keypoint estimation models can be found from [model zoo](https://mmpose.readthedocs.io/en/1.x/model_zoo/face_2d_keypoint.html).
Take [aflw model](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) as an example:

```shell
python demo/topdown_face_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --show --draw-heatmap
```

Visualization result:

<img src="https://user-images.githubusercontent.com/87690686/190857851-8d5afe60-fadf-4aa8-9a1c-5b32aaec7c79.jpg" height="500px" alt><br>

If you use a heatmap-based model and set argument `--draw-heatmap`, the predicted heatmap will be visualized together with the keypoints.

To save visualized results on disk:

```shell
python demo/topdown_face_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --draw-heatmap --output-root vis_results
```

To run demos on CPU:

```shell
python demo/topdown_face_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --show --draw-heatmap --device=cpu
```

### 2D Face Video Demo

Videos share the same interface with images. The difference is that the `${INPUT_PATH}` for videos can be the local path or **URL** link to video file.

```shell
python demo/topdown_face_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input demo/resources/<demo_face.mp4> \
    --show --draw-heatmap --output-root vis_results
```

<img src="https://user-images.githubusercontent.com/87690686/190858159-b224b06a-7d34-4716-a8bc-4d127a39b90c.gif" height="500px" alt><br>

The original video can be downloaded from [Google Drive](https://drive.google.com/file/d/1kQt80t6w802b_vgVcmiV_QfcSJ3RWzmb/view?usp=sharing).

### Speed Up Inference

For 2D face keypoint estimation models, try to edit the config file. For example, set `model.test_cfg.flip_test=False` in [aflw_hrnetv2](../../configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py#90).
