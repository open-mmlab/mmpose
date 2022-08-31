## 2D Face Keypoint Demo

<img src="https://user-images.githubusercontent.com/11788150/109144943-ccd44900-779c-11eb-9e9d-8682e7629654.gif" width="600px" alt><br>

We provide a demo script to test a single image or video with top-down pose estimators and face detectors.Please install `face_recognition` before running the demo, by `pip install face_recognition`. For more details, please refer to https://github.com/ageitgey/face_recognition.

### 2D Face Image Demo

```shell
python demo/topdown_face_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --img-root ${IMG_ROOT} --json-file ${JSON_FILE} \
    --input ${INPUT_PATH} [--output-root ${OUTPUT_DIR}] \
    [--show] [--device ${GPU_ID or CPU}] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

The pre-trained face keypoint estimation model can be found from [model zoo](https://mmpose.readthedocs.io/en/1.x/modelzoo_tasks/face.html).
Take [aflw model](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) as an example:

```shell
python demo/top_down_img_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --show --draw-heatmap
```

Visualization result:

<img src="https://user-images.githubusercontent.com/26127467/187676149-97b36b55-94f3-4c2a-b831-c6d1b839f029.jpg" height="500px" alt><br>

If you use a heatmap-based model and set argument `--draw-heatmap`, the predicted heatmap will be visualized together with the keypoints.

To save visualized results on disk:

```shell
python demo/top_down_img_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --draw-heatmap --output-root vis_results
```

To run demos on CPU:

```shell
python demo/top_down_img_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --show --draw-heatmap --device=cpu
```

### 2D Face Video Demo

Videos share same interface with images. The difference is, the `${INPUT_PATH}` for videos can be the local path or **URL** link to video file.

```shell
python demo/top_down_img_demo.py \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input demo/resources/<demo_face.mp4> \
    --show --draw-heatmap --output-root vis_results
```

<img src="https://user-images.githubusercontent.com/26127467/187677697-ba44030e-e4d8-4cc4-b112-39d94bb3ef45.gif" height="500px" alt><br>

The original video can be downloaded from [Google Drive](https://drive.google.com/file/d/1kQt80t6w802b_vgVcmiV_QfcSJ3RWzmb/view?usp=sharing).

### Speed Up Inference

For 2D face keypoint estimation models, try to edit the config file. For example, set `model.test_cfg.flip_test=False` in [aflw_hrnetv2](../../configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py).
