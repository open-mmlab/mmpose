## 3D Hand Demo

<img src="https://user-images.githubusercontent.com/28900607/121288285-b8fcbf00-c915-11eb-98e4-ba846de12987.gif" width="600px" alt><br>

### 3D Hand Estimation Image Demo

#### Using gt hand bounding boxes as input

We provide a demo script to test a single image, given gt json file.

```shell
python demo/interhand3d_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --json-file ${JSON_FILE} \
    --img-root ${IMG_ROOT} \
    [--camera-param-file ${CAMERA_PARAM_FILE}] \
    [--gt-joints-file ${GT_JOINTS_FILE}]\
    [--show] \
    [--device ${GPU_ID or CPU}] \
    [--out-img-root ${OUTPUT_DIR}] \
    [--rebase-keypoint-height] \
    [--show-ground-truth]
```

Example with gt keypoints and camera parameters:

```shell
python demo/interhand3d_img_demo.py \
    configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth \
    --json-file tests/data/interhand2.6m/test_interhand2.6m_data.json \
    --img-root tests/data/interhand2.6m \
    --camera-param-file tests/data/interhand2.6m/test_interhand2.6m_camera.json \
    --gt-joints-file tests/data/interhand2.6m/test_interhand2.6m_joint_3d.json \
    --out-img-root vis_results \
    --rebase-keypoint-height \
    --show-ground-truth
```

Example without gt keypoints and camera parameters:

```shell
python demo/interhand3d_img_demo.py \
    configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3d_all_256x256-b9c1cf4c_20210506.pth \
    --json-file tests/data/interhand2.6m/test_interhand2.6m_data.json \
    --img-root tests/data/interhand2.6m \
    --out-img-root vis_results \
    --rebase-keypoint-height
```
