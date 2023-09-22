## 3D Hand Demo

<img src="https://user-images.githubusercontent.com/28900607/121288285-b8fcbf00-c915-11eb-98e4-ba846de12987.gif" width="600px" alt><br>

### 3D Hand Estimation Image Demo

#### Using gt hand bounding boxes as input

We provide a demo script to test a single image, given gt json file.

```shell
python demo/hand3d_internet_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_FILE} \
    --output-root ${OUTPUT_ROOT} \
    [--save-predictions] \
    [--gt-joints-file ${GT_JOINTS_FILE}]\
    [--disable-rebase-keypoint] \
    [--show] \
    [--device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_THR}] \
    [--show-kpt-idx] \
    [--show-interval] \
    [--radius ${RADIUS}] \
    [--thickness ${THICKNESS}]
```

The pre-trained hand pose estimation model can be downloaded from [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_3d_keypoint.html).
Take [internet model](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth) as an example:

```shell
python demo/hand3d_internet_demo.py \
    configs/hand_3d_keypoint/internet/interhand3d/internet_res50_4xb16-20e_interhand3d-256x256.py \
    https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth \
    --input tests/data/interhand2.6m/image69148.jpg \
    --save-predictions \
    --output-root vis_results
```
