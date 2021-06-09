## 3D Human Pose Demo

<img src="https://user-images.githubusercontent.com/15977946/118820606-02df2000-b8e9-11eb-9984-b9228101e780.gif" width="600px" alt><br>

### 3D Human Pose Two-stage Estimation Image Demo

#### Using ground truth 2D poses as the 1st stage (pose detection) result, and inference the 2nd stage (2D-to-3D lifting)

We provide a demo script to test on single images with a given ground-truth Json file.

```shell
python demo/body3d_two_stage_img_demo.py \
    ${MMPOSE_CONFIG_FILE_STAGE_2} \
    ${MMPOSE_CHECKPOINT_FILE_STAGE_2} \
    --json-file ${JSON_FILE} \
    --img-root ${IMG_ROOT} \
    --only-second-stage \
    [--show] \
    [--device ${GPU_ID or CPU}] \
    [--out-img-root ${OUTPUT_DIR}] \
    [--rebase-keypoint-height] \
    [--show-ground-truth]
```

Example:

```shell
python demo/body3d_two_stage_img_demo.py \
    configs/body/3d_kpt_sview_rgb_img/pose_lift/h36m/simplebaseline3d_h36m.py \
    https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth \
    --json-file tests/data/h36m/h36m_coco.json \
    --img-root tests/data/h36m \
    --camera-param-file tests/data/h36m/cameras.pkl \
    --only-second-stage \
    --out-img-root vis_results \
    --rebase-keypoint-height \
    --show-ground-truth
```
