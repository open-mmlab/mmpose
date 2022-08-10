## 3D Multiview Human Pose Demo

### 3D Multiview Human Pose Estimation Image Demo

#### VoxelPose

We provide a demo script to test on multiview images with given camera parameters.
To run the demo:

```shell
python demo/body3d_multiview_detect_and_regress_img_demo.py \
    ${MMPOSE_CONFIG_FILE} \
    ${MMPOSE_CHECKPOINT_FILE} \
    --out-img-root ${OUT_IMG_ROOT} \
    --camera-param-file ${CAMERA_FILE} \
    [--img-root ${IMG_ROOT}] \
    [--visualize-single-view ${VIS_SINGLE_IMG}] \
    [--device ${GPU_ID or CPU}] \
    [--out-img-root ${OUTPUT_DIR}]
```

Example:

```shell
python demo/body3d_multiview_detect_and_regress_img_demo.py \
    configs/body/3d_kpt_mview_rgb_img/voxelpose/panoptic/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5.py \
    https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5-545c150e_20211103.pth \
    --out-img-root vis_results \
    --camera-param-file tests/data/panoptic_body3d/demo/camera_parameters.json \
    --visualize-single-view
```

##### Data Preparation

Currently, we only support CMU Panoptic data format. Users can leave the argument `--img-root` unset to automatically download our [default demo data](https://download.openmmlab.com/mmpose/demo/panoptic_body3d_demo.tar) (~6M). Users can also use custom data, which should be organized as follow:

```text
├── ${IMG_ROOT}
    │── camera_parameters.json
    │── camera0
        │-- 0.jpg
        │-- ...
    │── camera1
    │── ...
```

The camera parameters should be a dictionary that include a key "cameras". Under the key "cameras"
should be a list of dictionaries containing the camera parameters. Each dictionary under the list
should include a key "name", the value of which is the directory name of images of a certain camera view.

```text
{
 "cameras": [
  {"name": "camera0", ...},
  {"name": "camera1", ...},
  ...
}
```
