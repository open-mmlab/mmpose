## 3D Mesh Demo

<img src="https://user-images.githubusercontent.com/28900607/124615414-d33fa380-dea7-11eb-9ec4-a01d0931e028.gif" width="600px" alt><br>

### 3D Mesh Recovery Demo

We provide a demo script to recover human 3D mesh from a single image.

```shell
python demo/mesh_img_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --json-file ${JSON_FILE} \
    --img-root ${IMG_ROOT} \
    [--show] \
    [--device ${GPU_ID or CPU}] \
    [--out-img-root ${OUTPUT_DIR}]
```

Example:

```shell
python demo/mesh_img_demo.py \
    configs/body/3d_mesh_sview_rgb_img/hmr/mixed/res50_mixed_224x224.py \
    https://download.openmmlab.com/mmpose/mesh/hmr/hmr_mesh_224x224-c21e8229_20201015.pth \
    --json-file tests/data/h36m/h36m_coco.json \
    --img-root tests/data/h36m \
    --out-img-root vis_results
```
