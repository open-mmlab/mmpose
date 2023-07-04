## Webcam Demo

The original Webcam API has been deprecated starting from version v1.1.0. Users now have the option to utilize either the Inferencer or the demo script for conducting pose estimation using webcam input.

### Webcam Demo with Inferencer

Users can utilize the MMPose Inferencer to estimate human poses in webcam inputs by executing the following command:

```shell
python demo/inferencer_demo.py webcam --pose2d 'human'
```

For additional information about the arguments of Inferencer, please refer to the [Inferencer Documentation](/docs/en/user_guides/inference.md).

### Webcam Demo with Demo Script

All of the demo scripts, except for `demo/image_demo.py`, support webcam input.

Take `demo/topdown_demo_with_mmdet.py` as example, users can utilize this script with webcam input by specifying **`--input webcam`** in the command:

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
    --input webcam --output-root=vis_results/demo \
    --show --draw-heatmap
```
