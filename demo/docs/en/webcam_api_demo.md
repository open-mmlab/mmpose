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
# inference with webcam
python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input webcam \
    --show
```
