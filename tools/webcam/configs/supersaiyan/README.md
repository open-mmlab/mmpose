# Super Saiyan Effects

We can apply fun effects on videos with pose estimation results, like Super Saiyan transformation.

https://user-images.githubusercontent.com/11788150/150138076-2192079f-068a-4d43-bf27-2f1fd708cabc.mp4

## Instruction

### Get started

Launch the demo from the mmpose root directory:

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/supersaiyan/saiyan.py
```

### Hotkeys

| Hotkey | Function |
| -- | -- |
| s | Toggle the Super Saiyan effect on/off. |
| h | Show help information. |
| m | Show the monitoring information. |
| q | Exit. |

Note that the demo will automatically save the output video into a file `record.mp4`.

### Configuration

- **Choose a detection model**

Users can choose detection models from the [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/v2.20.0/model_zoo.html). Just set the `model_config` and `model_checkpoint` in the detector node accordingly, and the model will be automatically downloaded and loaded.

```python
# 'DetectorNode':
# This node performs object detection from the frame image using an
# MMDetection model.
dict(
    type='DetectorNode',
    name='Detector',
    model_config='demo/mmdetection_cfg/mask_rcnn_r50_fpn_2x_coco.py',
    model_checkpoint='https://download.openmmlab.com/'
    'mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/'
    'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392'
    '__segm_mAP-0.354_20200505_003907-3e542a40.pth',
    input_buffer='_input_',  # `_input_` is a runner-reserved buffer
    output_buffer='det_result'),
```

- **Choose a or more pose models**

In this demo we use two [top-down](https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap) pose estimation models for humans and animals respectively. Users can choose models from the [MMPose Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html). To apply different pose models on different instance types, you can add multiple pose estimator nodes with `cls_names` set accordingly.

```python
# 'TopDownPoseEstimatorNode':
# This node performs keypoint detection from the frame image using an
# MMPose top-down model. Detection results is needed.
dict(
    type='TopDownPoseEstimatorNode',
    name='Human Pose Estimator',
    model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
    'topdown_heatmap/coco-wholebody/'
    'vipnas_mbv3_coco_wholebody_256x192_dark.py',
    model_checkpoint='https://openmmlab-share.oss-cn-hangz'
    'hou.aliyuncs.com/mmpose/top_down/vipnas/vipnas_mbv3_co'
    'co_wholebody_256x192_dark-e2158108_20211205.pth',
    cls_names=['person'],
    input_buffer='det_result',
    output_buffer='human_pose')
```

- **Run the demo without GPU**

If you don't have GPU and CUDA in your device, the demo can run with only CPU by setting `device='cpu'` in all model nodes. For example:

```python
dict(
    type='DetectorNode',
    name='Detector',
    model_config='demo/mmdetection_cfg/mask_rcnn_r50_fpn_2x_coco.py',
    model_checkpoint='https://download.openmmlab.com/'
    'mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/'
    'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392'
    '__segm_mAP-0.354_20200505_003907-3e542a40.pth',
    device='cpu',
    input_buffer='_input_',  # `_input_` is a runner-reserved buffer
    output_buffer='det_result'),
```

- **Debug webcam and display**

You can launch the webcam runner with a debug config:

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/examples/test_camera.py
```
