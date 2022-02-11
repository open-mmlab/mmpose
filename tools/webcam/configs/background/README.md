# Matting Effects

We can apply background matting to the videos.

## Instruction

### Get started

Launch the demo from the mmpose root directory:

```shell
python tools/webcam/run_webcam.py --config tools/webcam/configs/background/background.py
```

### Hotkeys

| Hotkey | Function |
| -- | -- |
| b | Toggle the background matting effect on/off. |
| h | Show help information. |
| m | Show the monitoring information. |
| q | Exit. |

Note that the demo will automatically save the output video into a file `record.mp4`.

### Configuration

- **Choose a detection model**

Users can choose detection models from the [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/v2.20.0/model_zoo.html). Just set the `model_config` and `model_checkpoint` in the detector node accordingly, and the model will be automatically downloaded and loaded.
Note that in order to perform background matting, the model should be able to produce segmentation masks.

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
