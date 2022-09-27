## Webcam Demo

We provide a webcam demo tool which integrartes detection and 2D pose estimation for humans and animals. It can also apply fun effects like putting on sunglasses or enlarging the eyes, based on the pose estimation results.

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/124059525-ce20c580-da5d-11eb-8e4a-2d96cd31fe9f.gif" width="600px" alt><br>
</div>

### Get started

Launch the demo from the mmpose root directory:

```shell
# Run webcam demo with GPU
python demo/webcam_demo.py

# Run webcam demo with CPU
python demo/webcam_demo.py --cpu
```

The command above will use the default config file `demo/webcam_cfg/pose_estimation.py`. You can also specify the config file in the command:

```shell
# Use the config "pose_tracking.py" for higher infererence speed
python demo/webcam_demo.py --config demo/webcam_cfg/pose_tracking.py
```

### Hotkeys

| Hotkey | Function                                                         |
| ------ | ---------------------------------------------------------------- |
| v      | Toggle the pose visualization on/off.                            |
| s      | Toggle the sunglasses effect on/off. (NA for `pose_trakcing.py`) |
| b      | Toggle the big-eye effect on/off. (NA for `pose_trakcing.py`)    |
| h      | Show help information.                                           |
| m      | Show the monitoring information.                                 |
| q      | Exit.                                                            |

Note that the demo will automatically save the output video into a file `webcam_demo.mp4`.

### Usage and configuarations

Detailed configurations can be found in the config file.

- **Configure detection models**
  Users can choose detection models from the [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/v2.20.0/model_zoo.html). Just set the `model_config` and `model_checkpoint` in the detector node accordingly, and the model will be automatically downloaded and loaded.

  ```python
  # 'DetectorNode':
  # This node performs object detection from the frame image using an
  # MMDetection model.
  dict(
      type='DetectorNode',
      name='detector',
      model_config='demo/mmdetection_cfg/'
      'ssdlite_mobilenetv2_scratch_600e_coco.py',
      model_checkpoint='https://download.openmmlab.com'
      '/mmdetection/v2.0/ssd/'
      'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
      'scratch_600e_coco_20210629_110627-974d9307.pth',
      input_buffer='_input_',
      output_buffer='det_result')
  ```

- **Configure pose estimation models**
  In this demo we use two [top-down](https://github.com/open-mmlab/mmpose/tree/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap) pose estimation models for humans and animals respectively. Users can choose models from the [MMPose Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html). To apply different pose models on different instance types, you can add multiple pose estimator nodes with `cls_names` set accordingly.

  ```python
  # 'TopDownPoseEstimatorNode':
  # This node performs keypoint detection from the frame image using an
  # MMPose top-down model. Detection results is needed.
  dict(
      type='TopDownPoseEstimatorNode',
      name='human pose estimator',
      model_config='configs/wholebody/2d_kpt_sview_rgb_img/'
      'topdown_heatmap/coco-wholebody/'
      'vipnas_mbv3_coco_wholebody_256x192_dark.py',
      model_checkpoint='https://openmmlab-share.oss-cn-hangz'
      'hou.aliyuncs.com/mmpose/top_down/vipnas/vipnas_mbv3_co'
      'co_wholebody_256x192_dark-e2158108_20211205.pth',
      labels=['person'],
      input_buffer='det_result',
      output_buffer='human_pose'),
  dict(
      type='TopDownPoseEstimatorNode',
      name='animal pose estimator',
      model_config='configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap'
      '/animalpose/hrnet_w32_animalpose_256x256.py',
      model_checkpoint='https://download.openmmlab.com/mmpose/animal/'
      'hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
      labels=['cat', 'dog', 'horse', 'sheep', 'cow'],
      input_buffer='human_pose',
      output_buffer='animal_pose')
  ```

- **Run the demo on a local video file**
  You can use local video files as the demo input by set `camera_id` to the file path.

- **The computer doesn't have a camera?**
  A smart phone can serve as a webcam via apps like [Camo](https://reincubate.com/camo/) or [DroidCam](https://www.dev47apps.com/).

- **Test the camera and display**
  Run follow command for a quick test of video capturing and displaying.

  ```shell
  python demo/webcam_demo.py --config demo/webcam_cfg/test_camera.py
  ```
