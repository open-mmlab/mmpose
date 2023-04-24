## Webcam Demo

我们提供了同时支持人体和动物的识别和 2D 姿态预估 webcam demo 工具，用户也可以用这个脚本在姿态预测结果上加入譬如大眼和戴墨镜等好玩的特效。

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/124059525-ce20c580-da5d-11eb-8e4a-2d96cd31fe9f.gif" width="600px" alt><br>
</div>

### Get started

脚本使用方式很简单，直接在 MMPose 根路径使用：

```shell
# 使用 GPU
python demo/webcam_api_demo.py

# 仅使用 CPU
python demo/webcam_api_demo.py --cpu
```

该命令会使用默认的 `demo/webcam_cfg/human_pose.py` 作为配置文件，用户可以自行指定别的配置：

```shell
python demo/webcam_api_demo.py --config demo/webcam_cfg/human_pose.py
```

### Hotkeys

| Hotkey | Function                              |
| ------ | ------------------------------------- |
| v      | Toggle the pose visualization on/off. |
| h      | Show help information.                |
| m      | Show the monitoring information.      |
| q      | Exit.                                 |

注意：脚本会自动将实时结果保存成一个名为 `webcam_api_demo.mp4` 的视频文件。

### 配置使用

这里我们只进行一些基本的说明，更多的信息可以直接参考对应的配置文件。

- **设置检测模型**

  用户可以直接使用 [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/3.x/model_zoo.html) 里的识别模型，需要注意的是确保配置文件中的 DetectorNode 里的 `model_config` 和 `model_checkpoint` 需要对应起来，这样模型就会被自动下载和加载，例如：

  ```python
  # 'DetectorNode':
  # This node performs object detection from the frame image using an
  # MMDetection model.
  dict(
      type='DetectorNode',
      name='detector',
      model_config='demo/mmdetection_cfg/'
      'ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py',
      model_checkpoint='https://download.openmmlab.com'
      '/mmdetection/v2.0/ssd/'
      'ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_'
      'scratch_600e_coco_20210629_110627-974d9307.pth',
      input_buffer='_input_',
      output_buffer='det_result'),
  ```

- **设置姿态预估模型**

  这里我们用两个 [top-down](https://github.com/open-mmlab/mmpose/tree/latest/configs/body_2d_keypoint/topdown_heatmap) 结构的人体和动物姿态预估模型进行演示。用户可以自由使用 [MMPose Model Zoo](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo/body_2d_keypoint.html) 里的模型。需要注意的是，更换模型后用户需要在对应的 pose estimate node 里添加或修改对应的 `cls_names` ，例如：

  ```python
  # 'TopdownPoseEstimatorNode':
  # This node performs keypoint detection from the frame image using an
  # MMPose top-down model. Detection results is needed.
  dict(
      type='TopdownPoseEstimatorNode',
      name='human pose estimator',
      model_config='configs/wholebody_2d_keypoint/'
      'topdown_heatmap/coco-wholebody/'
      'td-hm_vipnas-mbv3_dark-8xb64-210e_coco-wholebody-256x192.py',
      model_checkpoint='https://download.openmmlab.com/mmpose/'
      'top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark'
      '-e2158108_20211205.pth',
      labels=['person'],
      input_buffer='det_result',
      output_buffer='human_pose'),
  dict(
      type='TopdownPoseEstimatorNode',
      name='animal pose estimator',
      model_config='configs/animal_2d_keypoint/topdown_heatmap/'
      'animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py',
      model_checkpoint='https://download.openmmlab.com/mmpose/animal/'
      'hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth',
      labels=['cat', 'dog', 'horse', 'sheep', 'cow'],
      input_buffer='human_pose',
      output_buffer='animal_pose'),
  ```

- **使用本地视频文件**

  如果想直接使用本地的视频文件，用户只需要把文件路径设置到 `camera_id` 就行。

- **本机没有摄像头怎么办**

  用户可以在自己手机安装上一些 app 就能替代摄像头，例如 [Camo](https://reincubate.com/camo/) 和 [DroidCam](https://www.dev47apps.com/) 。

- **测试摄像头和显示器连接**

  使用如下命令就能完成检测：

  ```shell
  python demo/webcam_api_demo.py --config demo/webcam_cfg/test_camera.py
  ```
