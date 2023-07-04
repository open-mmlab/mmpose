## 摄像头推理

从版本 v1.1.0 开始，原来的摄像头 API 已被弃用。用户现在可以选择使用推理器（Inferencer）或 Demo 脚本从摄像头读取的视频中进行姿势估计。

### 使用推理器进行摄像头推理

用户可以通过执行以下命令来利用 MMPose Inferencer 对摄像头输入进行人体姿势估计：

```shell
python demo/inferencer_demo.py webcam --pose2d 'human'
```

有关推理器的参数详细信息，请参阅 [推理器文档](/docs/en/user_guides/inference.md)。

### 使用 Demo 脚本进行摄像头推理

除了 `demo/image_demo.py` 之外，所有的 Demo 脚本都支持摄像头输入。

以 `demo/topdown_demo_with_mmdet.py` 为例，用户可以通过在命令中指定 **`--input webcam`** 来使用该脚本对摄像头输入进行推理：

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
