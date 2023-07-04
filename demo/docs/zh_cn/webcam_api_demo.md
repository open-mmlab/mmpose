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
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
    --input webcam --output-root=vis_results/demo \
    --show --draw-heatmap
```
