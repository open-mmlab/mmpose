## 2D Human Pose Demo

本节我们继续使用 demo 脚本演示 2D 人体关键点的识别。同样的，用户仍要确保开发环境已经安装了 3.0 版本以上的 [mmdet](https://github.com/open-mmlab/mmdetection) 。

### 2D 人体姿态 Top-Down 图片检测

#### 使用整张图片作为输入进行检测

此时输入的整张图片会被当作 bounding box 使用。

```shell
python demo/image_demo.py \
    ${IMG_FILE} ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --out-file ${OUTPUT_FILE} \
    [--device ${GPU_ID or CPU}] \
    [--draw_heatmap]
```

如果使用了 heatmap-based 模型同时设置了 `--draw-heatmap` ，预测的热图也会跟随关键点一同可视化出来。

用户可以在 [model zoo](https://mmpose.readthedocs.io/zh_CN/latest/model_zoo/body_2d_keypoint.html) 获取预训练好的关键点识别模型。

这里我们用 [coco model](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) 来进行演示：

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
```

使用 CPU 推理：

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --out-file vis_results.jpg \
    --draw-heatmap \
    --device=cpu
```

可视化结果如下：

<img src="https://user-images.githubusercontent.com/87690686/187824033-2cce0f55-034a-4127-82e2-52744178bc32.jpg" height="500px" alt><br>

#### 使用 MMDet 做人体 bounding box 检测

使用 MMDet 进行识别的命令如下所示：

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} \
    [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--show] [--draw-heatmap] [--device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR}] [--kpt-thr ${KPT_SCORE_THR}]
```

结合我们的具体例子：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --input tests/data/coco/000000197388.jpg --show --draw-heatmap \
    --output-root vis_results/
```

可视化结果如下：

<img src="https://user-images.githubusercontent.com/87690686/187824368-1f1631c3-52bf-4b45-bf9a-a70cd6551e1a.jpg" height="500px" alt><br>

想要本地保存识别结果，用户需要加上 `--save-predictions` 。

### 2D 人体姿态 Top-Down 视频检测

我们的脚本同样支持视频作为输入，由 MMDet 完成人体检测后 MMPose 完成 Top-Down 的姿态预估，视频推理时 `${INPUT_PATH}` 既可以是本地视频文件的路径也可以是视频文件的 **URL** 地址。

例如：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
    --input tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4 \
    --output-root=vis_results/demo --show --draw-heatmap
```

### 2D 人体姿态 Bottom-Up 图片和视频识别检测

除了 Top-Down ，我们也支持 Bottom-Up 不依赖人体识别器的人体姿态预估识别，使用方式如下：

```shell
python demo/bottomup_demo.py \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} \
    [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--show] [--device ${GPU_ID or CPU}] \
    [--kpt-thr ${KPT_SCORE_THR}]
```

结合具体示例如下：

```shell
python demo/bottomup_demo.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth \
    --input tests/data/coco/000000197388.jpg --output-root=vis_results \
    --show --save-predictions
```

其可视化结果如图所示：

<img src="https://user-images.githubusercontent.com/26127467/207224032-a8dab45d-39e4-4b4e-80e0-3c71a64f5f39.jpg" height="300px" alt><br>

### 使用 Inferencer 进行 2D 人体姿态识别检测

Inferencer 提供一个更便捷的推理接口，使得用户可以绕过模型的配置文件和 checkpoint 路径直接使用 model aliases ，支持包括图片路径、视频路径、图片文件夹路径和 webcams 在内的多种输入方式，例如可以这样使用：

```shell
python demo/inferencer_demo.py \
    tests/data/posetrack18/videos/000001_mpiinew_test/000001_mpiinew_test.mp4 \
    --pose2d human --vis-out-dir vis_results/posetrack18
```

该命令会对输入的 `tests/data/posetrack18` 下的视频进行推理并且把可视化结果存入 `vis_results/posetrack18` 文件夹下。

<img src="https://user-images.githubusercontent.com/26127467/229831445-44c9662b-edc5-4ef0-92a6-13558f0906cc.gif" alt="Image 1" height="300"/>

Inferencer 支持保存姿态的检测结果，具体的使用可参考 [inferencer document](https://mmpose.readthedocs.io/zh_CN/dev-1.x/user_guides/inference.html) 。

### 加速推理

对于 top-down 结构的模型，用户可以通过修改配置文件来加速，更多具体例子可以参考：

1. 设置 `model.test_cfg.flip_test=False`，如 [topdown-res50](/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_res50_8xb64-210e_coco-256x192.py#L56) 所示。
2. 使用更快的人体 bounding box 检测器，可参考 [MMDetection](https://mmdetection.readthedocs.io/zh_CN/3.x/model_zoo.html) 。
