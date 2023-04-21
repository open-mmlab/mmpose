## 2D Human Whole-Body Pose Demo

### 2D 人体全身姿态 Top-Down 图片识别

#### 使用整张图片作为输入进行检测

此时输入的整张图片会被当作 bounding box 使用。

```shell
python demo/image_demo.py \
    ${IMG_FILE} ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --out-file ${OUTPUT_FILE} \
    [--device ${GPU_ID or CPU}] \
    [--draw_heatmap]
```

用户可以在 [model zoo](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo/2d_wholebody_keypoint.html) 获取预训练好的关键点识别模型。

这里我们用 [coco-wholebody_vipnas_res50_dark](https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth) 来进行演示：

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-res50_dark-8xb64-210e_coco-wholebody-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth \
    --out-file vis_results.jpg
```

使用 CPU 推理：

```shell
python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_vipnas-res50_dark-8xb64-210e_coco-wholebody-256x192.py \
    https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192_dark-67c0ce35_20211112.pth \
    --out-file vis_results.jpg \
    --device=cpu
```

#### 使用 MMDet 进行人体 bounding box 检测

使用 MMDet 进行识别的命令格式如下：

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} \
    [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--show] [--draw-heatmap] [--device ${GPU_ID or CPU}] \
    [--bbox-thr ${BBOX_SCORE_THR}] [--kpt-thr ${KPT_SCORE_THR}]
```

具体可例如：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --input tests/data/coco/000000196141.jpg \
    --output-root vis_results/ --show
```

想要本地保存识别结果，用户需要加上 `--save-predictions` 。

### 2D 人体全身姿态 Top-Down 视频识别检测

我们的脚本同样支持视频作为输入，由 MMDet 完成人体检测后 MMPose 完成 Top-Down 的姿态预估。

例如：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --input https://user-images.githubusercontent.com/87690686/137440639-fb08603d-9a35-474e-b65f-46b5c06b68d6.mp4 \
    --output-root vis_results/ --show
```

可视化结果如下：

<img src="https://user-images.githubusercontent.com/87690686/190854069-634e1142-d13c-4863-9930-1120057ca77e.gif" height="350px" alt><br>

### 使用 Inferencer 进行 2D 人体全身姿态识别

Inferencer 提供一个更便捷的推理接口，使得用户可以绕过模型的配置文件和 checkpoint 路径直接使用 model aliases ，支持包括图片路径、视频路径、图片文件夹路径和 webcams 在内的多种输入方式，例如可以这样使用：

```shell
python demo/inferencer_demo.py tests/data/crowdpose \
    --pose2d wholebody --vis-out-dir vis_results/crowdpose
```

该命令会对输入的 `tests/data/crowdpose` 下所有图片进行推理并且把可视化结果存入 `vis_results/crowdpose` 文件夹下。

<img src="https://user-images.githubusercontent.com/26127467/229832887-31edb6d5-bcf0-44a4-a66f-9d523061a6e9.jpg" alt="Image 1" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229832908-bc82dbc9-5e43-4800-acc7-a7da85a653c7.jpg" alt="Image 2" height="200"/>

Inferencer 支持保存姿态的检测结果，具体的使用可参考 [Inferencer 文档](https://mmpose.readthedocs.io/zh_CN/dev-1.x/user_guides/#inferencer-a-unified-inference-interface) 。

### 加速推理

对于 top-down 结构的模型，用户可以通过修改配置文件来加速，更多具体例子可以参考：

1. 设置 `model.test_cfg.flip_test=False`，用户可参考 [pose_hrnet_w48_dark+](/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py#L90) 。
2. 使用更快的人体 bounding box 检测器，如 [MMDetection](https://mmdetection.readthedocs.io/zh_CN/3.x/model_zoo.html) 。
