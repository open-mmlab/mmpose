## 2D Animal Pose Demo

本系列文档我们会来介绍如何使用提供了的脚本进行完成基本的推理 demo ，本节先介绍如何对 top-down 结构和动物的 2D 姿态进行单张图片和视频推理，请确保你已经安装了 3.0 以上版本的 [MMDetection](https://github.com/open-mmlab/mmdetection) 。

### 2D 动物图片姿态识别推理

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} --det-cat-id ${DET_CAT_ID} \
    [--show] [--output-root ${OUTPUT_DIR}] [--save-predictions] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}] [--bbox-thr ${BBOX_SCORE_THR}] \
    [--device ${GPU_ID or CPU}]
```

用户可以在 [model zoo](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo/animal_2d_keypoint.html) 获取预训练好的关键点识别模型。

这里我们用 [animalpose model](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth) 来进行演示：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --show --draw-heatmap --det-cat-id=15
```

可视化结果如下：

<img src="https://user-images.githubusercontent.com/26127467/187644168-5915551a-0876-4b85-9454-7f92c84ba6fb.jpeg" height="500px" alt><br>

如果使用了 heatmap-based 模型同时设置了 `--draw-heatmap` ，预测的热图也会跟随关键点一同可视化出来。

`--det-cat-id=15` 参数用来指定模型只检测 `cat` 类型，这是基于 COCO 数据集的数据。

**COCO 数据集动物信息**

COCO 数据集共包含 80 个类别，其中有 10 种常见动物，类别如下：

(14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe')

对于其他类型的动物，我们也提供了一些训练好的动物检测模型，用户可以前往 [detection model zoo](/demo/docs/zh_cn/mmdet_modelzoo.md) 下载。

如果想本地保存可视化结果可使用如下命令：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --output-root vis_results --draw-heatmap --det-cat-id=15
```

如果想本地保存预测结果，需要使用 `--save-predictions` 。

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --output-root vis_results --save-predictions --draw-heatmap --det-cat-id=15
```

仅使用 CPU：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --show --draw-heatmap --det-cat-id=15 --device cpu
```

### 2D 动物视频姿态识别推理

视频和图片使用了同样的接口，区别在于视频推理时 `${INPUT_PATH}` 既可以是本地视频文件的路径也可以是视频文件的 **URL** 地址。

例如：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input demo/resources/<demo_dog.mp4> \
    --output-root vis_results --draw-heatmap --det-cat-id=16
```

<img src="https://user-images.githubusercontent.com/26127467/187655602-907db86e-710b-447a-8ec9-5b623d43d160.gif" height="500px" alt><br>

这段视频可以在 [Google Drive](https://drive.google.com/file/d/18d8K3wuUpKiDFHvOx0mh1TEwYwpOc5UO/view?usp=sharing) 下载。

### 使用 Inferencer 进行 2D 动物姿态识别推理

Inferencer 提供一个更便捷的推理接口，使得用户可以绕过模型的配置文件和 checkpoint 路径直接使用 model aliases ，支持包括图片路径、视频路径、图片文件夹路径和 webcams 在内的多种输入方式，例如可以这样使用：

```shell
python demo/inferencer_demo.py tests/data/ap10k \
    --pose2d animal --vis-out-dir vis_results/ap10k
```

该命令会对输入的 `tests/data/ap10k` 下所有的图片进行推理并且把可视化结果都存入 `vis_results/ap10k` 文件夹下。

<img src="https://user-images.githubusercontent.com/26127467/229789306-83ea56fa-12f2-4e27-9031-329d335ec26d.jpg" alt="Image 1" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229789324-7fef5688-422d-4663-a57c-d1e1d511e83c.jpg" alt="Image 2" height="200"/>

Inferencer 同样支持保存预测结果，更多的信息可以参考 [Inferencer 文档](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface) 。

### 加速推理

用户可以通过修改配置文件来加速，更多具体例子可以参考：

1. 设置 `model.test_cfg.flip_test=False`，如 [animalpose_hrnet-w32](../../configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py#85) 所示。
2. 使用更快的 bounding box 检测器，可参考 [MMDetection](https://mmdetection.readthedocs.io/zh_CN/3.x/model_zoo.html) 。
