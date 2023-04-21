## 2D Hand Keypoint Demo

本节我们继续通过 demo 脚本演示对单张图片或者视频的 2D 手部关键点的识别。同样的，用户仍要确保开发环境已经安装了 3.0 版本以上的 [MMDetection](https://github.com/open-mmlab/mmdetection) 。

我们在 [mmdet model zoo](/demo/docs/zh_cn/mmdet_modelzoo.md#手部-bounding-box-识别模型) 提供了预训练好的手部 Bounding Box 预测模型，用户可以前往下载。

### 2D 手部图片关键点识别

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} [--output-root ${OUTPUT_DIR}] \
    [--show] [--device ${GPU_ID or CPU}] [--save-predictions] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}] [--bbox-thr ${BBOX_SCORE_THR}]
```

用户可以在 [model zoo](https://mmpose.readthedocs.io/zh_CN/dev-1.x/model_zoo/hand_2d_keypoint.html) 获取预训练好的关键点识别模型。

这里我们用 [onehand10k model](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth) 来进行演示：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --draw-heatmap
```

可视化结果如下：

<img src="https://user-images.githubusercontent.com/26127467/187664103-cfbe0c4e-5876-42f9-9023-5fb58ce00d7b.jpg" height="500px" alt><br>

如果使用了 heatmap-based 模型同时设置了 `--draw-heatmap` ，预测的热图也会跟随关键点一同可视化出来。

如果想本地保存可视化结果可使用如下命令：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input tests/data/onehand10k/9.jpg \
    --output-root vis_results --show --draw-heatmap
```

如果想本地保存预测结果，需要添加 `--save-predictions` 。

如果想用 CPU 进行 demo 需添加 `--device cpu` ：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input tests/data/onehand10k/9.jpg \
    --show --draw-heatmap  --device cpu
```

### 2D 手部视频关键点识别推理

视频和图片使用了同样的接口，区别在于视频推理时 `${INPUT_PATH}` 既可以是本地视频文件的路径也可以是视频文件的 **URL** 地址。

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth \
    configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py \
    https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_onehand10k_256x256-30bc9c6b_20210330.pth \
    --input demo/resources/<demo_hand.mp4> \
    --output-root vis_results --show --draw-heatmap
```

<img src="https://user-images.githubusercontent.com/26127467/187665873-3ac836ec-8da5-45e1-8d78-c0abe962bd5e.gif" height="500px" alt><br>

这段视频可以在 [Google Drive](https://raw.githubusercontent.com/open-mmlab/mmpose/master/tests/data/nvgesture/sk_color.avi) 下载到。

### 使用 Inferencer 进行 2D 手部关键点识别推理

Inferencer 提供一个更便捷的推理接口，使得用户可以绕过模型的配置文件和 checkpoint 路径直接使用 model aliases ，支持包括图片路径、视频路径、图片文件夹路径和 webcams 在内的多种输入方式，例如可以这样使用：

```shell
python demo/inferencer_demo.py tests/data/onehand10k \
    --pose2d hand --vis-out-dir vis_results/onehand10k \
    --bbox-thr 0.5 --kpt-thr 0.05
```

该命令会对输入的 `tests/data/onehand10k` 下所有的图片进行推理并且把可视化结果都存入 `vis_results/onehand10k` 文件夹下。

<img src="https://user-images.githubusercontent.com/26127467/229824447-b444e92d-9b5b-4a50-9a32-68be3ff8c527.jpg" alt="Image 1" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229824466-6ae47a40-70a6-451d-94ee-4ffc34204a9c.jpg" alt="Image 2" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229824477-679201c3-1e0b-45fe-b0c7-bab67b245a10.jpg" alt="Image 3" height="200"/> <img src="https://user-images.githubusercontent.com/26127467/229824488-bd874362-7401-41a5-8209-51bad1563a11.jpg" alt="Image 4" height="200"/>

除此之外， Inferencer 也支持保存预测的姿态结果。具体信息可在 [Inferencer 文档](https://mmpose.readthedocs.io/zh_CN/dev-1.x/user_guides/inference.html) 查看。

### 加速推理

对于 2D 手部关键点预测模型，用户可以通过修改配置文件中的 `model.test_cfg.flip_test=False` 来加速，如 [onehand10k_hrnetv2](../../configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_hrnetv2-w18_8xb64-210e_onehand10k-256x256.py#90) 所示。
