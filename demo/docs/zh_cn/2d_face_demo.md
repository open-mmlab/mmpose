## 2D Face Keypoint Demo

本节我们继续演示如何使用 demo 脚本进行 2D 脸部关键点的识别。同样的，用户仍要确保开发环境已经安装了 3.0 版本以上的 [MMdetection](https://github.com/open-mmlab/mmdetection) 。

我们在 [mmdet model zoo](/demo/docs/zh_cn/mmdet_modelzoo.md#脸部-bounding-box-检测模型) 提供了一个预训练好的脸部 Bounding Box 预测模型，用户可以前往下载。

### 2D 脸部图片关键点识别推理

```shell
python demo/topdown_demo_with_mmdet.py \
    ${MMDET_CONFIG_FILE} ${MMDET_CHECKPOINT_FILE} \
    ${MMPOSE_CONFIG_FILE} ${MMPOSE_CHECKPOINT_FILE} \
    --input ${INPUT_PATH} [--output-root ${OUTPUT_DIR}] \
    [--show] [--device ${GPU_ID or CPU}] [--save-predictions] \
    [--draw-heatmap ${DRAW_HEATMAP}] [--radius ${KPT_RADIUS}] \
    [--kpt-thr ${KPT_SCORE_THR}] [--bbox-thr ${BBOX_SCORE_THR}]
```

用户可以在 [model zoo](https://mmpose.readthedocs.io/en/dev-1.x/model_zoo/face_2d_keypoint.html) 获取预训练好的脸部关键点识别模型。

这里我们用 [aflw model](https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth) 来进行演示：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --show --draw-heatmap
```

可视化结果如下图所示：

<img src="https://user-images.githubusercontent.com/26127467/220538388-582ce90d-751a-40dd-ac06-3bc078b773a0.jpg" height="500px" alt><br>

如果使用了 heatmap-based 模型同时设置了 `--draw-heatmap` ，预测的热图也会跟随关键点一同可视化出来。

如果想本地保存可视化结果可使用如下命令：

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input tests/data/cofw/001766.jpg \
    --draw-heatmap --output-root vis_results
```

### 2D 脸部视频关键点识别推理

视频和图片使用了同样的接口，区别在于视频推理时 `${INPUT_PATH}` 既可以是本地视频文件的路径也可以是视频文件的 **URL** 地址。

```shell
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py \
    https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth \
    configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py \
    https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth \
    --input demo/resources/<demo_face.mp4> \
    --show --draw-heatmap --output-root vis_results
```

<img src="https://user-images.githubusercontent.com/26127467/220541430-6ade5a59-3d06-466a-a94d-00c82ff96a00.gif" height="500px" alt><br>

这段视频可以在 [Google Drive](https://drive.google.com/file/d/1kQt80t6w802b_vgVcmiV_QfcSJ3RWzmb/view?usp=sharing) 下载。

### 使用 Inferencer 进行 2D 脸部关键点识别推理

Inferencer 提供一个更便捷的推理接口，使得用户可以绕过模型的配置文件和 checkpoint 路径直接使用 model aliases ，支持包括图片路径、视频路径、图片文件夹路径和 webcams 在内的多种输入方式，例如可以这样使用：

```shell
python demo/inferencer_demo.py tests/data/wflw \
    --pose2d face --vis-out-dir vis_results/wflw --radius 1
```

该命令会对输入的 `tests/data/wflw` 下所有的图片进行推理并且把可视化结果都存入 `vis_results/wflw` 文件夹下。

<img src="https://user-images.githubusercontent.com/26127467/229793095-702f9d3b-461f-45bd-8535-d628e33bc907.jpg" alt="Image 1" width="400"/>

<img src="https://user-images.githubusercontent.com/26127467/229793121-9969f014-70da-40b5-8561-e21c3edd1aeb.jpg" alt="Image 2" width="400"/>

除此之外， Inferencer 也支持保存预测的姿态结果。具体信息可在 [Inferencer 文档](https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface) 查看。

### 加速推理

对于 2D 脸部关键点预测模型，用户可以通过修改配置文件中的 `model.test_cfg.flip_test=False` 来加速，例如 [aflw_hrnetv2](../../../configs/face_2d_keypoint/topdown_heatmap/aflw/td-hm_hrnetv2-w18_8xb64-60e_aflw-256x256.py) 中的第 90 行。
