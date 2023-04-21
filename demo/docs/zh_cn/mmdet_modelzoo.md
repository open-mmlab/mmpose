## Pre-trained Detection Models

### 人体 Bounding Box 检测模型

MMDetection 提供了基于 COCO 的包括 `person` 在内的 80 个类别的预训练模型，用户可前往 [MMDetection Model Zoo](https://mmdetection.readthedocs.io/zh_CN/3.x/model_zoo.html) 下载并将其用作人体 bounding box 识别模型。

### 手部 Bounding Box 检测模型

对于手部 bounding box 检测模型，我们提供了一个通过 MMDetection 基于 OneHand10K 数据库训练的模型。

#### 基于 OneHand10K 测试集的测试结果

| Arch                                                              | Box AP |                               ckpt                                |                               log                                |
| :---------------------------------------------------------------- | :----: | :---------------------------------------------------------------: | :--------------------------------------------------------------: |
| [Cascade_R-CNN X-101-64x4d-FPN-1class](/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py) | 0.817  | [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth) | [log](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k_20201030.log.json) |

### 脸部 Bounding Box 检测模型

对于脸部 bounding box 检测模型，我们提供了一个通过 MMDetection 基于 COCO-Face 数据库训练的 YOLOX 检测器。

#### 基于 COCO-face 测试集的测试结果

| Arch                                                            | Box AP |                                                  ckpt                                                  |
| :-------------------------------------------------------------- | :----: | :----------------------------------------------------------------------------------------------------: |
| [YOLOX-s](/demo/mmdetection_cfg/yolox-s_8xb8-300e_coco-face.py) | 0.408  | [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/yolo-x_8xb8-300e_coco-face_13274d7c.pth) |

### 动物 Bounding Box 检测模型

#### COCO animals

COCO 数据集内包括了 10 种常见的 `animal` 类型：

(14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe') 。

用户如果需要使用以上类别的动物检测模型，可以前往 [MMDetection Model Zoo](https://mmdetection.readthedocs.io/zh_CN/3.x/model_zoo.html) 下载。

#### 基于 MacaquePose 测试集的测试结果

| Arch                                                              | Box AP |                               ckpt                                |                               log                                |
| :---------------------------------------------------------------- | :----: | :---------------------------------------------------------------: | :--------------------------------------------------------------: |
| [Faster_R-CNN_Res50-FPN-1class](/demo/mmdetection_cfg/faster_rcnn_r50_fpn_1class.py) | 0.840  | [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/faster_rcnn_r50_fpn_1x_macaque-f64f2812_20210409.pth) | [log](https://download.openmmlab.com/mmpose/mmdet_pretrained/faster_rcnn_r50_fpn_1x_macaque_20210409.log.json) |
| [Cascade_R-CNN X-101-64x4d-FPN-1class](/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py) | 0.879  | [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_macaque-e45e36f5_20210409.pth) | [log](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_macaque_20210409.log.json) |
