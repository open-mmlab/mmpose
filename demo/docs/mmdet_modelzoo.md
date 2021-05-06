## Pre-trained Detection Models

### Human Bounding Box Detection Models

For human bounding box detection models, please download from [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).
MMDetection provides 80-class COCO-pretrained models, which already includes the `person` category.

### Hand Bounding Box Detection Models

For hand bounding box detection, we simply train our hand box models on onehand10k dataset using MMDetection.

#### Hand detection results on OneHand10K test set

| Arch  | Box AP |   ckpt | log |
| :-------------- | :-----------: | :------: | :------: |
| [Cascade_R-CNN X-101-64x4d-FPN-1class](/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py)  |  0.817 |  [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth) | [log](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k_20201030.log.json) |

### Animal Bounding Box Detection Models

#### COCO animals

In COCO dataset, there are 80 object categories, including 10 common `animal` categories (16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe')
For animals in the categories, please download from [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).

#### Macaque detection results on MacaquePose test set

| Arch  | Box AP |   ckpt | log |
| :-------------- | :-----------: | :------: | :------: |
| [Faster_R-CNN_Res50-FPN-1class](/demo/mmdetection_cfg/faster_rcnn_r50_fpn_1class.py)  |  0.840 |  [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/faster_rcnn_r50_fpn_1x_macaque-f64f2812_20210409.pth) | [log](https://download.openmmlab.com/mmpose/mmdet_pretrained/faster_rcnn_r50_fpn_1x_macaque_20210409.log.json) |
| [Cascade_R-CNN X-101-64x4d-FPN-1class](/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py)  |  0.879 |  [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_macaque-e45e36f5_20210409.pth) | [log](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_macaque_20210409.log.json) |
