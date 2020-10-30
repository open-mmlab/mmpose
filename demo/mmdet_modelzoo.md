### Human Bounding Box Detection Models

For human bounding box detection models, please download from [MMDetection Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html).
MMDetection provides 80-class COCO-pretrained models, which already includes the `person` category.

### Hand Bounding Box Detection Models

For hand bounding box detection, we simply train our hand box models on onehand10k dataset using MMDetection.

#### Hand detection results on OneHand10K test set.

| Arch  | Box AP |   ckpt | log |
| :-------------- | :-----------: | :------: | :------: |
| [Cascade_R-CNN X-101-64x4d-FPN](/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k.py)  |  0.817 |  [ckpt](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth) | [log](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k_20201030.log.json) |
