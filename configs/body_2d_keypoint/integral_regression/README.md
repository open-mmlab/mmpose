# Top-down integral-regression-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. At the 2nd stage, integral regression based methods use a simple integral operation relates and unifies the heatmap and joint regression differentiably, thus obtain the keypoint coordinates given the features extracted from the bounding box area, following the paradigm introduced in [Integral Human Pose Regression](https://arxiv.org/abs/1711.08229).

## Results and Models

### COCO Dataset

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|        Model         | Input Size |  AP   |  AR   |                 Details and Download                  |
| :------------------: | :--------: | :---: | :---: | :---------------------------------------------------: |
| ResNet-50+Debias-IPR |  256x256   | 0.675 | 0.765 | [resnet_debias_coco.md](./coco/resnet_debias_coco.md) |
|    ResNet-50+DSNT    |  256x256   | 0.674 | 0.764 |   [resnet_dsnt_coco.md](./coco/resnet_dsnt_coco.md)   |
|    ResNet-50+IPR     |  256x256   | 0.633 | 0.730 |    [resnet_ipr_coco.md](./coco/resnet_ipr_coco.md)    |
