# Top-down SimCC-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. At the 2nd stage, SimCC  based methods reformulate human pose estimation as two classification tasks for horizontal and vertical coordinates, and uniformly divide each pixel into several bins, thus obtain the keypoint coordinates given the features extracted from the bounding box area, following the paradigm introduced in [SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation](https://arxiv.org/abs/2107.03332).

<div align=center>
<img src="https://user-images.githubusercontent.com/13503330/189811385-6395d118-055b-4bad-89e8-f84ffa2c2aa6.png">
</div>

## Results and Models

### COCO Dataset

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|             Model             | Input Size |  AP   |  AR   |               Details and Download                |
| :---------------------------: | :--------: | :---: | :---: | :-----------------------------------------------: |
|        ResNet-50+SimCC        |  384x288   | 0.735 | 0.790 |      [resnet_coco.md](./coco/resnet_coco.md)      |
|        ResNet-50+SimCC        |  256x192   | 0.721 | 0.781 |      [resnet_coco.md](./coco/resnet_coco.md)      |
|  S-ViPNAS-MobileNet-V3+SimCC  |  256x192   | 0.695 | 0.755 |      [vipnas_coco.md](./coco/vipnas_coco.md)      |
| MobileNet-V2+SimCC(wo/deconv) |  256x192   | 0.620 | 0.678 | [mobilenetv2_coco.md](./coco/mobilenetv2_coco.md) |
