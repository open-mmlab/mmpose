# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection and pose estimation.

They perform object detection first, followed by single-object pose estimation given object bounding boxes.
Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint.

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522977-5f355832-e9c1-442f-a34f-9d24fb0aefa8.png" height=400>
</div>

## Results and Models

### COCO Dataset

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|    Backbone     | Input Size |  AP   |   AR    |                Details and Download                 |
| :-------------: | :--------: | :---: | :-----: | :-------------------------------------------------: |
|     AlexNet     |  256x192   | 0.448 | 0.0.521 |        [cpm_coco.md](./coco/alexnet_coco.md)        |
|       CPM       |  256x192   | 0.623 | 0.0.685 |          [cpm_coco.md](./coco/cpm_coco.md)          |
|  HourglassNet   |  256x256   | 0.726 |  0.780  |    [hourglass_coco.md](./coco/hourglass_coco.md)    |
|   HRFormer-S    |  256x192   | 0.738 |  0.793  |     [hrformer_coco.md](./coco/hrformer_coco.md)     |
|   HRFormer-B    |  256x192   | 0.754 |  0.807  |     [hrformer_coco.md](./coco/hrformer_coco.md)     |
|    HRNet-w32    |  256x192   | 0.746 |  0.799  |        [hrnet_coco.md](./coco/hrnet_coco.md)        |
|    HRNet-w48    |  256x192   | 0.756 |  0.806  |        [hrnet_coco.md](./coco/hrnet_coco.md)        |
| HRNet-w48+Dark  |  256x192   | 0.764 |  0.814  |   [hrnet_dark_coco.md](./coco/hrnet_dark_coco.md)   |
|  HRNet-w48+UDP  |  256x192   | 0.773 |  0.820  |    [hrnet_udp_coco.md](./coco/hrnet_udp_coco.md)    |
|  LiteHRNet-18   |  256x192   | 0.642 |  0.705  |    [litehrnet_coco.md](./coco/litehrnet_coco.md)    |
|  LiteHRNet-30   |  256x192   | 0.676 |  0.736  |    [litehrnet_coco.md](./coco/litehrnet_coco.md)    |
|  MobileNet-v2   |  256x192   | 0.647 |  0.708  |  [mobilenetv2_coco.md](./coco/mobilenetv2_coco.md)  |
|   MSPN 1-stg    |  256x192   | 0.723 |  0.788  |         [mspn_coco.md](./coco/mspn_coco.md)         |
|   MSPN 4-stg    |  256x192   | 0.765 |  0.826  |         [mspn_coco.md](./coco/mspn_coco.md)         |
|      PVT-S      |  256x192   | 0.714 |  0.773  |          [pvt_coco.md](./coco/pvt_coco.md)          |
|    ResNet-50    |  256x192   | 0.718 |  0.773  |       [resnet_coco.md](./coco/resnet_coco.md)       |
|   ResNet-101    |  256x192   | 0.726 |  0.781  |       [resnet_coco.md](./coco/resnet_coco.md)       |
| ResNet-101+Dark |  256x192   | 0.732 |  0.786  |    [resnet_coco.md](./coco/resnet_dark_coco.md)     |
|   ResNeSt-50    |  256x192   | 0.720 |  0.775  |      [resnest_coco.md](./coco/resnest_coco.md)      |
|   ResNeSt-101   |  256x192   | 0.725 |  0.781  |      [resnest_coco.md](./coco/resnest_coco.md)      |
|  ResNetV1d-50   |  256x192   | 0.722 |  0.777  |    [resnetv1d_coco.md](./coco/resnetv1d_coco.md)    |
|  ResNetV1d-101  |  256x192   | 0.731 |  0.786  |    [resnetv1d_coco.md](./coco/resnetv1d_coco.md)    |
|   ResNeXt-50    |  256x192   | 0.715 |  0.771  |      [resnext_coco.md](./coco/resnext_coco.md)      |
|   ResNeXt-101   |  256x192   | 0.726 |  0.781  |      [resnext_coco.md](./coco/resnext_coco.md)      |
|     RCN-50      |  256x192   | 0.726 |  0.781  |          [rsn_coco.md](./coco/rsn_coco.md)          |
|    RCN-50-3x    |  256x192   | 0.749 |  0.812  |          [rsn_coco.md](./coco/rsn_coco.md)          |
|    SCNet-50     |  256x192   | 0.728 |  0.784  |        [scnet_coco.md](./coco/scnet_coco.md)        |
|    SCNet-101    |  256x192   | 0.733 |  0.789  |        [scnet_coco.md](./coco/scnet_coco.md)        |
|   SEResNet-50   |  256x192   | 0.729 |  0.784  |     [seresnet_coco.md](./coco/seresnet_coco.md)     |
|  SEResNet-101   |  256x192   | 0.734 |  0.790  |     [seresnet_coco.md](./coco/seresnet_coco.md)     |
|  ShuffleNet-v1  |  256x192   | 0.586 |  0.651  | [shufflenetv1_coco.md](./coco/shufflenetv1_coco.md) |
|  ShuffleNet-v2  |  256x192   | 0.598 |  0.664  | [shufflenetv2_coco.md](./coco/shufflenetv2_coco.md) |
|     Swin-T      |  256x192   | 0.724 |  0.782  |         [swin_coco.md](./coco/swin_coco.md)         |
|     Swin-B      |  256x192   | 0.737 |  0.794  |         [swin_coco.md](./coco/swin_coco.md)         |
|     Swin-L      |  256x192   | 0.743 |  0.798  |         [swin_coco.md](./coco/swin_coco.md)         |

### MPII Dataset
