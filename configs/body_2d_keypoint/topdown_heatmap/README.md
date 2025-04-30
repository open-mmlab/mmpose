# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the likelihood of being a keypoint, following the paradigm introduced in [Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522977-5f355832-e9c1-442f-a34f-9d24fb0aefa8.png" height=400>
</div>

## Results and Models

### COCO Dataset

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|      Model      | Input Size |  AP   |  AR   |                Details and Download                 |
| :-------------: | :--------: | :---: | :---: | :-------------------------------------------------: |
|    ViTPose-h    |  256x192   | 0.790 | 0.840 |      [vitpose_coco.md](./coco/vitpose_coco.md)      |
|  HRNet-w48+UDP  |  256x192   | 0.768 | 0.817 |    [hrnet_udp_coco.md](./coco/hrnet_udp_coco.md)    |
|   MSPN 4-stg    |  256x192   | 0.765 | 0.826 |         [mspn_coco.md](./coco/mspn_coco.md)         |
| HRNet-w48+Dark  |  256x192   | 0.764 | 0.814 |   [hrnet_dark_coco.md](./coco/hrnet_dark_coco.md)   |
|    HRNet-w48    |  256x192   | 0.756 | 0.809 |        [hrnet_coco.md](./coco/hrnet_coco.md)        |
|   HRFormer-B    |  256x192   | 0.754 | 0.807 |     [hrformer_coco.md](./coco/hrformer_coco.md)     |
|    RSN-50-3x    |  256x192   | 0.750 | 0.814 |          [rsn_coco.md](./coco/rsn_coco.md)          |
|    CSPNeXt-l    |  256x192   | 0.750 | 0.800 |  [cspnext_udp_coco.md](./coco/cspnext_udp_coco.md)  |
|    HRNet-w32    |  256x192   | 0.749 | 0.804 |        [hrnet_coco.md](./coco/hrnet_coco.md)        |
|     Swin-L      |  256x192   | 0.743 | 0.798 |         [swin_coco.md](./coco/swin_coco.md)         |
|    ViTPose-s    |  256x192   | 0.739 | 0.792 |      [vitpose_coco.md](./coco/vitpose_coco.md)      |
|   HRFormer-S    |  256x192   | 0.738 | 0.793 |     [hrformer_coco.md](./coco/hrformer_coco.md)     |
|     Swin-B      |  256x192   | 0.737 | 0.794 |         [swin_coco.md](./coco/swin_coco.md)         |
|  SEResNet-101   |  256x192   | 0.734 | 0.790 |     [seresnet_coco.md](./coco/seresnet_coco.md)     |
|    SCNet-101    |  256x192   | 0.733 | 0.789 |        [scnet_coco.md](./coco/scnet_coco.md)        |
| ResNet-101+Dark |  256x192   | 0.733 | 0.786 |  [resnet_dark_coco.md](./coco/resnet_dark_coco.md)  |
|    CSPNeXt-m    |  256x192   | 0.732 | 0.785 |  [cspnext_udp_coco.md](./coco/cspnext_udp_coco.md)  |
|  ResNetV1d-101  |  256x192   | 0.732 | 0.785 |    [resnetv1d_coco.md](./coco/resnetv1d_coco.md)    |
|   SEResNet-50   |  256x192   | 0.729 | 0.784 |     [seresnet_coco.md](./coco/seresnet_coco.md)     |
|    SCNet-50     |  256x192   | 0.728 | 0.784 |        [scnet_coco.md](./coco/scnet_coco.md)        |
|   ResNet-101    |  256x192   | 0.726 | 0.783 |       [resnet_coco.md](./coco/resnet_coco.md)       |
|   ResNeXt-101   |  256x192   | 0.726 | 0.781 |      [resnext_coco.md](./coco/resnext_coco.md)      |
|  HourglassNet   |  256x256   | 0.726 | 0.780 |    [hourglass_coco.md](./coco/hourglass_coco.md)    |
|   ResNeSt-101   |  256x192   | 0.725 | 0.781 |      [resnest_coco.md](./coco/resnest_coco.md)      |
|     RSN-50      |  256x192   | 0.724 | 0.790 |          [rsn_coco.md](./coco/rsn_coco.md)          |
|     Swin-T      |  256x192   | 0.724 | 0.782 |         [swin_coco.md](./coco/swin_coco.md)         |
|   MSPN 1-stg    |  256x192   | 0.723 | 0.788 |         [mspn_coco.md](./coco/mspn_coco.md)         |
|  ResNetV1d-50   |  256x192   | 0.722 | 0.777 |    [resnetv1d_coco.md](./coco/resnetv1d_coco.md)    |
|   ResNeSt-50    |  256x192   | 0.720 | 0.775 |      [resnest_coco.md](./coco/resnest_coco.md)      |
|    ResNet-50    |  256x192   | 0.718 | 0.774 |       [resnet_coco.md](./coco/resnet_coco.md)       |
|   ResNeXt-50    |  256x192   | 0.715 | 0.771 |      [resnext_coco.md](./coco/resnext_coco.md)      |
|      PVT-S      |  256x192   | 0.714 | 0.773 |          [pvt_coco.md](./coco/pvt_coco.md)          |
|    CSPNeXt-s    |  256x192   | 0.697 | 0.753 |  [cspnext_udp_coco.md](./coco/cspnext_udp_coco.md)  |
|  LiteHRNet-30   |  256x192   | 0.676 | 0.736 |    [litehrnet_coco.md](./coco/litehrnet_coco.md)    |
|  CSPNeXt-tiny   |  256x192   | 0.665 | 0.723 |  [cspnext_udp_coco.md](./coco/cspnext_udp_coco.md)  |
|  MobileNet-v2   |  256x192   | 0.648 | 0.709 |  [mobilenetv2_coco.md](./coco/mobilenetv2_coco.md)  |
|  LiteHRNet-18   |  256x192   | 0.642 | 0.705 |    [litehrnet_coco.md](./coco/litehrnet_coco.md)    |
|       CPM       |  256x192   | 0.627 | 0.689 |          [cpm_coco.md](./coco/cpm_coco.md)          |
|  ShuffleNet-v2  |  256x192   | 0.602 | 0.668 | [shufflenetv2_coco.md](./coco/shufflenetv2_coco.md) |
|  ShuffleNet-v1  |  256x192   | 0.587 | 0.654 | [shufflenetv1_coco.md](./coco/shufflenetv1_coco.md) |
|     AlexNet     |  256x192   | 0.448 | 0.521 |      [alexnet_coco.md](./coco/alexnet_coco.md)      |

### MPII Dataset

|     Model      | Input Size | PCKh@0.5 | PCKh@0.1 |                Details and Download                 |
| :------------: | :--------: | :------: | :------: | :-------------------------------------------------: |
| HRNet-w48+Dark |  256x256   |  0.905   |  0.360   |   [hrnet_dark_mpii.md](./mpii/hrnet_dark_mpii.md)   |
|   HRNet-w48    |  256x256   |  0.902   |  0.303   |     [hrnet_mpii.md](./mpii/cspnext_udp_mpii.md)     |
|   HRNet-w48    |  256x256   |  0.901   |  0.337   |        [hrnet_mpii.md](./mpii/hrnet_mpii.md)        |
|   HRNet-w32    |  256x256   |  0.900   |  0.334   |        [hrnet_mpii.md](./mpii/hrnet_mpii.md)        |
|  HourglassNet  |  256x256   |  0.889   |  0.317   |    [hourglass_mpii.md](./mpii/hourglass_mpii.md)    |
|   ResNet-152   |  256x256   |  0.889   |  0.303   |       [resnet_mpii.md](./mpii/resnet_mpii.md)       |
| ResNetV1d-152  |  256x256   |  0.888   |  0.300   |    [resnetv1d_mpii.md](./mpii/resnetv1d_mpii.md)    |
|    SCNet-50    |  256x256   |  0.888   |  0.290   |        [scnet_mpii.md](./mpii/scnet_mpii.md)        |
|  ResNeXt-152   |  256x256   |  0.887   |  0.294   |      [resnext_mpii.md](./mpii/resnext_mpii.md)      |
|  SEResNet-50   |  256x256   |  0.884   |  0.292   |     [seresnet_mpii.md](./mpii/seresnet_mpii.md)     |
|   ResNet-50    |  256x256   |  0.882   |  0.286   |       [resnet_mpii.md](./mpii/resnet_mpii.md)       |
|  ResNetV1d-50  |  256x256   |  0.881   |  0.290   |    [resnetv1d_mpii.md](./mpii/resnetv1d_mpii.md)    |
|      CPM       | 368x368\*  |  0.876   |  0.285   |          [cpm_mpii.md](./mpii/cpm_mpii.md)          |
|  LiteHRNet-30  |  256x256   |  0.869   |  0.271   |    [litehrnet_mpii.md](./mpii/litehrnet_mpii.md)    |
|  LiteHRNet-18  |  256x256   |  0.859   |  0.260   |    [litehrnet_mpii.md](./mpii/litehrnet_mpii.md)    |
|  MobileNet-v2  |  256x256   |  0.854   |  0.234   |  [mobilenetv2_mpii.md](./mpii/mobilenetv2_mpii.md)  |
| ShuffleNet-v2  |  256x256   |  0.828   |  0.205   | [shufflenetv2_mpii.md](./mpii/shufflenetv2_mpii.md) |
| ShuffleNet-v1  |  256x256   |  0.824   |  0.195   | [shufflenetv1_mpii.md](./mpii/shufflenetv1_mpii.md) |

### CrowdPose Dataset

Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

|   Model    | Input Size |  AP   |  AR   |                    Details and Download                    |
| :--------: | :--------: | :---: | :---: | :--------------------------------------------------------: |
| HRNet-w32  |  256x192   | 0.675 | 0.816 |    [hrnet_crowdpose.md](./crowdpose/hrnet_crowdpose.md)    |
| CSPNeXt-m  |  256x192   | 0.662 | 0.755 | [hrnet_crowdpose.md](./crowdpose/cspnext_udp_crowdpose.md) |
| ResNet-101 |  256x192   | 0.647 | 0.800 |   [resnet_crowdpose.md](./crowdpose/resnet_crowdpose.md)   |
| HRNet-w32  |  256x192   | 0.637 | 0.785 |   [resnet_crowdpose.md](./crowdpose/resnet_crowdpose.md)   |

### AIC Dataset

Results on AIC val set with ground-truth bounding boxes.

|   Model    | Input Size |  AP   |  AR   |         Details and Download         |
| :--------: | :--------: | :---: | :---: | :----------------------------------: |
| HRNet-w32  |  256x192   | 0.323 | 0.366 |  [hrnet_aic.md](./aic/hrnet_aic.md)  |
| ResNet-101 |  256x192   | 0.294 | 0.337 | [resnet_aic.md](./aic/resnet_aic.md) |

### JHMDB Dataset

|   Model   | Input Size | PCK(norm. by person size) | PCK (norm. by torso size) |            Details and Download            |
| :-------: | :--------: | :-----------------------: | :-----------------------: | :----------------------------------------: |
| ResNet-50 |  256x256   |           96.0            |           80.1            | [resnet_jhmdb.md](./jhmdb/resnet_jhmdb.md) |
|    CPM    |  368x368   |           89.8            |           65.7            |    [cpm_jhmdb.md](./jhmdb/cpm_jhmdb.md)    |

### PoseTrack2018 Dataset

Results on PoseTrack2018 val with ground-truth bounding boxes.

|   Model   | Input Size |  AP  |                     Details and Download                     |
| :-------: | :--------: | :--: | :----------------------------------------------------------: |
| HRNet-w48 |  256x192   | 84.6 |  [hrnet_posetrack18.md](./posetrack18/hrnet_posetrack18.md)  |
| HRNet-w32 |  256x192   | 83.4 |  [hrnet_posetrack18.md](./posetrack18/hrnet_posetrack18.md)  |
| ResNet-50 |  256x192   | 81.2 | [resnet_posetrack18.md](./posetrack18/resnet_posetrack18.md) |

### Human-Art Dataset

Results on Human-Art validation dataset with detector having human AP of 56.2 on Human-Art validation dataset

|   Model   | Input Size |  AP   |  AR   |                 Details and Download                  |
| :-------: | :--------: | :---: | :---: | :---------------------------------------------------: |
| ViTPose-s |  256x192   | 0.381 | 0.448 | [vitpose_humanart.md](./humanart/vitpose_humanart.md) |
| ViTPose-b |  256x192   | 0.410 | 0.475 | [vitpose_humanart.md](./humanart/vitpose_humanart.md) |

Results on Human-Art validation dataset with ground-truth bounding-box

|   Model   | Input Size |  AP   |  AR   |                 Details and Download                  |
| :-------: | :--------: | :---: | :---: | :---------------------------------------------------: |
| ViTPose-s |  256x192   | 0.738 | 0.768 | [vitpose_humanart.md](./humanart/vitpose_humanart.md) |
| ViTPose-b |  256x192   | 0.759 | 0.790 | [vitpose_humanart.md](./humanart/vitpose_humanart.md) |
