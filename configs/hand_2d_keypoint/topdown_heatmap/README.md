# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the likelihood of being a keypoint, following the paradigm introduced in [Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522977-5f355832-e9c1-442f-a34f-9d24fb0aefa8.png" height=400>
</div>

## Results and Models

### COCO-WholeBody-Hand Dataset

Results on COCO-WholeBody-Hand val set

|      Model       | Input Size | PCK@0.2 |  AUC  | EPE  |                                       Details and Download                                       |
| :--------------: | :--------: | :-----: | :---: | :--: | :----------------------------------------------------------------------------------------------: |
| HRNetv2-w18+Dark |  256x256   |  0.814  | 0.840 | 4.37 | [hrnetv2_dark_coco_wholebody_hand.md](./coco_wholebody_hand/hrnetv2_dark_coco_wholebody_hand.md) |
|   HRNetv2-w18    |  256x256   |  0.813  | 0.840 | 4.39 |      [hrnetv2_coco_wholebody_hand.md](./coco_wholebody_hand/hrnetv2_coco_wholebody_hand.md)      |
|   HourglassNet   |  256x256   |  0.804  | 0.835 | 4.54 |    [hourglass_coco_wholebody_hand.md](./coco_wholebody_hand/hourglass_coco_wholebody_hand.md)    |
|     SCNet-50     |  256x256   |  0.803  | 0.834 | 4.55 |        [scnet_coco_wholebody_hand.md](./coco_wholebody_hand/scnet_coco_wholebody_hand.md)        |
|    ResNet-50     |  256x256   |  0.800  | 0.833 | 4.64 |       [resnet_coco_wholebody_hand.md](./coco_wholebody_hand/resnet_coco_wholebody_hand.md)       |
|   LiteHRNet-18   |  256x256   |  0.795  | 0.830 | 4.77 |    [litehrnet_coco_wholebody_hand.md](./coco_wholebody_hand/litehrnet_coco_wholebody_hand.md)    |
|   MobileNet-v2   |  256x256   |  0.795  | 0.829 | 4.77 |  [mobilenetv2_coco_wholebody_hand.md](./coco_wholebody_hand/mobilenetv2_coco_wholebody_hand.md)  |

### FreiHand Dataset

Results on FreiHand val & test set

|   Model   | Input Size | PCK@0.2 |  AUC  | EPE  |                   Details and Download                    |
| :-------: | :--------: | :-----: | :---: | :--: | :-------------------------------------------------------: |
| ResNet-50 |  224x224   |  0.999  | 0.868 | 3.27 | [resnet_freihand2d.md](./freihand2d/resnet_freihand2d.md) |

### OneHand10K Dataset

Results on OneHand10K val set

|      Model       | Input Size | PCK@0.2 |  AUC  |  EPE  |                         Details and Download                          |
| :--------------: | :--------: | :-----: | :---: | :---: | :-------------------------------------------------------------------: |
| HRNetv2-w18+Dark |  256x256   |  0.990  | 0.572 | 23.96 | [hrnetv2_dark_onehand10k.md](./onehand10k/hrnetv2_dark_onehand10k.md) |
| HRNetv2-w18+UDP  |  256x256   |  0.990  | 0.571 | 23.88 |  [hrnetv2_udp_onehand10k.md](./onehand10k/hrnetv2_udp_onehand10k.md)  |
|   HRNetv2-w18    |  256x256   |  0.990  | 0.567 | 24.26 |      [hrnetv2_onehand10k.md](./onehand10k/hrnetv2_onehand10k.md)      |
|    ResNet-50     |  256x256   |  0.989  | 0.555 | 25.16 |       [resnet_onehand10k.md](./onehand10k/resnet_onehand10k.md)       |
|   MobileNet-v2   |  256x256   |  0.986  | 0.537 | 28.56 |  [mobilenetv2_onehand10k.md](./onehand10k/mobilenetv2_onehand10k.md)  |

### RHD Dataset

Results on RHD test set

|      Model       | Input Size | PCK@0.2 |  AUC  | EPE  |                  Details and Download                  |
| :--------------: | :--------: | :-----: | :---: | :--: | :----------------------------------------------------: |
| HRNetv2-w18+Dark |  256x256   |  0.992  | 0.903 | 2.18 | [hrnetv2_dark_rhd2d.md](./rhd2d/hrnetv2_dark_rhd2d.md) |
| HRNetv2-w18+UDP  |  256x256   |  0.992  | 0.902 | 2.19 |  [hrnetv2_udp_rhd2d.md](./rhd2d/hrnetv2_udp_rhd2d.md)  |
|   HRNetv2-w18    |  256x256   |  0.992  | 0.902 | 2.21 |      [hrnetv2_rhd2d.md](./rhd2d/hrnetv2_rhd2d.md)      |
|    ResNet-50     |  256x256   |  0.991  | 0.898 | 2.32 |       [resnet_rhd2d.md](./rhd2d/resnet_rhd2d.md)       |
|   MobileNet-v2   |  256x256   |  0.985  | 0.883 | 2.79 |  [mobilenetv2_rhd2d.md](./rhd2d/mobilenetv2_rhd2d.md)  |
