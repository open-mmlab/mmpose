# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint, following the paradigm introduced in [Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522977-5f355832-e9c1-442f-a34f-9d24fb0aefa8.png" height=400>
</div>

## Results and Models

### Animal-Pose Dataset

Results on AnimalPose validation set (1117 instances)

|   Model    | Input Size |  AP   |  AR   |                   Details and Download                    |
| :--------: | :--------: | :---: | :---: | :-------------------------------------------------------: |
| HRNet-w32  |  256x256   | 0.740 | 0.780 |  [hrnet_animalpose.md](./animalpose/hrnet_animalpose.md)  |
| HRNet-w48  |  256x256   | 0.738 | 0.778 |  [hrnet_animalpose.md](./animalpose/hrnet_animalpose.md)  |
| ResNet-152 |  256x256   | 0.704 | 0.748 | [resnet_animalpose.md](./animalpose/resnet_animalpose.md) |
| ResNet-101 |  256x256   | 0.696 | 0.736 | [resnet_animalpose.md](./animalpose/resnet_animalpose.md) |
| ResNet-50  |  256x256   | 0.691 | 0.736 | [resnet_animalpose.md](./animalpose/resnet_animalpose.md) |

### AP-10K Dataset

Results on AP-10K validation set

|   Model    | Input Size |  AP   |                 Details and Download                 |
| :--------: | :--------: | :---: | :--------------------------------------------------: |
| HRNet-w48  |  256x256   | 0.728 |       [hrnet_ap10k.md](./ap10k/hrnet_ap10k.md)       |
| HRNet-w32  |  256x256   | 0.722 |       [hrnet_ap10k.md](./ap10k/hrnet_ap10k.md)       |
| ResNet-101 |  256x256   | 0.681 |      [resnet_ap10k.md](./ap10k/resnet_ap10k.md)      |
| ResNet-50  |  256x256   | 0.680 |      [resnet_ap10k.md](./ap10k/resnet_ap10k.md)      |
| CSPNeXt-m  |  256x256   | 0.703 | [cspnext_udp_ap10k.md](./ap10k/cspnext_udp_ap10k.md) |

### Desert Locust Dataset

Results on Desert Locust test set

|   Model    | Input Size |  AUC  | EPE  |             Details and Download              |
| :--------: | :--------: | :---: | :--: | :-------------------------------------------: |
| ResNet-152 |  160x160   | 0.925 | 1.49 | [resnet_locust.md](./locust/resnet_locust.md) |
| ResNet-101 |  160x160   | 0.907 | 2.03 | [resnet_locust.md](./locust/resnet_locust.md) |
| ResNet-50  |  160x160   | 0.900 | 2.27 | [resnet_locust.md](./locust/resnet_locust.md) |

### Grévy’s Zebra Dataset

Results on Grévy’s Zebra test set

|   Model    | Input Size |  AUC  | EPE  |            Details and Download            |
| :--------: | :--------: | :---: | :--: | :----------------------------------------: |
| ResNet-152 |  160x160   | 0.921 | 1.67 | [resnet_zebra.md](./zebra/resnet_zebra.md) |
| ResNet-101 |  160x160   | 0.915 | 1.83 | [resnet_zebra.md](./zebra/resnet_zebra.md) |
| ResNet-50  |  160x160   | 0.914 | 1.87 | [resnet_zebra.md](./zebra/resnet_zebra.md) |

### Animal-Kingdom Dataset

Results on AnimalKingdom test set

|   Model   | Input Size |     class     | PCK(0.05) |                 Details and Download                  |
| :-------: | :--------: | :-----------: | :-------: | :---------------------------------------------------: |
| HRNet-w32 |  256x256   |      P1       |  0.6323   | [hrnet_animalkingdom.md](./ak/hrnet_animalkingdom.md) |
| HRNet-w32 |  256x256   |      P2       |  0.3741   | [hrnet_animalkingdom.md](./ak/hrnet_animalkingdom.md) |
| HRNet-w32 |  256x256   |  P3_mammals   |   0.571   | [hrnet_animalkingdom.md](./ak/hrnet_animalkingdom.md) |
| HRNet-w32 |  256x256   | P3_amphibians |  0.5358   | [hrnet_animalkingdom.md](./ak/hrnet_animalkingdom.md) |
| HRNet-w32 |  256x256   |  P3_reptiles  |   0.51    | [hrnet_animalkingdom.md](./ak/hrnet_animalkingdom.md) |
| HRNet-w32 |  256x256   |   P3_birds    |  0.7671   | [hrnet_animalkingdom.md](./ak/hrnet_animalkingdom.md) |
| HRNet-w32 |  256x256   |   P3_fishes   |  0.6406   | [hrnet_animalkingdom.md](./ak/hrnet_animalkingdom.md) |
