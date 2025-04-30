# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the likelihood of being a keypoint, following the paradigm introduced in [Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522977-5f355832-e9c1-442f-a34f-9d24fb0aefa8.png" height=400>
</div>

## Results and Models

### 300W Dataset

Results on 300W dataset

|    Model    | Input Size | NME<sub>*common*</sub> | NME<sub>*challenge*</sub> | NME<sub>*full*</sub> | NME<sub>*test*</sub> |           Details and Download            |
| :---------: | :--------: | :--------------------: | :-----------------------: | :------------------: | :------------------: | :---------------------------------------: |
| HRNetv2-w18 |  256x256   |          2.92          |           5.64            |         3.45         |         4.10         | [hrnetv2_300w.md](./300w/hrnetv2_300w.md) |

### AFLW Dataset

Results on AFLW dataset

|      Model       | Input Size | NME<sub>*full*</sub> | NME<sub>*frontal*</sub> |                Details and Download                 |
| :--------------: | :--------: | :------------------: | :---------------------: | :-------------------------------------------------: |
| HRNetv2-w18+Dark |  256x256   |         1.35         |          1.19           | [hrnetv2_dark_aflw.md](./aflw/hrnetv2_dark_aflw.md) |
|   HRNetv2-w18    |  256x256   |         1.41         |          1.27           |      [hrnetv2_aflw.md](./aflw/hrnetv2_aflw.md)      |

### COCO-WholeBody-Face Dataset

Results on COCO-WholeBody-Face val set

|      Model       | Input Size |  NME   |                                       Details and Download                                       |
| :--------------: | :--------: | :----: | :----------------------------------------------------------------------------------------------: |
| HRNetv2-w18+Dark |  256x256   | 0.0513 | [hrnetv2_dark_coco_wholebody_face.md](./coco_wholebody_face/hrnetv2_dark_coco_wholebody_face.md) |
|     SCNet-50     |  256x256   | 0.0567 |        [scnet_coco_wholebody_face.md](./coco_wholebody_face/scnet_coco_wholebody_face.md)        |
|   HRNetv2-w18    |  256x256   | 0.0569 |      [hrnetv2_coco_wholebody_face.md](./coco_wholebody_face/hrnetv2_coco_wholebody_face.md)      |
|    ResNet-50     |  256x256   | 0.0582 |       [resnet_coco_wholebody_face.md](./coco_wholebody_face/resnet_coco_wholebody_face.md)       |
|   HourglassNet   |  256x256   | 0.0587 |    [hourglass_coco_wholebody_face.md](./coco_wholebody_face/hourglass_coco_wholebody_face.md)    |
|   MobileNet-v2   |  256x256   | 0.0611 |  [mobilenetv2_coco_wholebody_face.md](./coco_wholebody_face/mobilenetv2_coco_wholebody_face.md)  |

### COFW Dataset

Results on COFW dataset

|    Model    | Input Size | NME  |           Details and Download            |
| :---------: | :--------: | :--: | :---------------------------------------: |
| HRNetv2-w18 |  256x256   | 3.48 | [hrnetv2_cofw.md](./cofw/hrnetv2_cofw.md) |

### WFLW  Dataset

Results on WFLW  dataset

|  Model  | Input Size | NME<sub>*test*</sub> | NME<sub>*pose*</sub> | NME<sub>*illumination*</sub> | NME<sub>*occlusion*</sub> | NME<sub>*blur*</sub> | NME<sub>*makeup*</sub> | NME<sub>*expression*</sub> |  Details and Download  |
| :-----: | :--------: | :------------------: | :------------------: | :--------------------------: | :-----------------------: | :------------------: | :--------------------: | :------------------------: | :--------------------: |
| HRNetv2-w18+Dark |  256x256   |         3.98         |         6.98         |             3.96             |           4.78            |         4.56         |          3.89          |            4.29            | [hrnetv2_dark_wflw.md](./wflw/hrnetv2_dark_wflw.md) |
| HRNetv2-w18+AWing |  256x256   |         4.02         |         6.94         |             3.97             |           4.78            |         4.59         |          3.87          |            4.28            | [hrnetv2_awing_wflw.md](./wflw/hrnetv2_awing_wflw.md) |
| HRNetv2-w18 |  256x256   |         4.06         |         6.97         |             3.99             |           4.83            |         4.58         |          3.94          |            4.33            | [hrnetv2_wflw.md](./wflw/hrnetv2_wflw.md) |
