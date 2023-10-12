# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the likelihood of being a keypoint, following the paradigm introduced in [Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522977-5f355832-e9c1-442f-a34f-9d24fb0aefa8.png" height=400>
</div>

## Results and Models

### COCO-WholeBody Dataset

Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

|        Model        | Input Size | Whole AP | Whole AR |                              Details and Download                               |
| :-----------------: | :--------: | :------: | :------: | :-----------------------------------------------------------------------------: |
|   HRNet-w48+Dark+   |  384x288   |  0.661   |  0.743   |  [hrnet_dark_coco-wholebody.md](./coco-wholebody/hrnet_dark_coco-wholebody.md)  |
|   HRNet-w32+Dark    |  256x192   |  0.582   |  0.671   |  [hrnet_dark_coco-wholebody.md](./coco-wholebody/hrnet_dark_coco-wholebody.md)  |
|      HRNet-w48      |  256x192   |  0.579   |  0.681   |       [hrnet_coco-wholebody.md](./coco-wholebody/hrnet_coco-wholebody.md)       |
|      CSPNeXt-m      |  256x192   |  0.567   |  0.641   | [cspnext_udp_coco-wholebody.md](./coco-wholebody/cspnext_udp_coco-wholebody.md) |
|      HRNet-w32      |  256x192   |  0.549   |  0.646   |    [hrnet_ubody-coco-wholebody.md](./ubody2d/hrnet_ubody-coco-wholebody.md)     |
|     ResNet-152      |  256x192   |  0.548   |  0.661   |      [resnet_coco-wholebody.md](./coco-wholebody/resnet_coco-wholebody.md)      |
|      HRNet-w32      |  256x192   |  0.536   |  0.636   |       [hrnet_coco-wholebody.md](./coco-wholebody/hrnet_coco-wholebody.md)       |
|     ResNet-101      |  256x192   |  0.531   |  0.645   |      [resnet_coco-wholebody.md](./coco-wholebody/resnet_coco-wholebody.md)      |
| S-ViPNAS-Res50+Dark |  256x192   |  0.528   |  0.632   | [vipnas_dark_coco-wholebody.md](./coco-wholebody/vipnas_dark_coco-wholebody.md) |
|      ResNet-50      |  256x192   |  0.521   |  0.633   |      [resnet_coco-wholebody.md](./coco-wholebody/resnet_coco-wholebody.md)      |
|   S-ViPNAS-Res50    |  256x192   |  0.495   |  0.607   |      [vipnas_coco-wholebody.md](./coco-wholebody/vipnas_coco-wholebody.md)      |

### UBody2D Dataset

Result on UBody val set, computed with gt keypoints.

|   Model   | Input Size | Whole AP | Whole AR |                           Details and Download                           |
| :-------: | :--------: | :------: | :------: | :----------------------------------------------------------------------: |
| HRNet-w32 |  256x192   |  0.690   |  0.729   | [hrnet_ubody-coco-wholebody.md](./ubody2d/hrnet_ubody-coco-wholebody.md) |
