# Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the likelihood of being a keypoint, following the paradigm introduced in [Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146522977-5f355832-e9c1-442f-a34f-9d24fb0aefa8.png" height=400>
</div>

## Results and Models

### DeepFashion Dataset

Results on DeepFashion dataset with ResNet backbones:

|        Model        | Input Size | PCK@0.2 | AUC  | EPE  |                     Details and Download                     |
| :-----------------: | :--------: | :-----: | :--: | :--: | :----------------------------------------------------------: |
| HRNet-w48-UDP-Upper |  256x192   |  96.1   | 60.9 | 15.1 |  [hrnet_deepfashion.md](./deepfashion/hrnet_deepfashion.md)  |
| HRNet-w48-UDP-Lower |  256x192   |  97.8   | 76.1 | 8.9  |  [hrnet_deepfashion.md](./deepfashion/hrnet_deepfashion.md)  |
| HRNet-w48-UDP-Full  |  256x192   |  98.3   | 67.3 | 11.7 |  [hrnet_deepfashion.md](./deepfashion/hrnet_deepfashion.md)  |
|   ResNet-50-Upper   |  256x192   |  95.4   | 57.8 | 16.8 | [resnet_deepfashion.md](./deepfashion/resnet_deepfashion.md) |
|   ResNet-50-Lower   |  256x192   |  96.5   | 74.4 | 10.5 | [resnet_deepfashion.md](./deepfashion/resnet_deepfashion.md) |
|   ResNet-50-Full    |  256x192   |  97.7   | 66.4 | 12.7 | [resnet_deepfashion.md](./deepfashion/resnet_deepfashion.md) |

### DeepFashion2 Dataset

Results on DeepFashion2 dataset

|              Model              | Input Size | PCK@0.2 |  AUC  | EPE  |                     Details and Download                      |
| :-----------------------------: | :--------: | :-----: | :---: | :--: | :-----------------------------------------------------------: |
|  ResNet-50-Short-Sleeved-Shirt  |  256x192   |  0.988  | 0.703 | 10.2 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|  ResNet-50-Long-Sleeved-Shirt   |  256x192   |  0.973  | 0.587 | 16.6 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
| ResNet-50-Short-Sleeved-Outwear |  256x192   |  0.966  | 0.408 | 24.0 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
| ResNet-50-Long-Sleeved-Outwear  |  256x192   |  0.987  | 0.517 | 18.1 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|         ResNet-50-Vest          |  256x192   |  0.981  | 0.643 | 12.7 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|         ResNet-50-Sling         |  256x192   |  0.940  | 0.557 | 21.6 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|        ResNet-50-Shorts         |  256x192   |  0.975  | 0.682 | 12.4 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|       ResNet-50-Trousers        |  256x192   |  0.973  | 0.625 | 14.8 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|         ResNet-50-Skirt         |  256x192   |  0.952  | 0.653 | 16.6 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|  ResNet-50-Short-Sleeved-Dress  |  256x192   |  0.980  | 0.603 | 15.6 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|  ResNet-50-Long-Sleeved-Dress   |  256x192   |  0.976  | 0.518 | 20.1 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|      ResNet-50-Vest-Dress       |  256x192   |  0.980  | 0.600 | 16.0 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
|      ResNet-50-Sling-Dress      |  256x192   |  0.967  | 0.544 | 19.5 | [res50_deepfashion2.md](./deepfashion2/res50_deepfashion2.md) |
