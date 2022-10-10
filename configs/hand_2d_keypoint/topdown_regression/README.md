# Top-down regression-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. At the 2nd stage, regression based methods directly regress the keypoint coordinates given the features extracted from the bounding box area, following the paradigm introduced in [Deeppose: Human pose estimation via deep neural networks](http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146515040-a82a8a29-d6bc-42f1-a2ab-7dfa610ce363.png">
</div>

## Results and Models

### OneHand10K Dataset

Results on OneHand10K val set

|   Model   | Input Size | PCK@0.2 |  AUC  |  EPE  |                   Details and Download                    |
| :-------: | :--------: | :-----: | :---: | :---: | :-------------------------------------------------------: |
| ResNet-50 |  256x256   |  0.990  | 0.485 | 34.21 | [resnet_onehand10k.md](./onehand10k/resnet_onehand10k.md) |

### RHD Dataset

Results on RHD test set

|   Model   | Input Size | PCK@0.2 |  AUC  | EPE  |            Details and Download            |
| :-------: | :--------: | :-----: | :---: | :--: | :----------------------------------------: |
| ResNet-50 |  256x256   |  0.988  | 0.865 | 3.32 | [resnet_rhd2d.md](./rhd2d/resnet_rhd2d.md) |
