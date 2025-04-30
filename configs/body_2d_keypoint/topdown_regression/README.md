# Top-down regression-based pose estimation

Top-down methods divide the task into two stages: object detection, followed by single-object pose estimation given object bounding boxes. At the 2nd stage, regression based methods directly regress the keypoint coordinates given the features extracted from the bounding box area, following the paradigm introduced in [Deeppose: Human pose estimation via deep neural networks](http://openaccess.thecvf.com/content_cvpr_2014/html/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.html).

<div align=center>
<img src="https://user-images.githubusercontent.com/15977946/146515040-a82a8a29-d6bc-42f1-a2ab-7dfa610ce363.png">
</div>

## Results and Models

### COCO Dataset

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|      Model       | Input Size |  AP   |  AR   |                   Details and Download                    |
| :--------------: | :--------: | :---: | :---: | :-------------------------------------------------------: |
|  ResNet-152+RLE  |  256x192   | 0.731 | 0.805 |      [resnet_rle_coco.md](./coco/resnet_rle_coco.md)      |
|  ResNet-101+RLE  |  256x192   | 0.722 | 0.768 |      [resnet_rle_coco.md](./coco/resnet_rle_coco.md)      |
|  ResNet-50+RLE   |  256x192   | 0.706 | 0.768 |      [resnet_rle_coco.md](./coco/resnet_rle_coco.md)      |
| MobileNet-v2+RLE |  256x192   | 0.593 | 0.644 | [mobilenetv2_rle_coco.md](./coco/mobilenetv2_rle_coco.md) |
|    ResNet-152    |  256x192   | 0.584 | 0.688 |          [resnet_coco.md](./coco/resnet_coco.md)          |
|    ResNet-101    |  256x192   | 0.562 | 0.670 |          [resnet_coco.md](./coco/resnet_coco.md)          |
|    ResNet-50     |  256x192   | 0.528 | 0.639 |          [resnet_coco.md](./coco/resnet_coco.md)          |

### MPII Dataset

|     Model     | Input Size | PCKh@0.5 | PCKh@0.1 |              Details and Download               |
| :-----------: | :--------: | :------: | :------: | :---------------------------------------------: |
| ResNet-50+RLE |  256x256   |  0.861   |  0.277   | [resnet_rle_mpii.md](./mpii/resnet_rle_mpii.md) |
|  ResNet-152   |  256x256   |  0.850   |  0.208   |     [resnet_mpii.md](./mpii/resnet_mpii.md)     |
|  ResNet-101   |  256x256   |  0.841   |  0.200   |     [resnet_mpii.md](./mpii/resnet_mpii.md)     |
|   ResNet-50   |  256x256   |  0.826   |  0.180   |     [resnet_mpii.md](./mpii/resnet_mpii.md)     |
