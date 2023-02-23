# RTMPose

TODO

## Results and Models

### COCO Dataset

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|       Model        | Input Size |  AP   |  AR   |           Details and Download            |
| :----------------: | :--------: | :---: | :---: | :---------------------------------------: |
|     RTMPose-t      |  256x192   | 0.682 | 0.736 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
|     RTMPose-s      |  256x192   | 0.716 | 0.768 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
|     RTMPose-m      |  256x192   | 0.746 | 0.795 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
|     RTMPose-l      |  256x192   | 0.758 | 0.806 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-t-aic-coco |  256x192   | 0.685 | 0.738 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-s-aic-coco |  256x192   | 0.722 | 0.772 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-m-aic-coco |  256x192   | 0.758 | 0.806 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-l-aic-coco |  256x192   | 0.765 | 0.813 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |

### MPII Dataset

|   Model   | Input Size | PCKh@0.5 | PCKh@0.1 |           Details and Download            |
| :-------: | :--------: | :------: | :------: | :---------------------------------------: |
| RTMPose-m |  256x256   |  0.907   |  0.348   | [rtmpose_mpii.md](./mpii/rtmpose_mpii.md) |

### CrowdPose Dataset

Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

|   Model   | Input Size |  AP   |  AR   |                  Details and Download                   |
| :-------: | :--------: | :---: | :---: | :-----------------------------------------------------: |
| RTMPose-m |  256x192   | 0.686 | 0.771 | [rtmpose_crowdpose.md](./crowdpose/rtmpose_crowpose.md) |
