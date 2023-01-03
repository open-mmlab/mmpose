# RTMPose

TODO

## Results and Models

### COCO Dataset

Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

|   Model   | Input Size |  AP   |  AR   |           Details and Download            |
| :-------: | :--------: | :---: | :---: | :---------------------------------------: |
| RTMPose-t |  256x192   | 0.679 | 0.709 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-s |  256x192   | 0.721 | 0.749 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-m |  256x192   | 0.759 | 0.785 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-l |  256x192   | 0.774 | 0.798 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |

### MPII Dataset

|   Model   | Input Size | PCKh@0.5 | PCKh@0.1 |           Details and Download            |
| :-------: | :--------: | :------: | :------: | :---------------------------------------: |
| RTMPose-m |  256x256   |  0.907   |  0.348   | [rtmpose_mpii.md](./mpii/rtmpose_mpii.md) |

### CrowdPose Dataset

Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

|   Model   | Input Size |  AP   |  AR   |                  Details and Download                   |
| :-------: | :--------: | :---: | :---: | :-----------------------------------------------------: |
| RTMPose-m |  256x192   | 0.678 | 0.763 | [rtmpose_crowdpose.md](./crowdpose/rtmpose_crowpose.md) |
