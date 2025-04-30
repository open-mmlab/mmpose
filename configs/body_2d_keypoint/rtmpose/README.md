# RTMPose

Recent studies on 2D pose estimation have achieved excellent performance on public benchmarks, yet its application in the industrial community still suffers from heavy model parameters and high latency.
In order to bridge this gap, we empirically study five aspects that affect the performance of multi-person pose estimation algorithms: paradigm, backbone network, localization algorithm, training strategy, and deployment inference, and present a high-performance real-time multi-person pose estimation framework, **RTMPose**, based on MMPose.
Our RTMPose-m achieves **75.8% AP** on COCO with **90+ FPS** on an Intel i7-11700 CPU and **430+ FPS** on an NVIDIA GTX 1660 Ti GPU, and RTMPose-l achieves **67.0% AP** on COCO-WholeBody with **130+ FPS**, outperforming existing open-source libraries.
To further evaluate RTMPose's capability in critical real-time applications, we also report the performance after deploying on the mobile device.

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
| RTMPose-m-aic-coco |  384x288   | 0.770 | 0.816 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |
| RTMPose-l-aic-coco |  384x288   | 0.773 | 0.819 | [rtmpose_coco.md](./coco/rtmpose_coco.md) |

### MPII Dataset

|   Model   | Input Size | PCKh@0.5 | PCKh@0.1 |           Details and Download            |
| :-------: | :--------: | :------: | :------: | :---------------------------------------: |
| RTMPose-m |  256x256   |  0.907   |  0.348   | [rtmpose_mpii.md](./mpii/rtmpose_mpii.md) |

### CrowdPose Dataset

Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

|   Model   | Input Size |  AP   |  AR   |                   Details and Download                   |
| :-------: | :--------: | :---: | :---: | :------------------------------------------------------: |
| RTMPose-m |  256x192   | 0.706 | 0.788 | [rtmpose_crowdpose.md](./crowdpose/rtmpose_crowdpose.md) |

### Human-Art Dataset

Results on Human-Art validation dataset with detector having human AP of 56.2 on Human-Art validation dataset

|   Model   | Input Size |  AP   |  AR   |                 Details and Download                  |
| :-------: | :--------: | :---: | :---: | :---------------------------------------------------: |
| RTMPose-s |  256x192   | 0.311 | 0.381 | [rtmpose_humanart.md](./humanart/rtmpose_humanart.md) |
| RTMPose-m |  256x192   | 0.355 | 0.417 | [rtmpose_humanart.md](./humanart/rtmpose_humanart.md) |
| RTMPose-l |  256x192   | 0.378 | 0.442 | [rtmpose_humanart.md](./humanart/rtmpose_humanart.md) |

Results on Human-Art validation dataset with ground-truth bounding-box

|   Model   | Input Size |  AP   |  AR   |                 Details and Download                  |
| :-------: | :--------: | :---: | :---: | :---------------------------------------------------: |
| RTMPose-s |  256x192   | 0.698 | 0.732 | [rtmpose_humanart.md](./humanart/rtmpose_humanart.md) |
| RTMPose-m |  256x192   | 0.728 | 0.759 | [rtmpose_humanart.md](./humanart/rtmpose_humanart.md) |
| RTMPose-l |  256x192   | 0.753 | 0.783 | [rtmpose_humanart.md](./humanart/rtmpose_humanart.md) |
