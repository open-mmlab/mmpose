# RTMPose

Recent studies on 2D pose estimation have achieved excellent performance on public benchmarks, yet its application in the industrial community still suffers from heavy model parameters and high latency.
In order to bridge this gap, we empirically study five aspects that affect the performance of multi-person pose estimation algorithms: paradigm, backbone network, localization algorithm, training strategy, and deployment inference, and present a high-performance real-time multi-person pose estimation framework, **RTMPose**, based on MMPose.
Our RTMPose-m achieves **75.8% AP** on COCO with **90+ FPS** on an Intel i7-11700 CPU and **430+ FPS** on an NVIDIA GTX 1660 Ti GPU, and RTMPose-l achieves **67.0% AP** on COCO-WholeBody with **130+ FPS**, outperforming existing open-source libraries.
To further evaluate RTMPose's capability in critical real-time applications, we also report the performance after deploying on the mobile device.

## Results and Models

### COCO-WholeBody-Hand Dataset

Results on COCO-WholeBody-Hand val set

|   Model   | Input Size | PCK@0.2 |  AUC  | EPE  |                                  Details and Download                                  |
| :-------: | :--------: | :-----: | :---: | :--: | :------------------------------------------------------------------------------------: |
| RTMPose-m |  256x256   |  0.815  | 0.837 | 4.51 | [rtmpose_coco_wholebody_hand.md](./coco_wholebody_hand/rtmpose_coco_wholebody_hand.md) |
