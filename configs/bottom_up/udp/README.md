# The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@InProceedings{Huang_2020_CVPR,
author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

Note that, UDP also adopts the unbiased encoding/decoding algorithm of [DARK](/configs/top_down/darkpose/README.md).

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w32_udp](/configs/bottom_up/udp/coco/hrnet_w32_coco_512x512_udp.py)  | 512x512 | 0.671 | 0.863 | 0.729 | 0.717 | 0.889 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp-91663bf9_20210220.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp_20210220.log.json) |
| [HRNet-w48_udp](/configs/bottom_up/udp/coco/hrnet_w48_coco_512x512_udp.py)  | 512x512 | 0.681 | 0.872 | 0.741 | 0.725 | 0.892 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_udp-de08fd8c_20210222.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w48_coco_512x512_udp_20210222.log.json) |
| [HigherHRNet-w32_udp](/configs/bottom_up/udp/coco/higher_hrnet32_coco_512x512_udp.py)  | 512x512 | 0.678 | 0.862 | 0.736 | 0.724 | 0.890 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_udp-8cc64794_20210222.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_udp_20210222.log.json) |
| [HigherHRNet-w48_udp](/configs/bottom_up/udp/coco/higher_hrnet48_coco_512x512_udp.py)  | 512x512 | 0.690 | 0.872 | 0.750 | 0.734 | 0.891 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp-7cad61ef_20210222.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet48_coco_512x512_udp_20210222.log.json) |
