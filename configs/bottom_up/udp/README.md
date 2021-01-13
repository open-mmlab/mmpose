# The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation

## Introduction

[ALGORITHM]

```latex
@InProceedings{Huang_2020_CVPR,
author = {Huang, Junjie and Zhu, Zheng and Guo, Feng and Huang, Guan},
title = {The Devil Is in the Details: Delving Into Unbiased Data Processing for Human Pose Estimation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [HRNet-w32_udp](/configs/bottom_up/udp/coco/hrnet_w32_coco_512x512_udp.py)  | 512x512 | 0.667 | 0.861 | 0.719 | 0.713 | 0.887 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp-7f47d165_20210104.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/hrnet_w32_coco_512x512_udp_20210104.log.json) |
| [HigherHRNet-w32_udp](/configs/bottom_up/udp/coco/higher_hrnet32_coco_512x512_udp.py)  | 512x512 | 0.674 | 0.868 | 0.722 | 0.716 | 0.890 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_udp-83e65040_20210104.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/higher_hrnet32_coco_512x512_udp_20210104.log.json) |
