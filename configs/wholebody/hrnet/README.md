# Deep high-resolution representation learning for human pose estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

## Results and models

### 2d Human Whole-Body Pose Estimation

#### Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [pose_hrnet_w32](/configs/wholebody/hrnet/coco-wholebody/hrnet_w32_coco_wholebody_256x192.py)  | 256x192 | 0.700 | 0.746 | 0.567 | 0.645 | 0.637 | 0.688 | 0.473 | 0.546 | 0.553 | 0.626 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_20200918.log.json) |
| [pose_hrnet_w32](/configs/wholebody/hrnet/coco-wholebody/hrnet_w32_coco_wholebody_384x288.py)  | 384x288 | 0.701 | 0.773 | 0.586 | 0.692 | 0.727 | 0.783 | 0.516 | 0.604 | 0.586 | 0.674 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_384x288-78cacac3_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_384x288_20200922.log.json) |
| [pose_hrnet_w48](/configs/wholebody/hrnet/coco-wholebody/hrnet_w48_coco_wholebody_256x192.py)  | 256x192 | 0.700 | 0.776 | 0.672 | 0.785 | 0.656 | 0.743 | 0.534 | 0.639 | 0.579 | 0.681 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192_20200922.log.json) |
| [pose_hrnet_w48](/configs/wholebody/hrnet/coco-wholebody/hrnet_w48_coco_wholebody_384x288.py)  | 384x288 | 0.722 | 0.790 | 0.694 | 0.799 | 0.777 | 0.834 | 0.587 | 0.679 | 0.631 | 0.716 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_20200922.log.json) |
