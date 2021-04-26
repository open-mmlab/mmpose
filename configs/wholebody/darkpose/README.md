# Distribution-aware coordinate representation for human pose estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```

## Results and models

### 2d Human Whole-Body Pose Estimation

#### Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [pose_hrnet_w32_dark](/configs/wholebody/darkpose/coco-wholebody/hrnet_w32_coco_wholebody_256x192_dark.py)  | 256x192 | 0.694 | 0.764 | 0.565 | 0.674 | 0.736 | 0.808 | 0.503 | 0.602 | 0.582 | 0.671 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark_20200922.log.json) |
| [pose_hrnet_w48_dark+](/configs/wholebody/darkpose/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py)  | 384x288 | 0.742 | 0.807 | 0.705 | 0.804 | 0.840 | 0.892 | 0.602 | 0.694 | 0.661 | 0.743 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark_20200918.log.json) |

Note: `+` means the model is first pre-trained on original COCO dataset, and then fine-tuned on COCO-WholeBody dataset. We find this will lead to better performance.
