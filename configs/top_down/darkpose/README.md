# Distribution-aware coordinate representation for human pose estimation

## Introduction

[ALGORITHM]

```latex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [dark_pose_resnet_50](/configs/top_down/darkpose/coco/res50_coco_256x192_dark.py) | 256x192 | 0.724 | 0.898 | 0.800 | 0.777 | 0.936 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_dark-43379d20_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_dark_20200709.log.json) |
| [dark_pose_resnet_50](/configs/top_down/darkpose/coco/res50_coco_384x288_dark.py) | 384x288 | 0.735 | 0.900 | 0.801 | 0.785 | 0.937 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_dark-33d3e5e5_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_dark_20210203.log.json) |
| [dark_pose_resnet_101](/configs/top_down/darkpose/coco/res101_coco_256x192_dark.py) | 256x192 | 0.732 | 0.899 | 0.808 | 0.786 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_dark-64d433e6_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_dark_20200812.log.json) |
| [dark_pose_resnet_101](/configs/top_down/darkpose/coco/res101_coco_384x288_dark.py) | 384x288 | 0.749 | 0.902 | 0.816 | 0.799 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_dark-cb45c88d_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_dark_20210203.log.json) |
| [dark_pose_resnet_152](/configs/top_down/darkpose/coco/res152_coco_256x192_dark.py) | 256x192 | 0.745 | 0.905 | 0.821 | 0.797 | 0.942 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_dark-ab4840d5_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_dark_20200812.log.json) |
| [dark_pose_resnet_152](/configs/top_down/darkpose/coco/res152_coco_384x288_dark.py) | 384x288 | 0.757 | 0.909 | 0.826 | 0.806 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark-d3b8ebd7_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_dark_20210203.log.json) |
| [dark_pose_hrnet_w32](/configs/top_down/darkpose/coco/hrnet_w32_coco_256x192_dark.py) | 256x192 | 0.757 | 0.907 | 0.823 | 0.808 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_dark-07f147eb_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_dark_20200812.log.json) |
| [dark_pose_hrnet_w32](/configs/top_down/darkpose/coco/hrnet_w32_coco_384x288_dark.py) | 384x288 | 0.766 | 0.907 | 0.831 | 0.815 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_dark-307dafc2_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_dark_20210203.log.json) |
| [dark_pose_hrnet_w48](/configs/top_down/darkpose/coco/hrnet_w48_coco_256x192_dark.py) | 256x192 | 0.764 | 0.907 | 0.830 | 0.814 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_dark-8cba3197_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_dark_20200812.log.json) |
| [dark_pose_hrnet_w48](/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py) | 384x288 | 0.772 | 0.910 | 0.836 | 0.820 | 0.946 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark_20210203.log.json) |

#### Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [dark_pose_hrnet_w32](/configs/top_down/darkpose/coco-wholebody/hrnet_w32_coco_wholebody_256x192_dark.py)  | 256x192 | 0.694 | 0.764 | 0.565 | 0.674 | 0.736 | 0.808 | 0.503 | 0.602 | 0.582 | 0.671 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark_20200922.log.json) |
| [dark_pose_hrnet_w48+](/configs/top_down/darkpose/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py)  | 384x288 | 0.742 | 0.807 | 0.705 | 0.804 | 0.840 | 0.892 | 0.602 | 0.694 | 0.661 | 0.743 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark_20200918.log.json) |

Note: `+` means the model is first pre-trained on original COCO dataset, and then fine-tuned on COCO-WholeBody dataset. We find this will lead to better performance.

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [dark_pose_hrnet_w32](/configs/top_down/darkpose/mpii/hrnet_w32_mpii_256x256_dark.py) | 256x256 | 0.904 | 0.396 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark_20200927.log.json) |
| [dark_pose_hrnet_w48](/configs/top_down/darkpose/mpii/hrnet_w48_mpii_256x256_dark.py) | 256x256 | 0.905 | 0.401 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark-0decd39f_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark_20200927.log.json) |
