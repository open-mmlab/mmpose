# ResNetV1D

## Introduction

<!-- [BACKBONE] -->

```bibtex
@inproceedings{he2019bag,
  title={Bag of tricks for image classification with convolutional neural networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2019}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnetv1d_50](/configs/top_down/resnetv1d/coco/resnetv1d50_coco_256x192.py)  | 256x192 | 0.722 | 0.897 | 0.799 | 0.777 | 0.933 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192-a243b840_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_50](/configs/top_down/resnetv1d/coco/resnetv1d50_coco_384x288.py)  | 384x288 | 0.730 | 0.900 | 0.799 | 0.780 | 0.934 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_384x288-01f3fbb9_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_coco_384x288_20200727.log.json) |
| [pose_resnetv1d_101](/configs/top_down/resnetv1d/coco/resnetv1d101_coco_256x192.py) | 256x192 | 0.731 | 0.899 | 0.809 | 0.786 | 0.938 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_256x192-5bd08cab_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_101](/configs/top_down/resnetv1d/coco/resnetv1d101_coco_384x288.py) | 384x288 | 0.748 | 0.902 | 0.816 | 0.799 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_384x288-5f9e421d_20200730.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_coco_384x288-20200730.log.json) |
| [pose_resnetv1d_152](/configs/top_down/resnetv1d/coco/resnetv1d152_coco_256x192.py) | 256x192 | 0.737 | 0.902 | 0.812 | 0.791 | 0.940 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_256x192-c4df51dc_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_256x192_20200727.log.json) |
| [pose_resnetv1d_152](/configs/top_down/resnetv1d/coco/resnetv1d152_coco_384x288.py) | 384x288 | 0.752 | 0.909 | 0.821 | 0.802 | 0.944 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-626c622d_20200730.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_coco_384x288-20200730.log.json) |

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_resnetv1d_50](/configs/top_down/resnetv1d/mpii/resnetv1d50_mpii_256x256.py) | 256x256 | 0.881 | 0.290 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_mpii_256x256-2337a92e_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_mpii_256x256_20200812.log.json) |
| [pose_resnetv1d_101](/configs/top_down/resnetv1d/mpii/resnetv1d101_mpii_256x256.py) | 256x256 | 0.883 | 0.295 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_mpii_256x256-2851d710_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_mpii_256x256_20200812.log.json) |
| [pose_resnetv1d_152](/configs/top_down/resnetv1d/mpii/resnetv1d152_mpii_256x256.py) | 256x256 | 0.888 | 0.300 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_mpii_256x256-8b10a87c_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_mpii_256x256_20200812.log.json) |
