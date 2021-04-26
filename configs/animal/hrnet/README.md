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

### 2d Animal Landmark Detection

#### Results on Horse-10 test set

|Set   | Arch  | Input Size | PCK@0.3 |  NME  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: |:------: |:------: |
|split1| [pose_hrnet_w32](/configs/animal/hrnet/horse10/hrnet_w32_horse10_256x256-split1.py) | 256x256 | 0.951 | 0.122 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split1-401d901a_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_hrnet_w32](/configs/animal/hrnet/horse10/hrnet_w32_horse10_256x256-split2.py) | 256x256 | 0.949 | 0.116 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split2-04840523_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_hrnet_w32](/configs/animal/hrnet/horse10/hrnet_w32_horse10_256x256-split3.py) | 256x256 | 0.939 | 0.153 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split3-4db47400_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_hrnet_w48](/configs/animal/hrnet/horse10/hrnet_w48_horse10_256x256-split1.py) | 256x256 | 0.973 | 0.095 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split1-3c950d3b_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_hrnet_w48](/configs/animal/hrnet/horse10/hrnet_w48_horse10_256x256-split2.py) | 256x256 | 0.969 | 0.101 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split2-8ef72b5d_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_hrnet_w48](/configs/animal/hrnet/horse10/hrnet_w48_horse10_256x256-split3.py) | 256x256 | 0.961 | 0.128 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split3-0232ec47_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split3_20210405.log.json) |

#### Results on MacaquePose with ground-truth detection bounding boxes

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/animal/hrnet/macaque/hrnet_w32_macaque_256x192.py)  | 256x192 | 0.814 | 0.953 | 0.918 | 0.851 | 0.969 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192-f7e9e04f_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_macaque_256x192_20210407.log.json) |
| [pose_hrnet_w48](/configs/animal/hrnet/macaque/hrnet_w48_macaque_256x192.py)  | 256x192 | 0.818 | 0.963 | 0.917 | 0.855 | 0.971 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192_20210407.log.json) |

#### Results on ATRW validation set

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/animal/hrnet/atrw/hrnet_w32_atrw_256x256.py)  | 256x256 | 0.912 | 0.973 | 0.959 | 0.938 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_atrw_256x256-f027f09a_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_atrw_256x256_20210414.log.json) |
| [pose_hrnet_w48](/configs/animal/hrnet/atrw/hrnet_w48_atrw_256x256.py)  | 256x256 | 0.911 | 0.972 | 0.946 | 0.937 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_atrw_256x256-ac088892_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_atrw_256x256_20210414.log.json) |
