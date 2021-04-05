# Simple baselines for human pose estimation and tracking

## Introduction

[ALGORITHM]

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

## Results and models

### 2d Animal Landmark Detection

#### Results on Horse-10 test set

|Set   | Arch  | Input Size | PCK@0.3 |  NME  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: |:------: |:------: |
|split1| [pose_resnet_50](/configs/animal/resnet/horse10/res50_horse10_256x256-split1.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1-3a3dc37e_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_50](/configs/animal/resnet/horse10/res50_horse10_256x256-split2.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2-65e2a508_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_50](/configs/animal/resnet/horse10/res50_horse10_256x256-split3.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3-9637d4eb_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_resnet_101](/configs/animal/resnet/horse10/res101_horse10_256x256-split1.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1-1b7c259c_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_101](/configs/animal/resnet/horse10/res101_horse10_256x256-split2.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2-30e2fa87_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_101](/configs/animal/resnet/horse10/res101_horse10_256x256-split3.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3-2eea5bb1_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_resnet_152](/configs/animal/resnet/horse10/res152_horse10_256x256-split1.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1-7e81fe2d_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_152](/configs/animal/resnet/horse10/res152_horse10_256x256-split2.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2-3b3404a3_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_152](/configs/animal/resnet/horse10/res152_horse10_256x256-split3.py) | 256x256 | - | - | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3-c957dac5_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3_20210405.log.json) |
