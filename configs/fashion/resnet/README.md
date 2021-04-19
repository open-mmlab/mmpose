# Simple baselines for human pose estimation and tracking

## Introduction

<!-- [ALGORITHM] -->

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

### 2d Fashion Landmark Detection

#### Results on DeepFashion val set

|Set   | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: | :------: |:------: |:------: |
|upper | [pose_resnet_50](/configs/fashion/resnet/deepfashion/res50_deepfashion_upper_256x192.py) | 256x256 | 0.954 | 0.578 | 16.8 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_upper_256x192-41794f03_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_upper_256x192_20210124.log.json) |
|lower | [pose_resnet_50](/configs/fashion/resnet/deepfashion/res50_deepfashion_lower_256x192.py) | 256x256 | 0.965 | 0.744 | 10.5 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_lower_256x192-1292a839_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_lower_256x192_20210124.log.json) |
|full  | [pose_resnet_50](/configs/fashion/resnet/deepfashion/res50_deepfashion_full_256x192.py)  | 256x256 | 0.977 | 0.664 | 12.7 | [ckpt](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_full_256x192-0dbd6e42_20210124.pth) | [log](https://download.openmmlab.com/mmpose/fashion/resnet/res50_deepfashion_full_256x192_20210124.log.json) |
