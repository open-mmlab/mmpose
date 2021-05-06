# Deeppose: Human pose estimation via deep neural networks

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{toshev2014deeppose,
  title={Deeppose: Human pose estimation via deep neural networks},
  author={Toshev, Alexander and Szegedy, Christian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1653--1660},
  year={2014}
}
```

## Results and models

### 2d Fashion Landmark Detection

#### Results on DeepFashion val set

|Set   | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: | :------: |:------: |:------: |
|upper | [deeppose_resnet_50](/configs/fashion/deeppose/deepfashion/deeppose_res50_deepfashion_upper_256x192.py) | 256x256 | 0.965 | 0.535 | 17.2 | [ckpt](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_upper_256x192-497799fb_20210309.pth) | [log](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_upper_256x192_20210309.log.json) |
|lower | [deeppose_resnet_50](/configs/fashion/deeppose/deepfashion/deeppose_res50_deepfashion_lower_256x192.py) | 256x256 | 0.971 | 0.678 | 11.8 | [ckpt](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_lower_256x192-94e0e653_20210309.pth) | [log](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_lower_256x192_20210309.log.json) |
|full  | [deeppose_resnet_50](/configs/fashion/deeppose/deepfashion/deeppose_res50_deepfashion_full_256x192.py)  | 256x256 | 0.983 | 0.602 | 14.0 | [ckpt](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_full_256x192-4e0273e2_20210309.pth) | [log](https://download.openmmlab.com/mmpose/fashion/deeppose/deeppose_res50_deepfashion_full_256x192_20210309.log.json) |
