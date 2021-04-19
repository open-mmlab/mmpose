# Mobilenetv2: Inverted residuals and linear bottlenecks

## Introduction

<!-- [BACKBONE] -->

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenetv2](/configs/top_down/mobilenet_v2/coco/mobilenetv2_coco_256x192.py)  | 256x192 | 0.646 | 0.874 | 0.723 | 0.707 | 0.917 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192_20200727.log.json) |
| [pose_mobilenetv2](/configs/top_down/mobilenet_v2/coco/mobilenetv2_coco_384x288.py)  | 384x288 | 0.673 | 0.879 | 0.743 | 0.729 | 0.916 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288-26be4816_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_384x288_20200727.log.json) |

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_mobilenetv2](/configs/top_down/mobilenet_v2/mpii/mobilenet_v2_mpii_256x256.py) | 256x256 | 0.854 | 0.235 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_mpii_256x256-e068afa7_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_mpii_256x256_20200812.log.json) |
