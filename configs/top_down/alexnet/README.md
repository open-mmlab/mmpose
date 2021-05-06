# Imagenet classification with deep convolutional neural networks

## Introduction

<!-- [BACKBONE] -->

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_alexnet](/configs/top_down/alexnet/coco/alexnet_coco_256x192.py)  | 256x192 | 0.397 | 0.758 | 0.381 | 0.478 | 0.822 | [ckpt](https://download.openmmlab.com/mmpose/top_down/alexnet/alexnet_coco_256x192-a7b1fd15_20200727.pth) | [log](https://download.openmmlab.com/mmpose/top_down/alexnet/alexnet_coco_256x192_20200727.log.json) |
