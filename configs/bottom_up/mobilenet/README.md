# Associative Embedding (AE) + Mobilenetv2

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{newell2017associative,
  title={Associative embedding: End-to-end learning for joint detection and grouping},
  author={Newell, Alejandro and Huang, Zhiao and Deng, Jia},
  booktitle={Advances in neural information processing systems},
  pages={2277--2287},
  year={2017}
}
```

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

#### Results on COCO val2017 without multi-scale test

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenetv2](/configs/bottom_up/mobilenet/coco/mobilenetv2_coco_512x512.py)  | 512x512 | 0.380 | 0.671 | 0.368 | 0.473 | 0.741 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512_20200816.log.json) |

#### Results on COCO val2017 with multi-scale test. 3 default scales (\[2, 1, 0.5\]) are used

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_mobilenetv2](/configs/bottom_up/mobilenet/coco/mobilenetv2_coco_512x512.py)  | 512x512 | 0.442 | 0.696 | 0.422 | 0.517 | 0.766 | [ckpt](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512-4d96e309_20200816.pth) | [log](https://download.openmmlab.com/mmpose/bottom_up/mobilenetv2_coco_512x512_20200816.log.json) |
