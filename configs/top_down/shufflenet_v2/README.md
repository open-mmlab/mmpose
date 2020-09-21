# Shufflenet v2: Practical guidelines for efficient cnn architecture design

## Introduction
```
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
  year={2018}
}
```

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_shufflenetv2](/configs/top_down/shufflenet_v2/coco/shufflenetv2_coco_256x192.py)  | 256x192 | 0.599 | 0.854 | 0.663 | 0.664 | 0.899 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192-0aba71c7_20200921.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_256x192_20200921.log.json) |
| [pose_shufflenetv2](/configs/top_down/shufflenet_v2/coco/shufflenetv2_coco_384x288.py)  | 384x288 | 0.636 | 0.865 | 0.705 | 0.697 | 0.909 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288-fb38ac3a_20200921.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/shufflenetv2/shufflenetv2_coco_384x288_20200921.log.json) |
