# Learning delicate local representations for multi-person pose estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@misc{cai2020learning,
    title={Learning Delicate Local Representations for Multi-Person Pose Estimation},
    author={Yuanhao Cai and Zhicheng Wang and Zhengxiong Luo and Binyi Yin and Angang Du and Haoqian Wang and Xinyu Zhou and Erjin Zhou and Xiangyu Zhang and Jian Sun},
    year={2020},
    eprint={2003.04030},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [rsn_18](/configs/top_down/rsn/coco/rsn18_coco_256x192.py) | 256x192 | 0.704 | 0.887 | 0.779 | 0.771 | 0.926 | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/rsn18_coco_256x192-72f4b4a7_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/rsn18_coco_256x192_20201127.log.json) |
| [rsn_50](/configs/top_down/rsn/coco/rsn50_coco_256x192.py) | 256x192 | 0.723 | 0.896 | 0.800 | 0.788 | 0.934 | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/rsn50_coco_256x192-72ffe709_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/rsn50_coco_256x192_20201127.log.json) |
| [2xrsn_50](/configs/top_down/rsn/coco/2xrsn50_coco_256x192.py) | 256x192 | 0.745 | 0.899 | 0.818 | 0.809 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/2xrsn50_coco_256x192-50648f0e_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/2xrsn50_coco_256x192_20201127.log.json) |
| [3xrsn_50](/configs/top_down/rsn/coco/3xrsn50_coco_256x192.py) | 256x192 | 0.750 | 0.900 | 0.823 | 0.813 | 0.940 | [ckpt](https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192-58f57a68_20201127.pth) | [log](https://download.openmmlab.com/mmpose/top_down/rsn/3xrsn50_coco_256x192_20201127.log.json) |
