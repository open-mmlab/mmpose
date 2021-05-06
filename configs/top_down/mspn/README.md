# Rethinking on multi-stage networks for human pose estimation

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@article{li2019rethinking,
  title={Rethinking on Multi-Stage Networks for Human Pose Estimation},
  author={Li, Wenbo and Wang, Zhicheng and Yin, Binyi and Peng, Qixiang and Du, Yuming and Xiao, Tianzi and Yu, Gang and Lu, Hongtao and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1901.00148},
  year={2019}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [mspn_50](/configs/top_down/mspn/coco/mspn50_coco_256x192.py) | 256x192 | 0.723 | 0.895 | 0.794 | 0.788 | 0.933 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192-8fbfb5d0_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/mspn50_coco_256x192_20201123.log.json) |
| [2xmspn_50](/configs/top_down/mspn/coco/2xmspn50_coco_256x192.py) | 256x192 | 0.754 | 0.903 | 0.825 | 0.815 | 0.941 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/2xmspn50_coco_256x192-c8765a5c_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/2xmspn50_coco_256x192_20201123.log.json) |
| [3xmspn_50](/configs/top_down/mspn/coco/3xmspn50_coco_256x192.py) | 256x192 | 0.758 | 0.904 | 0.830 | 0.821 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/3xmspn50_coco_256x192-e348f18e_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/3xmspn50_coco_256x192_20201123.log.json) |
| [4xmspn_50](/configs/top_down/mspn/coco/4xmspn50_coco_256x192.py) | 256x192 | 0.764 | 0.906 | 0.835 | 0.826 | 0.944 | [ckpt](https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192-7b837afb_20201123.pth) | [log](https://download.openmmlab.com/mmpose/top_down/mspn/4xmspn50_coco_256x192_20201123.log.json) |
