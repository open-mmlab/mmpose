# Stacked hourglass networks for human pose estimation

## Introduction
```
@inproceedings{newell2016stacked,
  title={Stacked hourglass networks for human pose estimation},
  author={Newell, Alejandro and Yang, Kaiyu and Deng, Jia},
  booktitle={European conference on computer vision},
  pages={483--499},
  year={2016},
  organization={Springer}
}
```

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| pose_hourglass_52 | 256x256 | 0.726 | 0.896 | 0.799 | 0.780 | 0.934 | [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hourglass/hourglass52_coco_256x256-4ec713ba_20200709.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmpose/top_down/hourglass/hourglass52_coco_256x256_20200709.log.json) |
