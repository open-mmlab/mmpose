# Convolutional pose machines

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{wei2016convolutional,
  title={Convolutional pose machines},
  author={Wei, Shih-En and Ramakrishna, Varun and Kanade, Takeo and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={4724--4732},
  year={2016}
}
```

## Results and models

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [cpm](/configs/top_down/cpm/coco/cpm_coco_256x192.py)  | 256x192 | 0.623 | 0.859 | 0.704 | 0.686 | 0.903 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_256x192-aa4ba095_20200817.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_256x192_20200817.log.json) |
| [cpm](/configs/top_down/cpm/coco/cpm_coco_384x288.py)  | 384x288 | 0.650 | 0.864 | 0.725 | 0.708 | 0.905 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288-80feb4bc_20200821.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_coco_384x288_20200821.log.json) |

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [cpm](/configs/top_down/cpm/mpii/cpm_mpii_368x368.py) | 368x368 | 0.876 | 0.285 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368-116e62b8_20200822.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368_20200822.log.json) |

#### Results on Sub-JHMDB dataset

The models are pre-trained on MPII dataset only. NO test-time augmentation (multi-scale /rotation testing) is used.

##### Normalized by Person Size

| Split| Arch        | Input Size | Head | Sho  | Elb | Wri | Hip | Knee | Ank | Mean | ckpt    | log     |
| :--- | :--------:  | :--------: | :---: | :---: |:---: |:---: |:---: |:---:  |:---: | :---: | :-----: |:------: |
| Sub1 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub1_368x368.py) | 368x368 | 96.1 | 91.9 | 81.0 |  78.9 | 96.6 | 90.8| 87.3 | 89.5 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368-2d2585c9_20201122.pth)  | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368_20201122.log.json) |
| Sub2 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub2_368x368.py) | 368x368 | 98.1 | 93.6 | 77.1 |  70.9 | 94.0 | 89.1| 84.7 | 87.4 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368-fc742f1f_20201122.pth)  | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368_20201122.log.json) |
| Sub3 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub3_368x368.py) | 368x368 | 97.9 | 94.9 | 87.3 |  84.0 | 98.6 | 94.4| 86.2 | 92.4 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368-49337155_20201122.pth)  | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368_20201122.log.json) |
| Average |  cpm                                                       | 368x368 | 97.4 | 93.5 | 81.5 |  77.9 | 96.4 | 91.4| 86.1 | 89.8 | -        | -       |

##### Normalized by Torso Size

| Split| Arch        | Input Size | Head | Sho  | Elb | Wri | Hip | Knee | Ank | Mean | ckpt    | log     |
| :--- | :--------:  | :--------: | :---: | :---: |:---: |:---: |:---: |:---:  |:---: | :---: | :-----: |:------: |
| Sub1 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub1_368x368.py) | 368x368 | 89.0 | 63.0 | 54.0 |  54.9 | 68.2 | 63.1 | 61.2 | 66.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368-2d2585c9_20201122.pth)  | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub1_368x368_20201122.log.json) |
| Sub2 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub2_368x368.py) | 368x368 | 90.3 | 57.9 | 46.8 |  44.3 | 60.8 | 58.2 | 62.4 | 61.1 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368-fc742f1f_20201122.pth)  | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub2_368x368_20201122.log.json) |
| Sub3 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub3_368x368.py) | 368x368 | 91.0 | 72.6 | 59.9 |  54.0 | 73.2 | 68.5 | 65.8 | 70.3 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368-49337155_20201122.pth)  | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_jhmdb_sub3_368x368_20201122.log.json) |
| Average |  cpm                                                       | 368x368 | 90.1 | 64.5 | 53.6 |  51.1 | 67.4 | 63.3 | 63.1 | 65.7 | -        | -       |
