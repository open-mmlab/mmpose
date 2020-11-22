# Convolutional pose machines

## Introduction
```
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


#### Results on MPII val set.

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [cpm](/configs/top_down/cpm/mpii/cpm_mpii_368x368.py) | 368x368 | 0.876 | 0.325 | [ckpt](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368-116e62b8_20200822.pth) | [log](https://download.openmmlab.com/mmpose/top_down/cpm/cpm_mpii_368x368_20200822.log.json) |


#### Results on Sub-JHMDB dataset.
The models are pre-trained on MPII dataset only. NO test-time augmentation (multi-scale /rotation testing) is used.

#####Normalized by Person Size

| Split| Arch        | Input Size | Head | Sho  | Elb | Wri | Hip | Knee | Ank | Mean | ckpt    | log     |
| :--- | :--------:  | :--------: | :---: | :---: |:---: |:---: |:---: |:---:  |:---: | :---: | :-----: |:------: |
| Sub1 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub1_368x368.py) | 368x368 | 99.4 | 76.9 | 92.9 |  89.1 | 83.4 | 86.6| 98.0 | 89.5 | -        | -       |
| Sub2 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub2_368x368.py) | 368x368 | 98.8 | 95.0 | 84.4 |  77.1 | 84.1 | 80.0| 94.7 | 87.4 | -        | -       |
| Sub3 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub3_368x368.py) | 368x368 | 94.8 | 98.4 | 87.9 |  86.6 | 86.5 | 93.8| 95.8 | 92.5 | -        | -       |
| Average |  cpm                                                       | 368x368 | 97.7 | 90.1 | 88.4 |  84.3 | 84.7 | 86.8| 96.2 | 89.8 | -        | -       |


#####Normalized by Torso Size

| Split| Arch        | Input Size | Head | Sho  | Elb | Wri | Hip | Knee | Ank | Mean | ckpt    | log     |
| :--- | :--------:  | :--------: | :---: | :---: |:---: |:---: |:---: |:---:  |:---: | :---: | :-----: |:------: |
| Sub1 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub1_368x368.py) | 368x368 | 88.7 | 46.4 | 75.3 |  53.3 | 55.0 | 59.6 | 85.7 | 65.8 | -        | -       |
| Sub2 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub2_368x368.py) | 368x368 | 86.5 | 76.7 | 50.2 |  41.3 | 50.8 | 54.8 | 74.2 | 60.9 | -        | -       |
| Sub3 |  [cpm](/configs/top_down/cpm/jhmdb/cpm_jhmdb_sub3_368x368.py) | 368x368 | 72.0 | 83.2 | 57.1 |  59.2 | 65.3 | 73.7 | 70.9 | 70.4 | -        | -       |
| Average |  cpm                                                       | 368x368 | 82.4 | 68.8 | 60.9 |  51.3 | 57.0 | 62.7 | 76.9 | 65.7 | -        | -       |
