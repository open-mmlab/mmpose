# Simple baselines for human pose estimation and tracking

## Introduction

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

## Results and models

### 2d Animal Landmark Detection

#### Results on Horse-10 test set

|Set   | Arch  | Input Size | PCK@0.3 |  NME  | ckpt    | log     |
| :--- | :---: | :--------: | :------: | :------: |:------: |:------: |
|split1| [pose_resnet_50](/configs/animal/resnet/horse10/res50_horse10_256x256-split1.py) | 256x256 | 0.956 | 0.113 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1-3a3dc37e_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_50](/configs/animal/resnet/horse10/res50_horse10_256x256-split2.py) | 256x256 | 0.954 | 0.111 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2-65e2a508_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_50](/configs/animal/resnet/horse10/res50_horse10_256x256-split3.py) | 256x256 | 0.946 | 0.129 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3-9637d4eb_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_resnet_101](/configs/animal/resnet/horse10/res101_horse10_256x256-split1.py) | 256x256 | 0.958 | 0.115 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1-1b7c259c_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_101](/configs/animal/resnet/horse10/res101_horse10_256x256-split2.py) | 256x256 | 0.955 | 0.115 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2-30e2fa87_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_101](/configs/animal/resnet/horse10/res101_horse10_256x256-split3.py) | 256x256 | 0.946 | 0.126 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3-2eea5bb1_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3_20210405.log.json) |
|split1| [pose_resnet_152](/configs/animal/resnet/horse10/res152_horse10_256x256-split1.py) | 256x256 | 0.969 | 0.105 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1-7e81fe2d_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1_20210405.log.json) |
|split2| [pose_resnet_152](/configs/animal/resnet/horse10/res152_horse10_256x256-split2.py) | 256x256 | 0.970 | 0.103 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2-3b3404a3_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2_20210405.log.json) |
|split3| [pose_resnet_152](/configs/animal/resnet/horse10/res152_horse10_256x256-split3.py) | 256x256 | 0.957 | 0.131 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3-c957dac5_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3_20210405.log.json) |

#### Results on MacaquePose with ground-truth detection bounding boxes

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/animal/resnet/macaque/res50_macaque_256x192.py)  | 256x192 | 0.799 | 0.952 | 0.919 | 0.837 | 0.964 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192-98f1dd3a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_macaque_256x192_20210407.log.json) |
| [pose_resnet_101](/configs/animal/resnet/macaque/res101_macaque_256x192.py) | 256x192 | 0.790 | 0.953 | 0.908 | 0.828 | 0.967 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_macaque_256x192-e3b9c6bb_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_macaque_256x192_20210407.log.json) |
| [pose_resnet_152](/configs/animal/resnet/macaque/res152_macaque_256x192.py) | 256x192 | 0.794 | 0.951 | 0.915 | 0.834 | 0.968 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_macaque_256x192-c42abc02_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_macaque_256x192_20210407.log.json) |

#### Results on Vinegar Fly test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/resnet/fly/res50_fly_192x192.py) | 192x192 | 0.996 | 0.910 | 2.00 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_fly_192x192-5d0ee2d9_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_fly_192x192_20210407.log.json) |
|[pose_resnet_101](/configs/animal/resnet/fly/res101_fly_192x192.py) | 192x192 | 0.996 | 0.912 | 1.95 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_fly_192x192-41a7a6cc_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_fly_192x192_20210407.log.json) |
|[pose_resnet_152](/configs/animal/resnet/fly/res152_fly_192x192.py) | 192x192 | 0.997 | 0.917 | 1.78 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_fly_192x192-fcafbd5a_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_fly_192x192_20210407.log.json) |

#### Results on Desert Locust test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/resnet/locust/res50_locust_160x160.py) | 160x160 | 0.999 | 0.899 | 2.27 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160-9efca22b_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_locust_160x160_20210407.log.json) |
|[pose_resnet_101](/configs/animal/resnet/locust/res101_locust_160x160.py) | 160x160 | 0.999 | 0.907 | 2.03 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160-d77986b3_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_locust_160x160_20210407.log.json) |
|[pose_resnet_152](/configs/animal/resnet/locust/res152_locust_160x160.py) | 160x160 | 1.000 | 0.926 | 1.48 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160-4ea9b372_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_locust_160x160_20210407.log.json) |

#### Results on Grévy’s Zebra test set

|  Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :-------- | :--------: | :------: | :------: | :------: |:------: |:------: |
|[pose_resnet_50](/configs/animal/resnet/zebra/res50_zebra_160x160.py) | 160x160 | 1.000 | 0.914 | 1.86 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_zebra_160x160-5a104833_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_zebra_160x160_20210407.log.json) |
|[pose_resnet_101](/configs/animal/resnet/zebra/res101_zebra_160x160.py) | 160x160 | 1.000 | 0.916 | 1.82 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_zebra_160x160-e8cb2010_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_zebra_160x160_20210407.log.json) |
|[pose_resnet_152](/configs/animal/resnet/zebra/res152_zebra_160x160.py) | 160x160 | 1.000 | 0.921 | 1.66 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_zebra_160x160-05de71dd_20210407.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_zebra_160x160_20210407.log.json) |

#### Results on ATRW validation set

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/animal/resnet/atrw/res50_atrw_256x256.py)  | 256x256 | 0.900 | 0.973 | 0.932 | 0.929 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_atrw_256x256-546c4594_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_atrw_256x256_20210414.log.json) |
| [pose_resnet_101](/configs/animal/resnet/atrw/res101_atrw_256x256.py) | 256x256 | 0.898 | 0.973 | 0.936 | 0.927 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_atrw_256x256-da93f371_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_atrw_256x256_20210414.log.json) |
| [pose_resnet_152](/configs/animal/resnet/atrw/res152_atrw_256x256.py) | 256x256 | 0.896 | 0.973 | 0.931 | 0.927 | 0.985 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_atrw_256x256-2bb8e162_20210414.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_atrw_256x256_20210414.log.json) |
