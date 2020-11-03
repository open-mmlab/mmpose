# Simple baselines for human pose estimation and tracking

## Introduction
```
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
}
```

## Results and models

### 2d Hand Pose Estimation

#### Results on OneHand10K val set.

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/hand/resnet/onehand10k/res50_onehand10k_256x256.py) | 256x256 | 0.985 | 0.536 | 27.3 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256_20200813.log.json) |

#### Results on FreiHand val & test set.

| Set | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :--------: | :------: | :------: | :------: |:------: |:------: |
|val| [pose_resnet_50](/configs/hand/resnet/freihand/res50_freihand_224x224.py) | 224x224 | 0.993 | 0.868 | 3.25 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_freihand_224x224-ff0799bc_20200914.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_freihand_224x224_20200914.log.json) |
|test| [pose_resnet_50](/configs/hand/resnet/freihand/res50_freihand_224x224.py) | 224x224 | 0.992 | 0.868 | 3.27 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_freihand_224x224-ff0799bc_20200914.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_freihand_224x224_20200914.log.json) |

#### Results on CMU Panoptic (MPII+NZSL val set).

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/hand/resnet/panoptic/res50_panoptic_256x256.py) | 256x256 | 0.998 | 0.708 | 9.24 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_panoptic_256x256-5f55ca1a_20200925.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_panoptic_256x256_20200925.log.json) |

#### Results on InterHand2.6M val & test set.

|Train Set| Set | Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--- | :--------: | :--------: | :------: | :------: | :------: |:------: |:------: |
|Human_annot|val(M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.973 | 0.828 | 5.15 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(H)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.973 | 0.826 | 5.27 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.975 | 0.841 | 4.90 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Human_annot|test(H+M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_human_256x256.py) | 256x256 | 0.975 | 0.839 | 4.97 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human-77b27d1a_20201029.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_human_20201029.log.json) |
|Machine_annot|val(M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.970 | 0.824 | 5.39 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(H)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.969 | 0.821 | 5.52 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.972 | 0.838 | 5.03 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|Machine_annot|test(H+M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_machine_256x256.py) | 256x256 | 0.972 | 0.837 | 5.11 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine-8f3efe9a_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_machine_20201102.log.json) |
|All|val(M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.977 | 0.840 | 4.66 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(H)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.839 | 4.65 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.838 | 4.42 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
|All|test(H+M)| [pose_resnet_50](/configs/hand/resnet/interhand2d/res50_interhand2d_all_256x256.py) | 256x256 | 0.979 | 0.851 | 4.46 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all-78cc95d4_20201102.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_interhand2d_256x256_all_20201102.log.json) |
