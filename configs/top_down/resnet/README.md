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

### 2d Human Pose Estimation

#### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/coco/res50_coco_256x192.py)  | 256x192 | 0.718 | 0.898 | 0.795 | 0.773 | 0.937 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_20200709.log.json) |
| [pose_resnet_50](/configs/top_down/resnet/coco/res50_coco_384x288.py)  | 384x288 | 0.731 | 0.900 | 0.799 | 0.783 | 0.931 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288-e6f795e9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_20200709.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/coco/res101_coco_256x192.py) | 256x192 | 0.726 | 0.899 | 0.806 | 0.781 | 0.939 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_20200708.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/coco/res101_coco_384x288.py) | 384x288 | 0.748 | 0.905 | 0.817 | 0.798 | 0.940 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288-8c71bdc9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_20200709.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/coco/res152_coco_256x192.py) | 256x192 | 0.735 | 0.905 | 0.812 | 0.790 | 0.943 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_20200709.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/coco/res152_coco_384x288.py) | 384x288 | 0.750 | 0.908 | 0.821 | 0.800 | 0.942 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_20200709.log.json) |

#### Results on OCHuman test dataset with ground-truth bounding boxes

Following the common setting, the models are trained on COCO train dataset, and evaluate on OCHuman dataset.

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/coco/res50_coco_256x192.py)  | 256x192 | 0.546 | 0.726 | 0.593 | 0.592 | 0.755 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192_20200709.log.json) |
| [pose_resnet_50](/configs/top_down/resnet/coco/res50_coco_384x288.py)  | 384x288 | 0.539 | 0.723 | 0.574 | 0.588 | 0.756 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288-e6f795e9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_384x288_20200709.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/coco/res101_coco_256x192.py) | 256x192 | 0.559 | 0.724 | 0.606 | 0.605 | 0.751 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192_20200708.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/coco/res101_coco_384x288.py) | 384x288 | 0.571 | 0.715 | 0.615 | 0.615 | 0.748 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288-8c71bdc9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_384x288_20200709.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/coco/res152_coco_256x192.py) | 256x192 | 0.570 | 0.725 | 0.617 | 0.616 | 0.754 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192_20200709.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/coco/res152_coco_384x288.py) | 384x288 | 0.582 | 0.723 | 0.627 | 0.627 | 0.752 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288_20200709.log.json) |

#### Results on AIC val set with ground-truth bounding boxes

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_101](/configs/top_down/resnet/aic/res101_aic_256x192.py) | 256x192 | 0.294 | 0.736 | 0.174 | 0.337 | 0.763 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192-79b35445_20200826.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192_20200826.log.json) |

#### Results on MHP v2.0 val set

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_101](/configs/top_down/resnet/mhp/res50_mhp_256x192.py) | 256x192 | 0.583 | 0.897 | 0.669 | 0.636 | 0.918 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mhp_256x192-28c5b818_20201229.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mhp_256x192_20201229.log.json) |

Note that, the evaluation metric used here is mAP (adapted from COCO), which may be different from the official evaluation [codes](https://github.com/ZhaoJ9014/Multi-Human-Parsing/tree/master/Evaluation/Multi-Human-Pose).
Please be cautious if you use the results in papers.

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/mpii/res50_mpii_256x256.py) | 256x256 | 0.882 | 0.286 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256_20200812.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/mpii/res101_mpii_256x256.py) | 256x256 | 0.888 | 0.290 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256-416f5d71_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256_20200812.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/mpii/res152_mpii_256x256.py) | 256x256 | 0.889 | 0.303 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256-3ecba29d_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256_20200812.log.json) |

#### Results on MPII-TRB val set

| Arch  | Input Size | Skeleton Acc   | Contour Acc   | Mean Acc | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/mpii_trb/res50_mpii_trb_256x256.py)  | 256x256 | 0.887 | 0.858 | 0.868 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_trb_256x256-896036b8_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_trb_256x256_20200812.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/mpii_trb/res101_mpii_trb_256x256.py)  | 256x256 | 0.890 | 0.863 | 0.873 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_trb_256x256-cfad2f05_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_trb_256x256_20200812.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/mpii_trb/res152_mpii_trb_256x256.py)  | 256x256 | 0.897 | 0.868 | 0.879 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_trb_256x256-dd369ce6_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_trb_256x256_20200812.log.json) |

#### Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |
| [pose_resnet_50](/configs/top_down/resnet/crowdpose/res50_crowdpose_256x192.py)  | 256x192 | 0.637 | 0.808 | 0.692 | 0.739 | 0.650 | 0.506 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192-c6a526b6_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192_20201227.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/crowdpose/res101_crowdpose_256x192.py)  | 256x192 | 0.647 | 0.810 | 0.703 | 0.744 | 0.658 | 0.522 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192-8f5870f4_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192_20201227.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/crowdpose/res101_crowdpose_320x256.py)  | 320x256 | 0.661 | 0.821 | 0.714 | 0.759 | 0.671 | 0.536 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256-c88c512a_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256_20201227.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/crowdpose/res152_crowdpose_256x192.py)  | 256x192 | 0.656 | 0.818 | 0.712 | 0.754 | 0.666 | 0.532 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192-dbd49aba_20201227.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192_20201227.log.json) |

#### Results on PoseTrack2018 val with ground-truth bounding boxes

| Arch  | Input Size | Head | Shou | Elb | Wri | Hip | Knee | Ankl | Total  | ckpt    | log     |
| :--- | :--------: | :------: |:------: |:------: |:------: |:------: |:------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/posetrack18/res50_posetrack18_256x192.py) | 256x192 | 86.5 | 87.5 | 82.3 | 75.6 | 79.9 | 78.6 | 74.0 | 81.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192-a62807c7_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

#### Results on PoseTrack2018 val with [MMDetection](https://github.com/open-mmlab/mmdetection) pre-trained [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) (X-101-64x4d-FPN) human detector

| Arch  | Input Size | Head | Shou | Elb | Wri | Hip | Knee | Ankl | Total  | ckpt    | log     |
| :--- | :--------: | :------: |:------: |:------: |:------: |:------: |:------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/posetrack18/res50_posetrack18_256x192.py) | 256x192 | 78.9 | 81.9 | 77.8 | 70.8 | 75.3 | 73.2 | 66.4 | 75.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192-a62807c7_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

#### Results on Sub-JHMDB dataset

The models are pre-trained on MPII dataset only. *NO* test-time augmentation (multi-scale /rotation testing) is used.

##### Normalized by Person Size

| Split| Arch        | Input Size | Head | Sho  | Elb | Wri | Hip | Knee | Ank | Mean | ckpt    | log     |
| :--- | :--------:  | :--------: | :---: | :---: |:---: |:---: |:---: |:---:  |:---: | :---: | :-----: |:------: |
| Sub1 |  [pose_resnet_50](/configs/top_down/resnet/jhmdb/res50_jhmdb_sub1_256x256.py) | 256x256 | 99.1 | 98.0 | 93.8 |  91.3 | 99.4 | 96.5| 92.8 | 96.1 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256-932cb3b4_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2 |  [pose_resnet_50](/configs/top_down/resnet/jhmdb/res50_jhmdb_sub2_256x256.py) | 256x256 | 99.3 | 97.1 | 90.6 |  87.0 | 98.9 | 96.3| 94.1 | 95.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256-83d606f7_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3 |  [pose_resnet_50](/configs/top_down/resnet/jhmdb/res50_jhmdb_sub3_256x256.py) | 256x256 | 99.0 | 97.9 | 94.0 |  91.6 | 99.7 | 98.0| 94.7 | 96.7 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256_20201122.log.json) |
| Average |  pose_resnet_50                                                            | 256x256 | 99.2 | 97.7 | 92.8 |  90.0 | 99.3 | 96.9| 93.9 | 96.0 | -        | -       |
| Sub1 |  [pose_resnet_50 (2 Deconv.)](/configs/top_down/resnet/jhmdb/res50_2deconv_jhmdb_sub1_256x256.py) | 256x256 | 99.1 | 98.5 | 94.6 |  92.0 | 99.4 | 94.6| 92.5 | 96.1 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256-f0574a52_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2 |  [pose_resnet_50 (2 Deconv.)](/configs/top_down/resnet/jhmdb/res50_2deconv_jhmdb_sub2_256x256.py) | 256x256 | 99.3 | 97.8 | 91.0 |  87.0 | 99.1 | 96.5| 93.8 | 95.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256-f63af0ff_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3 |  [pose_resnet_50 (2 Deconv.)](/configs/top_down/resnet/jhmdb/res50_2deconv_jhmdb_sub3_256x256.py) | 256x256 | 98.8 | 98.4 | 94.3 |  92.1 | 99.8 | 97.5| 93.8 | 96.7 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256-c4bc2ddb_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256_20201122.log.json) |
| Average |  pose_resnet_50 (2 Deconv.)                                                                    | 256x256 | 99.1 | 98.2 | 93.3 |  90.4 | 99.4 | 96.2| 93.4 | 96.0 | -        | -       |

##### Normalized by Torso Size

| Split| Arch        | Input Size | Head | Sho  | Elb | Wri | Hip | Knee | Ank | Mean | ckpt    | log     |
| :--- | :--------:  | :--------: | :---: | :---: |:---: |:---: |:---: |:---:  |:---: | :---: | :-----: |:------: |
| Sub1 |  [pose_resnet_50](/configs/top_down/resnet/jhmdb/res50_jhmdb_sub1_256x256.py) | 256x256 | 93.3 | 83.2 | 74.4 |  72.7 | 85.0 | 81.2 | 78.9 | 81.9 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256-932cb3b4_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2 |  [pose_resnet_50](/configs/top_down/resnet/jhmdb/res50_jhmdb_sub2_256x256.py) | 256x256 | 94.1 | 74.9 | 64.5 |  62.5 | 77.9 | 71.9 | 78.6 | 75.5 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256-83d606f7_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3 |  [pose_resnet_50](/configs/top_down/resnet/jhmdb/res50_jhmdb_sub3_256x256.py) | 256x256 | 97.0 | 82.2 | 74.9 |  70.7 | 84.7 | 83.7 | 84.2 | 82.9 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256-c4ec1a0b_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_jhmdb_sub3_256x256_20201122.log.json) |
| Average |  pose_resnet_50                                                            | 256x256 | 94.8 | 80.1 | 71.3 |  68.6 | 82.5 | 78.9 | 80.6 | 80.1 | -        | -       |
| Sub1 |  [pose_resnet_50 (2 Deconv.)](/configs/top_down/resnet/jhmdb/res50_2deconv_jhmdb_sub1_256x256.py) | 256x256 | 92.4 | 80.6 | 73.2 |  70.5 | 82.3 | 75.4| 75.0 | 79.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256-f0574a52_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub1_256x256_20201122.log.json) |
| Sub2 |  [pose_resnet_50 (2 Deconv.)](/configs/top_down/resnet/jhmdb/res50_2deconv_jhmdb_sub2_256x256.py) | 256x256 | 93.4 | 73.6 | 63.8 |  60.5 | 75.1 | 68.4| 75.5 | 73.7 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256-f63af0ff_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub2_256x256_20201122.log.json) |
| Sub3 |  [pose_resnet_50 (2 Deconv.)](/configs/top_down/resnet/jhmdb/res50_2deconv_jhmdb_sub3_256x256.py) | 256x256 | 96.1 | 81.2 | 72.6 |  67.9 | 83.6 | 80.9| 81.5 | 81.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256-c4bc2ddb_20201122.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_2deconv_jhmdb_sub3_256x256_20201122.log.json) |
| Average |  pose_resnet_50 (2 Deconv.)                                                                    | 256x256 | 94.0 | 78.5 | 69.9 |  66.3 | 80.3 | 74.9| 77.3 | 78.0 | -        | -       |
