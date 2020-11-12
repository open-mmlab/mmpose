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


#### Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [pose_resnet_50](/configs/top_down/resnet/coco-wholebody/res50_coco_wholebody_256x192.py)  | 256x192 | 0.652 | 0.739 | 0.614 | 0.746 | 0.608 | 0.716 | 0.460 | 0.584 | 0.457 | 0.578 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192-9e37ed88_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_50](/configs/top_down/resnet/coco-wholebody/res50_coco_wholebody_384x288.py)  | 384x288 | 0.666 | 0.747 | 0.635 | 0.763 | 0.732 | 0.812 | 0.537 | 0.647 | 0.573 | 0.671 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288-ce11e294_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/coco-wholebody/res101_coco_wholebody_256x192.py)  | 256x192 | 0.670 | 0.754 | 0.640 | 0.767 | 0.611 | 0.723 | 0.463 | 0.589 | 0.533 | 0.647 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192-7325f982_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/coco-wholebody/res101_coco_wholebody_384x288.py)  | 384x288 | 0.692 | 0.770 | 0.680 | 0.798 | 0.747 | 0.822 | 0.549 | 0.658 | 0.597 | 0.692 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288-6c137b9a_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_wholebody_384x288_20201004.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/coco-wholebody/res152_coco_wholebody_256x192.py)  | 256x192 | 0.682 | 0.764 | 0.662 | 0.788 | 0.624 | 0.728 | 0.482 | 0.606 | 0.548 | 0.661 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192-5de8ae23_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_256x192_20201004.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/coco-wholebody/res152_coco_wholebody_384x288.py)  | 384x288 | 0.703 | 0.780 | 0.693 | 0.813 | 0.751 | 0.825 | 0.559 | 0.667 | 0.610 | 0.705 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288-eab8caa8_20201004.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_wholebody_384x288_20201004.log.json) |


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


#### Results on AIC val set.

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_resnet_101](/configs/top_down/resnet/aic/res101_aic_256x192.py) | 256x192 | 0.650 | 0.947 | 0.726 | 0.680 | 0.954 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192-79b35445_20200826.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_aic_256x192_20200826.log.json) |


#### Results on MPII val set.

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/mpii/res50_mpii_256x256.py) | 256x256 | 0.882 | 0.329 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256_20200812.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/mpii/res101_mpii_256x256.py) | 256x256 | 0.887 | 0.333 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256-416f5d71_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256_20200812.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/mpii/res152_mpii_256x256.py) | 256x256 | 0.890 | 0.347 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256-3ecba29d_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256_20200812.log.json) |


#### Results on MPII-TRB val set.

| Arch  | Input Size | Skeleton Acc   | Contour Acc   | Mean Acc | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/mpii_trb/res50_mpii_trb_256x256.py)  | 256x256 | 0.887 | 0.858 | 0.868 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_trb_256x256-896036b8_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_trb_256x256_20200812.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/mpii_trb/res101_mpii_trb_256x256.py)  | 256x256 | 0.890 | 0.863 | 0.873 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_trb_256x256-cfad2f05_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_trb_256x256_20200812.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/mpii_trb/res152_mpii_trb_256x256.py)  | 256x256 | 0.897 | 0.868 | 0.879 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_trb_256x256-dd369ce6_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_trb_256x256_20200812.log.json) |


#### Results on CrowdPose test with [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) human detector.

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AP (E) | AP (M) | AP (H) | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: | :------: |
| [pose_resnet_50](/configs/top_down/resnet/crowdpose/res50_crowdpose_256x192.py)  | 256x192 | 0.634 | 0.805 | 0.689 | 0.742 | 0.646 | 0.503 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192-f1ddd03a_20201017.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_crowdpose_256x192_20201017.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/crowdpose/res101_crowdpose_256x192.py)  | 256x192 | 0.647 | 0.814 | 0.703 | 0.748 | 0.658 | 0.518 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192-ad003be3_20201017.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_256x192_20201017.log.json) |
| [pose_resnet_101](/configs/top_down/resnet/crowdpose/res101_crowdpose_320x256.py)  | 320x256 | 0.659 | 0.819 | 0.713 | 0.757 | 0.669 | 0.536 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256-7723ede3_20201017.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_crowdpose_320x256_20201017.log.json) |
| [pose_resnet_152](/configs/top_down/resnet/crowdpose/res152_crowdpose_256x192.py)  | 256x192 | 0.653 | 0.815 | 0.710 | 0.751 | 0.665 | 0.526 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192-a6507ad0_20201017.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_crowdpose_256x192_20201017.log.json) |

#### Results on PoseTrack2018 val with ground-truth bounding boxes.

| Arch  | Input Size | Head | Shou | Elb | Wri | Hip | Knee | Ankl | Total  | ckpt    | log     |
| :--- | :--------: | :------: |:------: |:------: |:------: |:------: |:------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/posetrack18/res50_posetrack18_256x192.py) | 256x192 | 86.5 | 87.5 | 82.3 | 75.6 | 79.9 | 78.6 | 74.0 | 81.0 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192-a62807c7_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.

#### Results on PoseTrack2018 val with [MMDetection](https://github.com/open-mmlab/mmdetection) pre-trained [Cascade R-CNN](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) (X-101-64x4d-FPN) human detector.

| Arch  | Input Size | Head | Shou | Elb | Wri | Hip | Knee | Ankl | Total  | ckpt    | log     |
| :--- | :--------: | :------: |:------: |:------: |:------: |:------: |:------: | :------: | :------: |:------: |:------: |
| [pose_resnet_50](/configs/top_down/resnet/posetrack18/res50_posetrack18_256x192.py) | 256x192 | 78.9 | 81.9 | 77.8 | 70.8 | 75.3 | 73.2 | 66.4 | 75.2 | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192-a62807c7_20201028.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_posetrack18_256x192_20201028.log.json) |

The models are first pre-trained on COCO dataset, and then fine-tuned on PoseTrack18.
