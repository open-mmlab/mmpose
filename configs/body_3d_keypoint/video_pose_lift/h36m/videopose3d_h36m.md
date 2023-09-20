<!-- [BACKBONE] -->

<details>

<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html">VideoPose3D (CVPR'2019)</a></summary>

```bibtex
@inproceedings{pavllo20193d,
title={3d human pose estimation in video with temporal convolutions and semi-supervised training},
author={Pavllo, Dario and Feichtenhofer, Christoph and Grangier, David and Auli, Michael},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={7753--7762},
year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/6682899/">Human3.6M (TPAMI'2014)</a></summary>

```bibtex
@article{h36m_pami,
author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu, Cristian},
title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
publisher = {IEEE Computer Society},
volume = {36},
number = {7},
pages = {1325-1339},
month = {jul},
year = {2014}
}
```

</details>

Testing results on Human3.6M dataset with ground truth 2D detections, supervised training

| Arch                                                       | Receptive Field | MPJPE | P-MPJPE |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :-------------: | :---: | :-----: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [VideoPose3D-supervised-27frm](/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-27frm-supv_8xb128-160e_h36m.py) |       27        | 40.1  |  30.1   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_supervised_20210527.log.json) |
| [VideoPose3D-supervised-81frm](/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-81frm-supv_8xb128-160e_h36m.py) |       81        | 39.1  |  29.3   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_81frames_fullconv_supervised-1f2d1104_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_81frames_fullconv_supervised_20210527.log.json) |
| [VideoPose3D-supervised-243frm](/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m.py) |       243       | 37.6  |  28.3   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_20210527.log.json) |

Testing results on Human3.6M dataset with CPN 2D detections<sup>1</sup>, supervised training

| Arch                                                       | Receptive Field | MPJPE | P-MPJPE |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :-------------: | :---: | :-----: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [VideoPose3D-supervised-CPN-1frm](/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-1frm-supv-cpn-ft_8xb128-160e_h36m.py) |        1        | 53.0  |  41.3   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_1frame_fullconv_supervised_cpn_ft-5c3afaed_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_1frame_fullconv_supervised_cpn_ft_20210527.log.json) |
| [VideoPose3D-supervised-CPN-243frm](/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv-cpn-ft_8xb128-200e_h36m.py) |       243       | 47.9  |  38.0   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft_20210527.log.json) |

Testing results on Human3.6M dataset with ground truth 2D detections, semi-supervised training

| Training Data |                        Arch                         | Receptive Field | MPJPE | P-MPJPE | N-MPJPE |                        ckpt                         |                         log                         |
| :------------ | :-------------------------------------------------: | :-------------: | :---: | :-----: | :-----: | :-------------------------------------------------: | :-------------------------------------------------: |
| 10% S1        | [VideoPose3D-semi-supervised-27frm](/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-27frm-semi-supv_8xb64-200e_h36m.py) |       27        | 57.2  |  42.4   |  54.2   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised-54aef83b_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_20210527.log.json) |

Testing results on Human3.6M dataset with CPN 2D detections<sup>1</sup>, semi-supervised training

| Training Data |                        Arch                         | Receptive Field | MPJPE | P-MPJPE | N-MPJPE |                        ckpt                         |                         log                         |
| :------------ | :-------------------------------------------------: | :-------------: | :---: | :-----: | :-----: | :-------------------------------------------------: | :-------------------------------------------------: |
| 10% S1        | [VideoPose3D-semi-supervised-CPN-27frm](/configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-27frm-semi-supv-cpn-ft_8xb64-200e_h36m.py) |       27        | 67.3  |  50.4   |  63.6   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft-71be9cde_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft_20210527.log.json) |

<sup>1</sup> CPN 2D detections are provided by [official repo](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). The reformatted version used in this repository can be downloaded from [train_detection](https://download.openmmlab.com/mmpose/body3d/videopose/cpn_ft_h36m_dbb_train.npy) and [test_detection](https://download.openmmlab.com/mmpose/body3d/videopose/cpn_ft_h36m_dbb_test.npy).
