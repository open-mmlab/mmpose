<!-- [ALGORITHM] -->

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
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
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

Results on Human3.6M dataset with ground truth 2D detections, supervised training

| Arch                                                       | Receptive Field | MPJPE | P-MPJPE |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :-------------: | :---: | :-----: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [VideoPose3D](/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_27frames_fullconv_supervised.py) |       27        | 40.0  |  30.1   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_supervised_20210527.log.json) |
| [VideoPose3D](/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_81frames_fullconv_supervised.py) |       81        | 38.9  |  29.2   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_81frames_fullconv_supervised-1f2d1104_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_81frames_fullconv_supervised_20210527.log.json) |
| [VideoPose3D](/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised.py) |       243       | 37.6  |  28.3   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_20210527.log.json) |

Results on Human3.6M dataset with CPN 2D detections<sup>1</sup>, supervised training

| Arch                                                       | Receptive Field | MPJPE | P-MPJPE |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :-------------: | :---: | :-----: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [VideoPose3D](/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_1frame_fullconv_supervised_cpn_ft.py) |        1        | 52.9  |  41.3   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_1frame_fullconv_supervised_cpn_ft-5c3afaed_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_1frame_fullconv_supervised_cpn_ft_20210527.log.json) |
| [VideoPose3D](/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py) |       243       | 47.9  |  38.0   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft_20210527.log.json) |

Results on Human3.6M dataset with ground truth 2D detections, semi-supervised training

| Training Data |                        Arch                         | Receptive Field | MPJPE | P-MPJPE | N-MPJPE |                        ckpt                         |                         log                         |
| :------------ | :-------------------------------------------------: | :-------------: | :---: | :-----: | :-----: | :-------------------------------------------------: | :-------------------------------------------------: |
| 10% S1        | [VideoPose3D](/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_27frames_fullconv_semi-supervised.py) |       27        | 58.1  |  42.8   |  54.7   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised-54aef83b_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_20210527.log.json) |

Results on Human3.6M dataset with CPN 2D detections<sup>1</sup>, semi-supervised training

| Training Data |                        Arch                         | Receptive Field | MPJPE | P-MPJPE | N-MPJPE |                        ckpt                         |                         log                         |
| :------------ | :-------------------------------------------------: | :-------------: | :---: | :-----: | :-----: | :-------------------------------------------------: | :-------------------------------------------------: |
| 10% S1        | [VideoPose3D](/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_27frames_fullconv_semi-supervised_cpn_ft.py) |       27        | 67.4  |  50.1   |  63.2   | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft-71be9cde_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft_20210527.log.json) |

<sup>1</sup> CPN 2D detections are provided by [official repo](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). The reformatted version used in this repository can be downloaded from [train_detection](https://download.openmmlab.com/mmpose/body3d/videopose/cpn_ft_h36m_dbb_train.npy) and [test_detection](https://download.openmmlab.com/mmpose/body3d/videopose/cpn_ft_h36m_dbb_test.npy).
