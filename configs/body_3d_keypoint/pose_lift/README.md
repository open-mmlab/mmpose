# Single-view 3D Human Body Pose Estimation

## Video-based Single-view 3D Human Body Pose Estimation

Video-based 3D pose estimation is the detection and analysis of X, Y, Z coordinates of human body joints from a sequence of RGB images.

For single-person 3D pose estimation from a monocular camera, existing works can be classified into three categories:

(1) from 2D poses to 3D poses (2D-to-3D pose lifting)

(2) jointly learning 2D and 3D poses, and

(3) directly regressing 3D poses from images.

### Results and Models

#### Human3.6m Dataset

| Arch | Receptive Field | MPJPE | P-MPJPE | N-MPJPE | ckpt | log |

| :------------------------------------------------------ | :-------------: | :---: | :-----: | :-----: | :------------------------------------------------------: | :-----------------------------------------------------: |

| [VideoPose3D-supervised](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-27frm-supv_8xb128-80e_h36m.py) | 27 | 40.1 | 30.1 | / | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_supervised-fe8fbba9_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_supervised_20210527.log.json) |

| [VideoPose3D-supervised](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-81frm-supv_8xb128-80e_h36m.py) | 81 | 39.1 | 29.3 | / | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_81frames_fullconv_supervised-1f2d1104_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_81frames_fullconv_supervised_20210527.log.json) |

| [VideoPose3D-supervised](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-243frm-supv_8xb128-80e_h36m.py) | 243 | | | / | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_20210527.log.json) |

| [VideoPose3D-supervised-CPN](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-1frm-supv-cpn-ft_8xb128-80e_h36m.py) | 1 | 53.0 | 41.3 | / | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_1frame_fullconv_supervised_cpn_ft-5c3afaed_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_1frame_fullconv_supervised_cpn_ft_20210527.log.json) |

| [VideoPose3D-supervised-CPN](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-243frm-supv-cpn-ft_8xb128-200e_h36m.py) | 243 | | | / | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft_20210527.log.json) |

| [VideoPose3D-semi-supervised](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-27frm-semi-supv_8xb64-200e_h36m.py) | 27 | 57.2 | 42.4 | 54.2 | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised-54aef83b_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_20210527.log.json) |

| [VideoPose3D-semi-supervised-CPN](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_videopose3d-27frm-semi-supv-cpn-ft_8xb64-200e_h36m.py) | 27 | 67.3 | 50.4 | 63.6 | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft-71be9cde_20210527.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_27frames_fullconv_semi-supervised_cpn_ft_20210527.log.json) |

## Image-based Single-view 3D Human Body Pose Estimation

3D pose estimation is the detection and analysis of X, Y, Z coordinates of human body joints from an RGB image.
For single-person 3D pose estimation from a monocular camera, existing works can be classified into three categories:
(1) from 2D poses to 3D poses (2D-to-3D pose lifting)
(2) jointly learning 2D and 3D poses, and
(3) directly regressing 3D poses from images.

### Results and Models

#### Human3.6m Dataset

| Arch | MPJPE | P-MPJPE | N-MPJPE | ckpt | log |
| :------------------------------------------------------ | :-------------: | :---: | :-----: | :-----: | :------------------------------------------------------: | :-----------------------------------------------------: |
| [SimpleBaseline3D-tcn](/configs/body_3d_keypoint/pose_lift/h36m/pose-lift_simplebaseline3d_8xb64-200e_h36m.py) | 43.4 | 34.3 | /|[ckpt](https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth) | [log](https://download.openmmlab.com/mmpose/body3d/simple_baseline/20210415_065056.log.json) |
