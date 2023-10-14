# MotionBERT: A Unified Perspective on Learning Human Motion Representations

Motionbert proposes a pretraining stage in which a motion encoder is trained to recover the underlying 3D motion from noisy partial 2D observations. The motion representations acquired in this way incorporate geometric, kinematic, and physical knowledge about human motion, which can be easily transferred to multiple downstream tasks.

## Results and Models

### Human3.6m Dataset

| Arch                                                                  | MPJPE | P-MPJPE |                                 ckpt                                  | log |              Details and Download               |
| :-------------------------------------------------------------------- | :---: | :-----: | :-------------------------------------------------------------------: | :-: | :---------------------------------------------: |
| [MotionBERT\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-243frm_8xb32-240e_h36m.py) | 35.3  |  27.7   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_h36m-f554954f_20230531.pth) |  /  | [motionbert_h36m.md](./h36m/motionbert_h36m.md) |
| [MotionBERT-finetuned\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py) | 27.5  |  21.6   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth) |  /  | [motionbert_h36m.md](./h36m/motionbert_h36m.md) |

### Human3.6m Dataset from official repo <sup>1</sup>

| Arch                                                           | MPJPE | Average MPJPE | P-MPJPE |                              ckpt                               | log |              Details and Download               |
| :------------------------------------------------------------- | :---: | :-----------: | :-----: | :-------------------------------------------------------------: | :-: | :---------------------------------------------: |
| [MotionBERT\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-243frm_8xb32-240e_h36m-original.py) | 39.8  |     39.2      |  33.4   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_h36m-f554954f_20230531.pth) |  /  | [motionbert_h36m.md](./h36m/motionbert_h36m.md) |
| [MotionBERT-finetuned\*](/configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m-original.py) | 37.7  |     37.2      |  32.2   | [ckpt](https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth) |  /  | [motionbert_h36m.md](./h36m/motionbert_h36m.md) |

<sup>1</sup> Please refer to the [doc](./h36m/motionbert_h36m.md) for more details.

*Models with * are converted from the official repo. The config files of these models are only for validation. We don't ensure these config files' training accuracy and welcome you to contribute your reproduction results.*
