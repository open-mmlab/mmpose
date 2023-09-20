# A simple yet effective baseline for 3d human pose estimation

Simple 3D baseline proposes to break down the task of 3d human pose estimation into 2 stages: (1) Image → 2D pose (2) 2D pose → 3D pose.

The authors find that "lifting" ground truth 2D joint locations to 3D space is a task that can be solved with a low error rate. Based on the success of 2d human pose estimation, it directly "lifts" 2d joint locations to 3d space.

## Results and Models

### Human3.6m Dataset

| Arch                                        | MPJPE | P-MPJPE |                    ckpt                     |                     log                     |                    Details and Download                     |
| :------------------------------------------ | :---: | :-----: | :-----------------------------------------: | :-----------------------------------------: | :---------------------------------------------------------: |
| [SimpleBaseline3D](/configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m.py) | 43.4  |  34.3   | [ckpt](https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth) | [log](https://download.openmmlab.com/mmpose/body3d/simple_baseline/20210415_065056.log.json) | [simplebaseline3d_h36m.md](./h36m/simplebaseline3d_h36m.md) |
