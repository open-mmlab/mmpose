# InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image

## Results and Models

### InterHand2.6m 3D Dataset

| Arch                             |    Set    | MPJPE-single | MPJPE-interacting | MPJPE-all | MRRPE | APh  |               ckpt               |               log               |               Details and Download                |
| :------------------------------- | :-------: | :----------: | :---------------: | :-------: | :---: | :--: | :------------------------------: | :-----------------------------: | :-----------------------------------------------: |
| [InterNet_resnet_50](/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py) | test(H+M) |     9.47     |       13.40       |   11.59   | 29.28 | 0.99 | [ckpt](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth) | [log](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256_20210702.log.json) | [internet_interhand3d.md](./interhand3d/internet_interhand3d.md) |
| [InterNet_resnet_50](/configs/hand/3d_kpt_sview_rgb_img/internet/interhand3d/res50_interhand3d_all_256x256.py) |  val(M)   |    11.22     |       15.23       |   13.16   | 31.73 | 0.98 | [ckpt](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256-42b7f2ac_20210702.pth) | [log](https://download.openmmlab.com/mmpose/hand3d/internet/res50_intehand3dv1.0_all_256x256_20210702.log.json) | [internet_interhand3d.md](./interhand3d/internet_interhand3d.md) |
