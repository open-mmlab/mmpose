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
<summary align="right"><a href="https://ieeexplore.ieee.org/abstract/document/8374605/">MPI-INF-3DHP (3DV'2017)</a></summary>

```bibtex
@inproceedings{mono-3dhp2017,
  author = {Mehta, Dushyant and Rhodin, Helge and Casas, Dan and Fua, Pascal and Sotnychenko, Oleksandr and Xu, Weipeng and Theobalt, Christian},
  title = {Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision},
  booktitle = {3D Vision (3DV), 2017 Fifth International Conference on},
  url = {http://gvv.mpi-inf.mpg.de/3dhp_dataset},
  year = {2017},
  organization={IEEE},
  doi={10.1109/3dv.2017.00064},
}
```

</details>

Results on MPI-INF-3DHP dataset with ground truth 2D detections, supervised training

| Arch | Receptive Field | MPJPE | P-MPJPE | 3DPCK | 3DAUC | ckpt | log |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [VideoPose3D](configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/mpi_inf_3dhp/videopose3d_mpi-inf-3dhp_1frame_fullconv_supervised_gt.py) | 1 | 58.3 | 40.6 | 94.1 | 63.1 | [ckpt](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_mpi-inf-3dhp_1frame_fullconv_supervised_gt-d6ed21ef_20210603.pth) | [log](https://download.openmmlab.com/mmpose/body3d/videopose/videopose_mpi-inf-3dhp_1frame_fullconv_supervised_gt_20210603.log.json) |
