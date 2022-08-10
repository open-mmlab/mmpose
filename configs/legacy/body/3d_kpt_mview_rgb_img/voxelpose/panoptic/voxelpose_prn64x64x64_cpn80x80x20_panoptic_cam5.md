<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460188.pdf">VoxelPose (ECCV'2020)</a></summary>

```bibtex
@inproceedings{tumultipose,
  title={VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment},
  author={Tu, Hanyue and Wang, Chunyu and Zeng, Wenjun},
  booktitle={ECCV},
  year={2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content_iccv_2015/html/Joo_Panoptic_Studio_A_ICCV_2015_paper.html">CMU Panoptic (ICCV'2015)</a></summary>

```bibtex
@Article = {joo_iccv_2015,
author = {Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh},
title = {Panoptic Studio: A Massively Multiview System for Social Motion Capture},
booktitle = {ICCV},
year = {2015}
}
```

</details>

Results on CMU Panoptic dataset.

| Arch                                                       |  mAP  |  mAR  | MPJPE | Recall@500mm |                            ckpt                            |                            log                            |
| :--------------------------------------------------------- | :---: | :---: | :---: | :----------: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [prn64_cpn80_res50](/configs/body/3d_kpt_mview_rgb_img/voxelpose/panoptic/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5.py) | 97.31 | 97.99 | 17.57 |    99.85     | [ckpt](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5-545c150e_20211103.pth) | [log](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_panoptic_cam5_20211103.log.json) |
