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
<summary align="right"><a href="http://campar.in.tum.de/pub/belagiannis2014cvpr/belagiannis2014cvpr.pdf">Campus (CVPR'2014)</a></summary>

```bibtex
@inproceedings {belagian14multi,
    title = {{3D} Pictorial Structures for Multiple Human Pose Estimation},
    author = {Belagiannis, Vasileios and Amin, Sikandar and Andriluka, Mykhaylo and Schiele, Bernt and Navab
    Nassir and Ilic, Slobodan},
    booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2014},
    month = {June},
    organization={IEEE}
}
```

</details>

Results on Campus dataset.

| Arch                                                      | Actor 1 | Actor 2 | Actor 3 | Average |                            ckpt                            |                            log                            |
| :-------------------------------------------------------- | :-----: | :-----: | :-----: | :-----: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [prn32_cpn80_res50](/configs/body/3d_kpt_mview_rgb_img/voxelpose/campus/voxelpose_prn32x32x32_cpn80x80x20_campus_cam3.py) |  97.76  |  93.92  |  98.48  |  96.72  | [ckpt](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn32x32x32_cpn80x80x20_campus_cam3-3ecee30e_20220323.pth) | [log](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn32x32x32_cpn80x80x20_campus_cam3_20220323.log.json) |
| [prn64_cpn80_res50](/configs/body/3d_kpt_mview_rgb_img/voxelpose/campus/voxelpose_prn64x64x64_cpn80x80x20_campus_cam3.py) |  97.76  |  93.33  |  98.77  |  96.62  | [ckpt](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_campus_cam3-d8decbf7_20220323.pth) | [log](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_campus_cam3_20220323.log.json) |
