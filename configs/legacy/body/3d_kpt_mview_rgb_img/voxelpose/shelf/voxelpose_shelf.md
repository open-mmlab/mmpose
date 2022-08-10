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
<summary align="right"><a href="http://campar.in.tum.de/pub/belagiannis2014cvpr/belagiannis2014cvpr.pdf">Shelf (CVPR'2014)</a></summary>

```bibtex
@inproceedings {belagian14multi,
    title = {{3D} Pictorial Structures for Multiple Human Pose Estimation},
    author = {Belagiannis, Vasileios and Amin, Sikandar and Andriluka, Mykhaylo and Schiele, Bernt and Navab
    Nassir and Ilic, Slobo
    booktitle = {IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2014},
    month = {June},
    organization={IEEE}
}
```

</details>

Results on Shelf dataset.

| Arch                                                      | Actor 1 | Actor 2 | Actor 3 | Average |                            ckpt                            |                            log                            |
| :-------------------------------------------------------- | :-----: | :-----: | :-----: | :-----: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| [prn32_cpn48_res50](/configs/body/3d_kpt_mview_rgb_img/voxelpose/shelf/voxelpose_prn32x32x32_cpn48x48x12_shelf_cam5.py) |  99.10  |  94.86  |  97.52  |  97.16  | [ckpt](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn32x32x32_cpn48x48x12_shelf_cam5-24721ec7_20220323.pth) | [log](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn32x32x32_cpn48x48x12_shelf_cam5_20220323.log.json) |
| [prn64_cpn80_res50](/configs/body/3d_kpt_mview_rgb_img/voxelpose/shelf/voxelpose_prn64x64x64_cpn80x80x20_shelf_cam5.py) |  99.00  |  94.59  |  97.64  |  97.08  | [ckpt](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_shelf_cam5-f406fefe_20220323.pth) | [log](https://download.openmmlab.com/mmpose/body3d/voxelpose/voxelpose_prn64x64x64_cpn80x80x20_shelf_cam5_20220323.log.json) |
