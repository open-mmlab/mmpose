<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html">HRNet (CVPR'2019)</a></summary>

```bibtex
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="https://openaccess.thecvf.com/content/WACV2021/html/Mathis_Pretraining_Boosts_Out-of-Domain_Robustness_for_Pose_Estimation_WACV_2021_paper.html">Horse-10 (WACV'2021)</a></summary>

```bibtex
@inproceedings{mathis2021pretraining,
  title={Pretraining boosts out-of-domain robustness for pose estimation},
  author={Mathis, Alexander and Biasi, Thomas and Schneider, Steffen and Yuksekgonul, Mert and Rogers, Byron and Bethge, Matthias and Mathis, Mackenzie W},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1859--1868},
  year={2021}
}
```

</details>

Results on Horse-10 test set

| Set    |                           Arch                            | Input Size | PCK@0.3 |  NME  |                            ckpt                            |                            log                            |
| :----- | :-------------------------------------------------------: | :--------: | :-----: | :---: | :--------------------------------------------------------: | :-------------------------------------------------------: |
| split1 | [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w32_horse10_256x256-split1.py) |  256x256   |  0.951  | 0.122 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split1-401d901a_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split1_20210405.log.json) |
| split2 | [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w32_horse10_256x256-split2.py) |  256x256   |  0.949  | 0.116 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split2-04840523_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split2_20210405.log.json) |
| split3 | [pose_hrnet_w32](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w32_horse10_256x256-split3.py) |  256x256   |  0.939  | 0.153 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split3-4db47400_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_horse10_256x256_split3_20210405.log.json) |
| split1 | [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w48_horse10_256x256-split1.py) |  256x256   |  0.973  | 0.095 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split1-3c950d3b_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split1_20210405.log.json) |
| split2 | [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w48_horse10_256x256-split2.py) |  256x256   |  0.969  | 0.101 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split2-8ef72b5d_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split2_20210405.log.json) |
| split3 | [pose_hrnet_w48](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/hrnet_w48_horse10_256x256-split3.py) |  256x256   |  0.961  | 0.128 | [ckpt](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split3-0232ec47_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_horse10_256x256_split3_20210405.log.json) |
