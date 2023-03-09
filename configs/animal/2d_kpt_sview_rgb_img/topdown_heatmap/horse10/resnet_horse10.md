<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html">SimpleBaseline2D (ECCV'2018)</a></summary>

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
  year={2018}
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
| split1 | [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split1.py) |  256x256   |  0.956  | 0.113 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1-3a3dc37e_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split1_20210405.log.json) |
| split2 | [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split2.py) |  256x256   |  0.954  | 0.111 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2-65e2a508_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split2_20210405.log.json) |
| split3 | [pose_resnet_50](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res50_horse10_256x256-split3.py) |  256x256   |  0.946  | 0.129 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3-9637d4eb_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res50_horse10_256x256_split3_20210405.log.json) |
| split1 | [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res101_horse10_256x256-split1.py) |  256x256   |  0.958  | 0.115 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1-1b7c259c_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split1_20210405.log.json) |
| split2 | [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res101_horse10_256x256-split2.py) |  256x256   |  0.955  | 0.115 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2-30e2fa87_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split2_20210405.log.json) |
| split3 | [pose_resnet_101](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res101_horse10_256x256-split3.py) |  256x256   |  0.946  | 0.126 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3-2eea5bb1_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res101_horse10_256x256_split3_20210405.log.json) |
| split1 | [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res152_horse10_256x256-split1.py) |  256x256   |  0.969  | 0.105 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1-7e81fe2d_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split1_20210405.log.json) |
| split2 | [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res152_horse10_256x256-split2.py) |  256x256   |  0.970  | 0.103 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2-3b3404a3_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split2_20210405.log.json) |
| split3 | [pose_resnet_152](/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/horse10/res152_horse10_256x256-split3.py) |  256x256   |  0.957  | 0.131 | [ckpt](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3-c957dac5_20210405.pth) | [log](https://download.openmmlab.com/mmpose/animal/resnet/res152_horse10_256x256_split3_20210405.log.json) |
