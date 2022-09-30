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

<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html">DarkPose (CVPR'2020)</a></summary>

```bibtex
@inproceedings{zhang2020distribution,
  title={Distribution-aware coordinate representation for human pose estimation},
  author={Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={7093--7102},
  year={2020}
}
```

</details>

<!-- [DATASET] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2014/html/Andriluka_2D_Human_Pose_2014_CVPR_paper.html">MPII (CVPR'2014)</a></summary>

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt},
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

</details>

Results on MPII val set

| Arch                                                        | Input Size | Mean  | Mean@0.1 |                            ckpt                             |                             log                             |
| :---------------------------------------------------------- | :--------: | :---: | :------: | :---------------------------------------------------------: | :---------------------------------------------------------: |
| [pose_hrnet_w32_dark](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hrnet-w32_dark-8xb64-210e_mpii-256x256.py) |  256x256   | 0.904 |  0.354   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark-f1601c5b_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_dark_20200927.log.json) |
| [pose_hrnet_w48_dark](/configs/body_2d_keypoint/topdown_heatmap/mpii/td-hm_hrnet-w48_dark-8xb64-210e_mpii-256x256.py) |  256x256   | 0.905 |  0.360   | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark-0decd39f_20200927.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_dark_20200927.log.json) |
