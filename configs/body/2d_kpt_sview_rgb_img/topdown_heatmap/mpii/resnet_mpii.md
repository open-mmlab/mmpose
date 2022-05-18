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

<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html">ResNet (CVPR'2016)</a></summary>

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
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
| [pose_resnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/res50_mpii_256x256.py) |  256x256   | 0.882 |  0.286   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256-418ffc88_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res50_mpii_256x256_20200812.log.json) |
| [pose_resnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/res101_mpii_256x256.py) |  256x256   | 0.888 |  0.290   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256-416f5d71_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res101_mpii_256x256_20200812.log.json) |
| [pose_resnet_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/res152_mpii_256x256.py) |  256x256   | 0.889 |  0.303   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256-3ecba29d_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnet/res152_mpii_256x256_20200812.log.json) |
