<!-- [BACKBONE] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html">ResNetV1D (CVPR'2019)</a></summary>

```bibtex
@inproceedings{he2019bag,
  title={Bag of tricks for image classification with convolutional neural networks},
  author={He, Tong and Zhang, Zhi and Zhang, Hang and Zhang, Zhongyue and Xie, Junyuan and Li, Mu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2019}
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
| [pose_resnetv1d_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/resnetv1d50_mpii_256x256.py) |  256x256   | 0.881 |  0.290   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_mpii_256x256-2337a92e_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d50_mpii_256x256_20200812.log.json) |
| [pose_resnetv1d_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/resnetv1d101_mpii_256x256.py) |  256x256   | 0.883 |  0.295   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_mpii_256x256-2851d710_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d101_mpii_256x256_20200812.log.json) |
| [pose_resnetv1d_152](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/resnetv1d152_mpii_256x256.py) |  256x256   | 0.888 |  0.300   | [ckpt](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_mpii_256x256-8b10a87c_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/resnetv1d/resnetv1d152_mpii_256x256_20200812.log.json) |
