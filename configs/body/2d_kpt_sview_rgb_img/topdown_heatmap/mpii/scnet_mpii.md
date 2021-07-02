<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Improving_Convolutional_Networks_With_Self-Calibrated_Convolutions_CVPR_2020_paper.html">SCNet (CVPR'2020)</a></summary>

```bibtex
@inproceedings{liu2020improving,
  title={Improving Convolutional Networks with Self-Calibrated Convolutions},
  author={Liu, Jiang-Jiang and Hou, Qibin and Cheng, Ming-Ming and Wang, Changhu and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10096--10105},
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

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_scnet_50](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/scnet50_mpii_256x256.py) | 256x256 | 0.888 | 0.290 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256-a54b6af5_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256_20200812.log.json) |
| [pose_scnet_101](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/scnet101_mpii_256x256.py) | 256x256 | 0.886 | 0.293 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256-b4c2d184_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256_20200812.log.json) |
