<!-- [ALGORITHM] -->

<details>
<summary align="right"><a href="https://link.springer.com/chapter/10.1007/978-3-319-46484-8_29">Hourglass (ECCV'2016)</a></summary>

```bibtex
@inproceedings{newell2016stacked,
  title={Stacked hourglass networks for human pose estimation},
  author={Newell, Alejandro and Yang, Kaiyu and Deng, Jia},
  booktitle={European conference on computer vision},
  pages={483--499},
  year={2016},
  organization={Springer}
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
| [pose_hourglass_52](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hourglass52_mpii_256x256.py) | 256x256 | 0.889 | 0.317 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_256x256-ae358435_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_256x256_20200812.log.json) |
| [pose_hourglass_52](/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hourglass52_mpii_384x384.py) | 384x384 | 0.894 | 0.366 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_384x384-04090bc3_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_384x384_20200812.log.json) |
