<!-- [ALGORITHM] -->

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

<!-- [DATASET] -->

```bibtex
@inproceedings{andriluka14cvpr,
  author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
  title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2014},
  month = {June}
}
```

#### Results on MPII val set

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_hourglass_52](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/hourglass52_mpii_256x256.py) | 256x256 | 0.889 | 0.317 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_256x256-ae358435_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_256x256_20200812.log.json) |
| [pose_hourglass_52](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/hourglass52_mpii_384x384.py) | 384x384 | 0.894 | 0.366 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_384x384-04090bc3_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hourglass/hourglass52_mpii_384x384_20200812.log.json) |
