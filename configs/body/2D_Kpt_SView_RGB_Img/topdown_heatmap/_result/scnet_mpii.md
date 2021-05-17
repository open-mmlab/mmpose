<!-- [ALGORITHM] -->

```bibtex
@inproceedings{liu2020improving,
  title={Improving Convolutional Networks with Self-Calibrated Convolutions},
  author={Liu, Jiang-Jiang and Hou, Qibin and Cheng, Ming-Ming and Wang, Changhu and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10096--10105},
  year={2020}
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
| [pose_scnet_50](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/scnet50_mpii_256x256.py) | 256x256 | 0.888 | 0.290 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256-a54b6af5_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_mpii_256x256_20200812.log.json) |
| [pose_scnet_101](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/scnet101_mpii_256x256.py) | 256x256 | 0.886 | 0.293 | [ckpt](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256-b4c2d184_20200812.pth) | [log](https://download.openmmlab.com/mmpose/top_down/scnet/scnet101_mpii_256x256_20200812.log.json) |
