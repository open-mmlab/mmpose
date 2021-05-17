<!-- [BACKBONE] -->

```bibtex
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
  year={2018}
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
| [pose_shufflenetv2](/configs/body/2D_Kpt_SView_RGB_Img/topdown_heatmap/mpii/shufflenetv2_mpii_256x256.py) | 256x256 | 0.828 | 0.205 | [ckpt](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_mpii_256x256-4fb9df2d_20200925.pth) | [log](https://download.openmmlab.com/mmpose/top_down/shufflenetv2/shufflenetv2_mpii_256x256_20200925.log.json) |
