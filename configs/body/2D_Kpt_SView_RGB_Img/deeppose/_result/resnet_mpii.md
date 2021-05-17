<!-- [BACKBONE] -->

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
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
| [deeppose_resnet_50](/configs/body/2D_Kpt_SView_RGB_Img/deeppose/mpii/deeppose_res50_mpii_256x256.py) | 256x256 | 0.825 | 0.174 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256-c63cd0b6_20210203.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res50_mpii_256x256_20210203.log.json) |
| [deeppose_resnet_101](/configs/body/2D_Kpt_SView_RGB_Img/deeppose/mpii/deeppose_res101_mpii_256x256.py) | 256x256 | 0.841 | 0.193 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256-87516a90_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_mpii_256x256_20210205.log.json) |
| [deeppose_resnet_152](/configs/body/2D_Kpt_SView_RGB_Img/deeppose/mpii/deeppose_res152_mpii_256x256.py) | 256x256 | 0.850 | 0.198 | [ckpt](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256-15f5e6f9_20210205.pth) | [log](https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_mpii_256x256_20210205.log.json) |
