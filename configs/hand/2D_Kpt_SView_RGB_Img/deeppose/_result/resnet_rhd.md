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
@TechReport{zb2017hand,
  author={Christian Zimmermann and Thomas Brox},
  title={Learning to Estimate 3D Hand Pose from Single RGB Images},
  institution={arXiv:1705.01389},
  year={2017},
  note="https://arxiv.org/abs/1705.01389",
  url="https://lmb.informatik.uni-freiburg.de/projects/hand3d/"
}
```

#### Results on RHD test set

| Arch  | Input Size | PCK@0.2 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [deeppose_resnet_50](/configs/hand/2D_Kpt_SView_RGB_Img/deeppose/rhd2d/deeppose_res50_rhd2d_256x256.py) | 256x256 | 0.988 | 0.865 | 3.29 | [ckpt](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_rhd2d_256x256-37f1c4d3_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/deeppose/deeppose_res50_rhd2d_256x256_20210330.log.json) |
