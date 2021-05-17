<!-- [BACKBONE] -->

```bibtex
@inproceedings{sandler2018mobilenetv2,
  title={Mobilenetv2: Inverted residuals and linear bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4510--4520},
  year={2018}
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
| [pose_mobilenet_v2](/configs/hand/2D_Kpt_SView_RGB_Img/topdown_heatmap/rhd2d/mobilenetv2_rhd2d_256x256.py) | 256x256 | 0.985 | 0.883 | 2.80 | [ckpt](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_rhd2d_256x256-85fa02db_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/mobilenetv2/mobilenetv2_rhd2d_256x256_20210330.log.json) |
