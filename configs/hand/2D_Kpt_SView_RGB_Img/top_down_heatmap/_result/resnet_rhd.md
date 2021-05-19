<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2018simple,
  title={Simple baselines for human pose estimation and tracking},
  author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={466--481},
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
| [pose_hrnetv2_w18_udp](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/rhd2d/hrnetv2_w18_rhd2d_256x256_udp.py) | 256x256 | 0.992 | 0.902 | 2.21 | [ckpt](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_rhd2d_256x256_udp-63ba6007_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/udp/hrnetv2_w18_rhd2d_256x256_udp_20210330.log.json) |
