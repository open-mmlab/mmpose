<!-- [ALGORITHM] -->

```bibtex
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI}
  year={2019}
}
```

<!-- [DATASET] -->

```bibtex
@inproceedings{simon2017hand,
  title={Hand keypoint detection in single images using multiview bootstrapping},
  author={Simon, Tomas and Joo, Hanbyul and Matthews, Iain and Sheikh, Yaser},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={1145--1153},
  year={2017}
}
```

#### Results on CMU Panoptic (MPII+NZSL val set)

| Arch  | Input Size | PCKh@0.7 |  AUC  |  EPE  | ckpt    | log     |
| :--- | :--------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnetv2_w18](/configs/hand/2D_Kpt_SView_RGB_Img/top_down_heatmap/panoptic/hrnetv2_w18_panoptic_256x256.py) | 256x256 | 0.999 | 0.744 | 7.79 | [ckpt](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_panoptic_256x256-53b12345_20210330.pth) | [log](https://download.openmmlab.com/mmpose/hand/hrnetv2/hrnetv2_w18_panoptic_256x256_20210330.log.json) |
